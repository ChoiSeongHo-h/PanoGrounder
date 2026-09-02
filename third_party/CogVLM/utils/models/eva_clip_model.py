# Modified by the PanoGrounder Authors from CogVLM upstream.
import torch
import torch.nn as nn
import math
from sat.model.base_model import BaseModel
from sat.model.mixins import BaseMixin
from sat.model.official.vit_model import ViTProperty, InterpolatedPositionEmbeddingMixin, gelu
from sat import mpu
from .multiview_feature_model import MultiViewFeatureAdapter, PairwiseFusion
import torch.nn.functional as F

class IdentityMixin(BaseMixin):
    def __init__(self):
        super().__init__()

    def final_forward(self, logits, **kwargs):
        return logits[:, 1:]

import xformers.ops as xops
class XAttn(BaseMixin):
    def __init__(self, head_dim):
        super().__init__()
        self.scale = head_dim ** -0.5

    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
        dropout_p = 0. # xformers does not support dropout for eva hidden size

        query_layer = query_layer.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)

        out = xops.memory_efficient_attention(
            query_layer, key_layer, value_layer,
            p=dropout_p,
            scale=self.scale,
            )
        return out

    def attention_forward(self, hidden_states, mask, **kw_args):
        self = self.transformer.layers[kw_args['layer_id']].attention
        attention_fn = self.hooks['attention_fn']

        mixed_raw_layer = self.query_key_value(hidden_states)

        B, N, C = hidden_states.shape
        mixed_raw_layer = mixed_raw_layer.reshape(B, N, 3, self.num_attention_heads_per_partition, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
        query_layer, key_layer, value_layer = mixed_raw_layer[0], mixed_raw_layer[1], mixed_raw_layer[2]

        dropout_fn = self.attention_dropout if self.training else None

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.view(B, N, -1)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output

class XAttnRelPE(BaseMixin):
    def __init__(self, head_dim, num_attn_heads, property):
        super().__init__()
        self.pano_rel_pe_alpha = torch.nn.Parameter(torch.ones(num_attn_heads))  # [H]
        self.pano_rel_pe_beta = torch.nn.Parameter(torch.zeros(num_attn_heads))  # [H]
        self.scale = head_dim ** -0.5
        self.property = property
        self.num_heads = num_attn_heads

    def compute_lon_lat_positions(self, dtype, device):
        H, W = self.property.grid_size
        patch_size = self.property.patch_size
        img_h, img_w = self.property.image_size

        centers_x = (torch.arange(W, device=device, dtype=dtype) + 0.5) * patch_size
        centers_y = (torch.arange(H, device=device, dtype=dtype) + 0.5) * patch_size
        grid_y, grid_x = torch.meshgrid(centers_y, centers_x, indexing='ij')

        lon = 2 * math.pi * grid_x / img_w - math.pi
        lat = math.pi / 2 - math.pi * grid_y / img_h

        return torch.stack([lon, lat], dim=-1).view(-1, 2)  # (N, 2)

    def compute_rel_pe_distance(self, dtype, device):
        pano_coords = self.compute_lon_lat_positions(dtype, device)
        u = pano_coords[:, 0].unsqueeze(0)  # [1, N]
        v = pano_coords[:, 1].unsqueeze(0)  # [1, N]

        du = u - u.T
        dv = v - v.T

        sin2_dv = torch.sin(dv / 2) ** 2
        cos_vv = torch.cos(v) * torch.cos(v.T)
        sin2_du = torch.sin(du / 2) ** 2

        return sin2_dv + cos_vv * sin2_du  # [N, N]

    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                    attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
        dropout_p = 0.0

        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer   = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)

        B, N, H, C = query_layer.shape
        dtype = query_layer.dtype
        device = query_layer.device

        pre_len = self.transformer.property.pre_len
        post_len = self.transformer.property.post_len

        # Step 1: Compute great-circle distance only for real patches
        geo_dist = self.compute_rel_pe_distance(dtype, device)  # [N, N]

        # Step 2: Pad for pre/post tokens
        bias = geo_dist
        if pre_len > 0 or post_len > 0:
            bias = torch.nn.functional.pad(bias, (pre_len, post_len, pre_len, post_len), value=float('-inf'))  # [N+pre+post, N+pre+post]

        N_total = bias.shape[0]
        pad_len = (8 - (N_total % 8)) % 8

        # Step 3: Final padding to multiple of 8
        if pad_len > 0:
            bias = torch.nn.functional.pad(bias, (0, pad_len, 0, pad_len), value=float('-inf'))  # [N_pad, N_pad]

        # Step 4: Apply alpha, beta
        alpha = self.pano_rel_pe_alpha.view(H, 1, 1)  # [H, 1, 1]
        beta = self.pano_rel_pe_beta.view(H, 1, 1)
        # bias = alpha * bias + beta                    # [H, N_pad, N_pad]
        bias = bias.unsqueeze(0).expand(B, H, -1, -1)  # [B, H, N_pad, N_pad]

        # Step 5: q/k/v pad to match N_pad
        if pad_len > 0:
            pad_qkv = (0, 0, 0, 0, 0, pad_len)
            query_layer = torch.nn.functional.pad(query_layer, pad_qkv, value=0.0)
            key_layer   = torch.nn.functional.pad(key_layer, pad_qkv, value=0.0)
            value_layer = torch.nn.functional.pad(value_layer, pad_qkv, value=0.0)

        # Step 6: attention
        out = xops.memory_efficient_attention(
            query_layer, key_layer, value_layer,
            attn_bias=bias*0.01,
            p=dropout_p,
            scale=self.scale,
        )  # [B, N_pad, H, C]

        # Step 7: remove all pad
        out = out[:, :N_total, :, :]  # remove extra_pad only
        out = out.permute(0, 2, 1, 3)  # [B, H, N, C] → [B, N, H, C]
        return out

    def attention_forward(self, hidden_states, mask, **kw_args):
        self = self.transformer.layers[kw_args['layer_id']].attention
        attention_fn = self.hooks['attention_fn']

        mixed_raw_layer = self.query_key_value(hidden_states)

        B, N, C = hidden_states.shape
        mixed_raw_layer = mixed_raw_layer.reshape(B, N, 3, self.num_attention_heads_per_partition, -1).permute(2, 0, 3, 1, 4)
        query_layer, key_layer, value_layer = mixed_raw_layer[0], mixed_raw_layer[1], mixed_raw_layer[2]

        dropout_fn = self.attention_dropout if self.training else None

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.reshape(B, N, -1)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output

class NewLayerForward(BaseMixin):
    def __init__(self):
        super().__init__()

    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        self = self.transformer.layers[kw_args['layer_id']]

        # kw_args['layer_id']이 텐서라면 먼저 item()을 호출하여 정수로 변환
        current_layer_id_str = None
        if isinstance(kw_args['layer_id'], torch.Tensor):
            current_layer_id_str = str(kw_args['layer_id'].item())
        else:
            current_layer_id_str = str(kw_args['layer_id'])

        # 멀티뷰 피처를 ViT 레이어의 입력에 주입
        if ('multi_view_feature' in kw_args and
            kw_args['multi_view_feature'] is not None and
            hasattr(self.transformer, 'mv_adapters') and
            current_layer_id_str in self.transformer.mv_adapters):

            # 현재 레이어에 해당하는 어댑터를 가져옵니다.
            adapter = self.transformer.mv_adapters[current_layer_id_str]
            mv_control_signal = adapter(hidden_states[:, 1:], kw_args['multi_view_feature'])

            # 1. cls 토큰과 패치 토큰을 분리합니다.
            cls_token = hidden_states[:, :1]
            patch_tokens = hidden_states[:, 1:]

            # 2. 패치 토큰에만 control signal을 더하여 새로운 텐서를 생성합니다. (in-place가 아님)
            if hasattr(adapter, 'use_concat') and adapter.use_concat:
                updated_patch_tokens = mv_control_signal
            else:
                updated_patch_tokens = patch_tokens + mv_control_signal

            # 3. 분리했던 cls 토큰과 새로 계산된 패치 토큰을 다시 합칩니다.
            hidden_states = torch.cat([cls_token, updated_patch_tokens], dim=1)

        if ('xyz_raw_emb' in kw_args and
            kw_args['xyz_raw_emb'] is not None and
            hasattr(self.transformer, 'xyz_adapters') and
            current_layer_id_str in self.transformer.xyz_adapters):

            adapter = self.transformer.xyz_adapters[current_layer_id_str]
            xyz_control_signal = adapter(hidden_states[:, 1:], kw_args['xyz_raw_emb'])
            cls_token = hidden_states[:, :1]
            patch_tokens = hidden_states[:, 1:]
            updated_patch_tokens = patch_tokens + xyz_control_signal

            hidden_states = torch.cat([cls_token, updated_patch_tokens], dim=1)

        attention_input = hidden_states

        # Self attention.
        attention_output = self.input_layernorm(self.attention(attention_input, mask, **kw_args))

        # DropPath for attention
        if self.training and self.drop_path > 0.:
            if mpu.get_cuda_rng_tracker is not None:
                # drop_path must use model parallel rng tracker
                # the tracker is initialized as seed of `seed + model_parallel_rank`
                # deepspeed act-ckpt record the model parallel tracker states
                with mpu.get_cuda_rng_tracker().fork():
                    # drop_path percentage 0, others 1/(1-p)
                    random_tensor = (1-self.drop_path
                                    + torch.rand((attention_output.shape[0],), dtype=attention_output.dtype, device=attention_output.device)).floor_() / (1-self.drop_path)
                    attention_output = random_tensor.view(-1, 1, 1) * attention_output

        # Residual connection.
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states

        # MLP.
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input, **kw_args))

        # DropPath for mlp
        if self.training and self.drop_path > 0.:
            if mpu.get_cuda_rng_tracker is not None:
                with mpu.get_cuda_rng_tracker().fork():
                    random_tensor = (1-self.drop_path
                                    + torch.rand((mlp_output.shape[0],), dtype=mlp_output.dtype, device=mlp_output.device)).floor_() / (1-self.drop_path)
                    mlp_output = random_tensor.view(-1, 1, 1) * mlp_output

        # Second residual connection.
        output = mlp_input + mlp_output

        return output

class InterpolatedPanoPositionEmbeddingMixin(BaseMixin):
    def __init__(self, hidden_size, property):
        super().__init__()
        self.pano_abs_pe_linear = torch.nn.Linear(3, hidden_size)
        self.pano_abs_pe_alpha = torch.nn.Parameter(torch.tensor(0.0))
        self.hidden_size = hidden_size
        self.property = property

    def compute_spherical_positions(self, property, dtype, device):
        H, W = property.grid_size
        patch_size = property.patch_size
        img_h, img_w = property.image_size

        centers_x = (torch.arange(W, device=device, dtype=dtype) + 0.5) * patch_size
        centers_y = (torch.arange(H, device=device, dtype=dtype) + 0.5) * patch_size
        grid_y, grid_x = torch.meshgrid(centers_y, centers_x, indexing='ij')

        lon = 2 * math.pi * grid_x / img_w - math.pi        # [-π, π]
        lat = math.pi / 2 - math.pi * grid_y / img_h        # [π/2, -π/2]

        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)

        xyz = torch.stack([x, y, z], dim=-1)    # (H, W, 3)
        return xyz.view(-1, 3)                  # (num_patches, 3)

    def position_embedding_forward(self, position_ids, **kwargs):
        # 기본 positional embedding
        base_pe = self.transformer.position_embeddings.weight.unsqueeze(0)  # (1, seq_len, hidden)

        dtype = self.pano_abs_pe_linear.weight.dtype
        device = self.pano_abs_pe_linear.weight.device
        spherical_xyz = self.compute_spherical_positions(self.property, dtype, device)  # (num_patches, 3)

        pano_abs_embedding = self.pano_abs_pe_linear(spherical_xyz)  # (num_patches, hidden)

        # pre/post padding
        pre_len = self.transformer.property.pre_len
        post_len = self.transformer.property.post_len

        zeros_pre = torch.zeros(pre_len, self.hidden_size, device=device, dtype=dtype)
        zeros_post = torch.zeros(post_len, self.hidden_size, device=device, dtype=dtype)

        pano_abs_embedding = torch.cat([zeros_pre, pano_abs_embedding, zeros_post], dim=0)  # (seq_len, hidden)
        pano_abs_embedding = pano_abs_embedding.unsqueeze(0)

        return base_pe + self.pano_abs_pe_alpha * pano_abs_embedding  # Automatically broadcasted into batch_size in the transformer

    def reinit(self, parent_model=None, property=None):
        """
        new pre_len, new num_patches and new post_len should all be larger or equal than the old ones.
        """
        assert property is not None
        old_weight = self.transformer.position_embeddings.weight.data
        pre_weight = old_weight[:self.transformer.property.pre_len]
        post_weight = old_weight[self.transformer.property.pre_len+self.transformer.property.num_patches:]
        image_weight = old_weight[self.transformer.property.pre_len:self.transformer.property.pre_len+self.transformer.property.num_patches].reshape(1, self.transformer.property.grid_size[0], self.transformer.property.grid_size[1], -1).permute(0, 3, 1, 2)
        original_dtype = image_weight.dtype
        image_weight = torch.nn.functional.interpolate(image_weight.float(), size=property.grid_size, mode='bicubic', align_corners=False).permute(0, 2, 3, 1).reshape(property.num_patches, -1).to(original_dtype)
        self.transformer.position_embeddings = torch.nn.Embedding(property.seq_len, old_weight.shape[1]).type(old_weight.dtype).to(old_weight.device)
        torch.nn.init.normal_(self.transformer.position_embeddings.weight, mean=0.0, std=0.02)
        self.transformer.position_embeddings.weight.data[:self.transformer.property.pre_len] = pre_weight
        self.transformer.position_embeddings.weight.data[property.pre_len:property.pre_len+property.num_patches] = image_weight
        self.transformer.position_embeddings.weight.data[property.pre_len+property.num_patches:property.pre_len+property.num_patches+self.transformer.property.post_len] = post_weight
        self.transformer.property = property

class ResContextBlock(nn.Module):
    # From T. Cortinhal et al.
    # https://github.com/TiagoCortinhal/SalsaNext
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output

class ResBlock(nn.Module):
    # From T. Cortinhal et al.
    # https://github.com/TiagoCortinhal/SalsaNext
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1, resA2, resA3), dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA

        # pooling=False로 ResBlock을 초기화했으므로 이 부분은 실행되지 않음
        # 제공된 코드에서는 pooling=False이므로 ResBlock의 forward는 resB만 반환
        return resA

class ConvStem(nn.Module):
    def __init__(self,
                 in_channels=5,
                 base_channels=32,
                 img_size=(490, 490),
                 patch_stride=(14, 14),
                 embed_dim=384,
                 flatten=True,
                 hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 2 * base_channels

        self.base_channels = base_channels
        self.dropout_ratio = 0.2

        # Build stem, similar to the design in https://github.com/TiagoCortinhal/SalsaNext
        self.conv_block = nn.Sequential(
            ResContextBlock(in_channels, base_channels),
            ResContextBlock(base_channels, base_channels),
            ResContextBlock(base_channels, base_channels),
            ResBlock(base_channels, hidden_dim, self.dropout_ratio, pooling=False, drop_out=False))

        assert patch_stride[0] % 2 == 0
        assert patch_stride[1] % 2 == 0
        kernel_size = (patch_stride[0] + 1, patch_stride[1] + 1)
        padding = (patch_stride[0] // 2, patch_stride[1] // 2)
        self.proj_block = nn.Sequential(
             nn.AvgPool2d(kernel_size=kernel_size, stride=patch_stride, padding=padding),
             nn.Conv2d(hidden_dim, embed_dim, kernel_size=1))

        self.patch_stride = patch_stride
        self.patch_size = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

    def forward(self, x):
        B, C, H, W = x.shape  # B, in_channels, image_size[0], image_size[1]
        x_base = self.conv_block(x) # B, hidden_dim, image_size[0], image_size[1]
        x = self.proj_block(x_base)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x

class ImagePatchEmbeddingMixin(BaseMixin):
    def __init__(self, in_channels, hidden_size, property, use_rangevit=False, use_resize_xyz=False):
        super(ImagePatchEmbeddingMixin, self).__init__()
        self.use_rangevit = use_rangevit

        if self.use_rangevit:
            # ConvStem 사용.
            if use_resize_xyz:
                self.proj = ConvStem(
                    in_channels=in_channels,
                    # base_channels=32,
                    base_channels=16,
                    # base_channels=8,
                    img_size=(280, 280),
                    patch_stride=(8, 8), # ConvStem에서는 patch_size가 stride 역할을 함
                    embed_dim=hidden_size,
                    flatten=True,
                    # hidden_dim=256
                )
            else:
                self.proj = ConvStem(
                    in_channels=in_channels,
                    # base_channels=32,
                    # base_channels=16,
                    base_channels=8,
                    img_size=(490, 490),
                    patch_stride=(property.patch_size, property.patch_size), # ConvStem에서는 patch_size가 stride 역할을 함
                    embed_dim=hidden_size,
                    flatten=True,
                    # hidden_dim=256
                )
        else:
            # 기존의 간단한 Conv2d 프로젝션 방식
            self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=property.patch_size, stride=property.patch_size)

    def word_embedding_forward(self, input_ids, **kwargs):
        """
        Input:
        * input_ids with shape (batch_size, pre_len+post_len)
        * kwargs["image"] with shape (B, C, H, W)
        Output:
        * (batch_size, hidden_size)
        """
        images = kwargs["image"]

        if self.use_rangevit:
            # ConvStem
            image_embeddings = self.proj(images)
        else:
            # 기존 방식: Conv2d -> flatten -> transpose
            image_embeddings = self.proj(images)
            image_embeddings = image_embeddings.flatten(2).transpose(1, 2)

        pre_word_embeddings = self.transformer.word_embeddings(input_ids[:,:self.transformer.property.pre_len])
        post_word_embeddings = self.transformer.word_embeddings(input_ids[:,self.transformer.property.pre_len:self.transformer.property.pre_len+self.transformer.property.post_len])
        embeddings = torch.cat([pre_word_embeddings, image_embeddings, post_word_embeddings], dim=1)
        return embeddings

class EVA2CLIPModel(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        property = ViTProperty(args.image_size, args.patch_size, args.pre_len, args.post_len)
        args.max_sequence_length = property.pre_len + property.num_patches + property.post_len
        if 'activation_func' not in kwargs:
            kwargs['activation_func'] = gelu
        super().__init__(args, transformer=transformer, **kwargs)
        self.transformer.property = property
        use_rangevit = args.use_rangevit if hasattr(args, 'use_rangevit') else False
        use_resize_xyz = args.use_resize_xyz if hasattr(args, 'use_resize_xyz') else False
        self.add_mixin("patch_embedding", ImagePatchEmbeddingMixin(args.in_channels, args.hidden_size, property, use_rangevit=use_rangevit, use_resize_xyz=use_resize_xyz))

        if "pano_abs_pe" in args.pano_pe_types:
            self.add_mixin("pos_embedding", InterpolatedPanoPositionEmbeddingMixin(args.hidden_size, property))
        else:
            self.add_mixin("pos_embedding", InterpolatedPositionEmbeddingMixin())

        self.add_mixin("final", IdentityMixin())
        self.add_mixin("newpost", NewLayerForward())

        if "pano_rel_pe" in args.pano_pe_types:
            self.add_mixin("xattn", XAttnRelPE(args.hidden_size // args.num_attention_heads, args.num_attention_heads, property))
        else:
            self.add_mixin("xattn", XAttn(args.hidden_size // args.num_attention_heads))

        # ViT 레이어별 어댑터 초기화 (후반 1/3만)
        if hasattr(args, 'use_multi_view_feature') and args.use_multi_view_feature:
            if hasattr(args, 'use_fusion') and args.use_fusion:
                start_layer = args.mvf_adapter_start if hasattr(args, 'mvf_adapter_start') else 42
                end_layer = args.mvf_adapter_end if hasattr(args, 'mvf_adapter_end') else 63

                fusion_dropout = 0.0
                if hasattr(args, 'fusion_dropout'):
                    fusion_dropout = args.fusion_dropout

                fusion_intermediate_dim = None
                if hasattr(args, 'fusion_intermediate_dim'):
                    fusion_intermediate_dim = args.fusion_intermediate_dim

                debug_print = False
                if hasattr(args, 'debug_print'):
                    debug_print = args.debug_print

                detach_standard_grad = False
                if hasattr(args, 'detach_standard_grad'):
                    detach_standard_grad = args.detach_standard_grad

                control_only = False
                if hasattr(args, 'control_only'):
                    control_only = args.control_only

                wo_zero = False
                if hasattr(args, 'wo_zero'):
                    wo_zero = args.wo_zero

                use_concat = False
                if hasattr(args, 'use_concat'):
                    use_concat = args.use_concat


                # nn.ModuleDict를 사용하여 특정 레이어에만 어댑터를 생성
                self.mv_adapters = nn.ModuleDict({
                    str(i): PairwiseFusion(
                        standard_dim=args.hidden_size,
                        control_dim=args.multi_view_feature_dim,
                        dropout_p=fusion_dropout,
                        intermediate_dim=fusion_intermediate_dim,
                        name=f"Layer{i}_MVF",
                        debug_print=debug_print,
                        detach_standard_grad=detach_standard_grad,
                        control_only=control_only,
                        wo_zero=wo_zero,
                        use_concat=use_concat,
                    ) for i in range(start_layer, end_layer)
                })

                # 생성된 어댑터 딕셔너리를 transformer 모듈에 등록
                self.transformer.mv_adapters = self.mv_adapters

        if hasattr(args, 'use_xyz_encoder') and args.use_xyz_encoder:
            if hasattr(args, 'use_deep_xyz') and args.use_deep_xyz:
                start_layer = args.xyz_adapter_start if hasattr(args, 'xyz_adapter_start') else 21
                end_layer = args.xyz_adapter_end if hasattr(args, 'xyz_adapter_end') else 42

                debug_print = False
                if hasattr(args, 'debug_print'):
                    debug_print = args.debug_print

                detach_standard_grad = False
                if hasattr(args, 'detach_standard_grad'):
                    detach_standard_grad = args.detach_standard_grad

                control_only = False
                if hasattr(args, 'control_only'):
                    control_only = args.control_only

                wo_zero = False
                if hasattr(args, 'wo_zero'):
                    wo_zero = args.wo_zero

                use_concat = False
                if hasattr(args, 'use_concat'):
                    use_concat = args.use_concat

                self.xyz_adapters = nn.ModuleDict({
                    str(i): PairwiseFusion(
                        standard_dim=args.hidden_size,
                        control_dim=args.xyz_encoder_dim,
                        name=f"Layer{i}_XYZ",
                        debug_print=debug_print,
                        detach_standard_grad=detach_standard_grad,
                        control_only=control_only,
                        wo_zero = wo_zero,
                        use_concat=use_concat,
                    ) for i in range(start_layer, end_layer)
                })

                self.transformer.xyz_adapters = self.xyz_adapters

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('EVA2CLIP', 'EVA2CLIP Configurations')
        group.add_argument('--image-size', nargs='+', type=int, default=[224, 224])
        group.add_argument('--pre-len', type=int, default=1) # [cls] by default
        group.add_argument('--post-len', type=int, default=0) # empty by default, but sometimes with special tokens, such as [det] in yolos.
        group.add_argument('--in-channels', type=int, default=3)
        group.add_argument('--patch-size', type=int, default=16)
        return parser
