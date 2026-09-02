from sat.model.official.llama_model import LLaMAModel
import json
# Modified by the PanoGrounder Authors from CogVLM upstream.
import torch
from sat.model.base_model import BaseMixin
import torch.nn as nn
from .mixin import LlamaVisionExpertFCMixin, LlamaVisionExpertAttnMixin
import math
from transformers import AutoModel

from sat.resources.urls import MODEL_URLS
from .multiview_feature_model import MultiViewFeatureAdapter, PairwiseFusion, SemanticHead, RangeHead, XYZHead, DepthAwareFeatureFusion

MODEL_URLS["cogvlm-base-224"] = "r2://cogvlm-base-224.zip"
MODEL_URLS["cogvlm-base-490"] = "r2://cogvlm-base-490.zip"
MODEL_URLS["cogvlm-chat-v1.1"] = "r2://cogvlm-chat-v1.1.zip"
MODEL_URLS["cogvlm-grounding-base"] = "r2://cogvlm-grounding-base.zip"
MODEL_URLS["cogvlm-grounding-generalist-v1.1"] = "r2://cogvlm-grounding-generalist-v1.1.zip"


class GLU(nn.Module):
    def __init__(self, args, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, args.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(args.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(args.hidden_size, args.inner_hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.inner_hidden_size, bias=False)
        self.dense_4h_to_h = nn.Linear(args.inner_hidden_size, args.hidden_size, bias=False)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x

from .eva_clip_model import EVA2CLIPModel
import argparse
from copy import deepcopy
def override_dist_dtype_device_args(args, b={}):
    if args.mode == 'inference':
        minimal_args = argparse.Namespace(
            world_size=args.world_size,
            rank=args.rank,
            local_rank=args.local_rank,
            skip_init=args.skip_init,
            use_gpu_initialization=args.use_gpu_initialization,
            deepspeed=args.deepspeed,
            bf16=args.bf16,
            fp16=args.fp16,
            mode=args.mode,
            device=args.device
        )
    else:
        minimal_args = argparse.Namespace(
                world_size=args.world_size,
                rank=args.rank,
                local_rank=args.local_rank,
                skip_init=args.skip_init,
                use_gpu_initialization=args.use_gpu_initialization,
                deepspeed=args.deepspeed,
                bf16=args.bf16,
                fp16=args.fp16,
                mode=args.mode,
                checkpoint_activations=args.checkpoint_activations if not hasattr(args, 'vit_checkpoint_activations') else args.vit_checkpoint_activations,
                checkpoint_num_layers=args.checkpoint_num_layers,
                device=args.device,
                hidden_dropout=0.,
                attention_dropout=0.,
            )
    if hasattr(args, 'model_parallel_size'):
        b['model_parallel_size'] = args.model_parallel_size
    return argparse.Namespace(**deepcopy(b), **vars(minimal_args))

class ImageMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        self.args = args
        vit_args = override_dist_dtype_device_args(args, args.eva_args)
        xyz_encoder_args = vit_args

        vit_args.pano_pe_types = args.pano_pe_types
        vit_args.use_multi_view_feature = args.use_multi_view_feature if hasattr(args, 'use_multi_view_feature') else False
        vit_args.multi_view_feature_dim = args.multi_view_feature_dim if hasattr(args, 'multi_view_feature_dim') else 384
        vit_args.normalize_feature = args.normalize_feature if hasattr(args, 'normalize_feature') else False
        vit_args.use_cnn = args.use_cnn if hasattr(args, 'use_cnn') else False
        vit_args.mv_cnn_mid_dim = args.mv_cnn_mid_dim if hasattr(args, 'mv_cnn_mid_dim') else 512
        vit_args.use_fusion = args.use_fusion if hasattr(args, 'use_fusion') else False
        vit_args.use_deep_feature = args.use_deep_feature if hasattr(args, 'use_deep_feature') else False
        vit_args.fusion_dropout = args.fusion_dropout if hasattr(args, 'fusion_dropout') else 0.0
        vit_args.fusion_intermediate_dim = args.fusion_intermediate_dim if hasattr(args, 'fusion_intermediate_dim') else None
        vit_args.use_xyz_encoder = args.use_xyz_encoder if hasattr(args, 'use_xyz_encoder') else False
        vit_args.use_deep_xyz = args.use_deep_xyz if hasattr(args, 'use_deep_xyz') else False
        vit_args.xyz_encoder_dim = args.xyz_encoder_dim if hasattr(args, 'xyz_encoder_dim') else 384
        vit_args.debug_print = args.debug_print if hasattr(args, 'debug_print') else False
        vit_args.detach_standard_grad = args.detach_standard_grad if hasattr(args, 'detach_standard_grad') else False
        vit_args.xyz_adapter_start = args.xyz_adapter_start if hasattr(args, 'xyz_adapter_start') else 21
        vit_args.xyz_adapter_end = args.xyz_adapter_end if hasattr(args, 'xyz_adapter_end') else 42
        vit_args.mvf_adapter_start = args.mvf_adapter_start if hasattr(args, 'mvf_adapter_start') else 42
        vit_args.mvf_adapter_end = args.mvf_adapter_end if hasattr(args, 'mvf_adapter_end') else 63
        vit_args.control_only = args.control_only if hasattr(args, 'control_only') else False
        vit_args.wo_zero = args.wo_zero if hasattr(args, 'wo_zero') else False
        vit_args.use_concat = args.use_concat if hasattr(args, 'use_concat') else False
        self.vit_model = EVA2CLIPModel(EVA2CLIPModel.get_args(**vars(vit_args)))

        self.in_features = 1792
        self.linear_proj = GLU(args, self.in_features)
        self.image_length = args.image_length
        self.boi = nn.Parameter(torch.zeros(1, 1, args.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, args.hidden_size))
        self.latest_vit_features = None
        self.latest_range_features = None

        if hasattr(args, 'use_xyz_encoder') and args.use_xyz_encoder:
            self.xyz_encoder = None
            if hasattr(args, 'use_range_dino') and args.use_range_dino:
                self.xyz_encoder = AutoModel.from_pretrained(
                    "facebook/dinov3-vits16plus-pretrain-lvd1689m"
                ).to(args.device)
            else:
                xyz_encoder_args.num_layers = 12
                xyz_encoder_args.hidden_size = args.xyz_encoder_dim if hasattr(args, 'xyz_encoder_dim') else 384
                xyz_encoder_args.num_attention_heads = xyz_encoder_args.hidden_size // 64
                xyz_encoder_args.inner_hidden_size = xyz_encoder_args.hidden_size * 4
                xyz_encoder_args.debug_print = args.debug_print if hasattr(args, 'debug_print') else False

                xyz_encoder_args.use_xyz_encoder = False
                xyz_encoder_args.use_deep_xyz = False
                xyz_encoder_args.use_xyz_aug = args.use_xyz_aug if hasattr(args, 'use_xyz_aug') else False
                xyz_encoder_args.xyz_aug_num = args.xyz_aug_num if hasattr(args, 'xyz_aug_num') else 8

                xyz_encoder_args.use_multi_view_feature = False
                xyz_encoder_args.use_fusion = False
                xyz_encoder_args.use_deep_feature = False

                use_fourier_features = args.use_fourier_features if hasattr(args, 'use_fourier_features') else False
                fourier_feature_dim = args.fourier_feature_dim if hasattr(args, 'fourier_feature_dim') else 6

                use_rangevit = args.use_rangevit if hasattr(args, 'use_rangevit') else False
                xyz_encoder_args.use_rangevit = use_rangevit

                use_resize_xyz = args.use_resize_xyz if hasattr(args, 'use_resize_xyz') else False
                xyz_encoder_args.use_resize_xyz = use_resize_xyz

                xyz_encoder_args.in_channels = 4
                if use_fourier_features:
                    xyz_encoder_args.in_channels += fourier_feature_dim * 6
                if use_rangevit:
                    xyz_encoder_args.in_channels += 1

                self.xyz_encoder = EVA2CLIPModel(EVA2CLIPModel.get_args(**vars(xyz_encoder_args)))

            if not hasattr(args, 'use_deep_xyz') or hasattr(args, 'use_deep_xyz') and not args.use_deep_xyz:
                self.xyz_adapter = nn.Linear(xyz_encoder_args.hidden_size, args.hidden_size)
                # 어댑터의 가중치와 편향을 0으로 초기화
                with torch.no_grad(): # 그래디언트 계산을 비활성화하여 효율적으로 처리
                    nn.init.zeros_(self.xyz_adapter.weight)
                    nn.init.zeros_(self.xyz_adapter.bias)


        # [추가] 멀티뷰 피처 어댑터 초기화
        if hasattr(args, 'use_multi_view_feature') and args.use_multi_view_feature:
            if hasattr(args, 'pre_insert_feature') and args.pre_insert_feature:
                output_dim = self.in_features
            else:
                output_dim = args.hidden_size

            if hasattr(args, 'use_fusion') and args.use_fusion:
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

                self.mv_adapter = PairwiseFusion(
                    standard_dim=output_dim,
                    control_dim=args.multi_view_feature_dim,
                    dropout_p=fusion_dropout,
                    intermediate_dim=fusion_intermediate_dim,
                    name="MVF",
                    debug_print=debug_print,
                    detach_standard_grad=detach_standard_grad,
                    control_only=args.control_only if hasattr(args, 'control_only') else False,
                    wo_zero=args.wo_zero if hasattr(args, 'wo_zero') else False,
                    use_concat=args.use_concat if hasattr(args, 'use_concat') else False,
                )
            else:
                self.mv_adapter = MultiViewFeatureAdapter(
                    args=args,
                    input_dim=args.multi_view_feature_dim,
                    output_dim=output_dim
                )

    def word_embedding_forward(self, input_ids, output_cross_layer, **kw_args):
        vision_inputs = {}
        for k in kw_args:
            if k.startswith('vision_') and k != 'vision_expert_mask':
                vision_inputs[k[7:]] = kw_args[k]
        # 멀티뷰 피처를 vision_inputs에도 추가
        if 'multi_view_feature' in kw_args:
            vision_inputs['multi_view_feature'] = kw_args['multi_view_feature']
        if input_ids.shape[1] == 1 or not vision_inputs:
            return self.transformer.word_embeddings(input_ids)

        xyz_raw_emb = None
        if hasattr(self, 'xyz_encoder') and 'input_geometry' in kw_args:
            # B와 N을 가져옵니다.
            B, N, H, W, C = kw_args['input_geometry'].shape

            if hasattr(self.args, 'use_range_dino') and self.args.use_range_dino:
                range_input = kw_args['input_geometry'][..., :4].reshape(B * N, H, W, C).permute(0, 3, 1, 2)
                dino_output = self.xyz_encoder(pixel_values=range_input)
                last_hidden_state = dino_output.last_hidden_state
                xyz_raw_emb = last_hidden_state[:, 5:, :]
                self.latest_range_features = xyz_raw_emb
            else:
                # 결과를 저장할 딕셔너리를 초기화합니다.
                vision_inputs_xyz = {}

                # 1. input_ids와 attention_mask를 (B, 1) -> (B*N, 1)으로 확장합니다.
                # 각 배치 샘플에 N개의 이미지가 있으므로, 텍스트 입력도 N번 반복해줍니다.
                vision_inputs_xyz['input_ids'] = vision_inputs['input_ids'].unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
                vision_inputs_xyz['attention_mask'] = vision_inputs['attention_mask'].unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

                # 2. 이미지 텐서의 차원을 (B, N, H, W, C) -> (B*N, C, H, W)로 변경합니다.
                # 먼저 B와 N 차원을 합치고, 그 다음 C(채널) 차원을 앞으로 가져옵니다.
                input_geometry_reshaped = kw_args['input_geometry'].reshape(B * N, H, W, C)
                vision_inputs_xyz['image'] = input_geometry_reshaped.permute(0, 3, 1, 2)

                vision_inputs_xyz['position_ids'] = vision_inputs['position_ids']
                xyz_raw_emb = self.xyz_encoder(**vision_inputs_xyz)[0]

            _, L, D = xyz_raw_emb.shape

            # (B*N, SeqLen, EmbDim) -> (B, N, SeqLen, EmbDim)
            # [128, 1225, 384] -> [16, 8, 1225, 384]
            reshaped_emb = xyz_raw_emb.view(B, N, L, D)

            # 2. N 차원(dim=1)에 대해 평균을 계산하여 정보를 집약합니다.
            # 이 연산을 통해 N 차원이 사라집니다.
            # (B, N, SeqLen, EmbDim) -> (B, SeqLen, EmbDim)
            # [16, 8, 1225, 384] -> [16, 1225, 384]
            aggregated_emb = reshaped_emb.mean(dim=1)

            vision_inputs['xyz_raw_emb'] = aggregated_emb

        image_emb = self.vit_model(**vision_inputs)[0]
        self.latest_vit_features = image_emb

        # [수정] 멀티뷰 피처 처리 및 융합 로직
        if hasattr(self, 'mv_adapter'):
            multi_view_feature = kw_args['multi_view_feature']

            # pre_insert_feature 플래그에 따라 분기
            if hasattr(self.args, 'pre_insert_feature') and self.args.pre_insert_feature:
                # Case 1: linear_proj 전에 융합
                mv_control_signal = self.mv_adapter(image_emb, multi_view_feature)
                if hasattr(self.args, 'use_concat') and self.args.use_concat:
                    image_emb = mv_control_signal
                else:
                    image_emb = image_emb + mv_control_signal
                image_emb = self.linear_proj(image_emb)
            else:
                # Case 2: linear_proj 후에 융합 (기존 방식)
                image_emb = self.linear_proj(image_emb)
                mv_control_signal = self.mv_adapter(image_emb, multi_view_feature)
                if hasattr(self.mv_adapter, 'use_concat') and self.mv_adapter.use_concat:
                    image_emb = mv_control_signal
                else:
                    image_emb = image_emb + mv_control_signal
        else:
            # 멀티뷰 어댑터가 없는 경우, 기존 linear_proj만 수행
            image_emb = self.linear_proj(image_emb)

        if hasattr(self, 'xyz_encoder') and 'input_geometry' in kw_args and xyz_raw_emb is not None and not self.args.use_deep_xyz:
            # 단순 선형 변환 사용
            xyz_emb = self.xyz_adapter(xyz_raw_emb)
            image_emb = image_emb + xyz_emb

            if hasattr(self.args, 'debug_print') and self.args.debug_print:
                print("xyz_emb.mean():", xyz_emb.mean().item())
                print("xyz_emb.min():", xyz_emb.min().item())
                print("xyz_emb.max():", xyz_emb.max().item())

        image_embed_mask = kw_args['image_embed_mask']
        word_embedding = self.transformer.word_embeddings(input_ids).clone()
        word_embedding[image_embed_mask.bool()] = torch.cat([self.boi.repeat(len(image_emb), 1, 1), image_emb, self.eoi.repeat(len(image_emb), 1, 1)], dim=1).reshape(-1, image_emb.shape[-1])
        return word_embedding.contiguous()


class CogVLMModel(LLaMAModel):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer=transformer, **kwargs)
        self.image_length = args.image_length
        self.add_mixin("eva", ImageMixin(args))
        self.del_mixin("mlp")
        self.add_mixin("mlp", LlamaVisionExpertFCMixin(args.hidden_size, args.inner_hidden_size, args.num_layers, 32))
        self.del_mixin("rotary")
        self.add_mixin("rotary", LlamaVisionExpertAttnMixin(args.hidden_size, args.num_attention_heads, args.num_layers, 32))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CogVLM', 'CogVLM Configurations')
        group.add_argument('--image_length', type=int, default=256)
        group.add_argument('--eva_args', type=json.loads, default={})
        return super().add_model_specific_args(parser)

    def forward(self, input_ids, vision_expert_mask, image_embed_mask, **kwargs):
        if input_ids.shape[1] > 1:
            return super().forward(input_ids=input_ids, vision_expert_mask=vision_expert_mask, image_embed_mask=image_embed_mask, output_hidden_states=True, **kwargs)
        return super().forward(input_ids=input_ids, output_hidden_states=True, **kwargs)


class FineTuneTrainCogVLMModel(CogVLMModel):
    def __init__(self, args, transformer=None, **kw_args):
        super().__init__(args, transformer=transformer, **kw_args)
        self.args = args
        # If you want to use model parallel with a mp_size=1 checkpoint, and meanwhile you also want to use lora,
        # you have to add_mixin after loading model checkpoint.

        if hasattr(args, 'use_semantic_head') and args.use_semantic_head:
            # ViT 특징 차원(1792)을 입력으로 받음
            vit_feature_dim = 1792
            self.semantic_head = SemanticHead(
                input_dim=vit_feature_dim,
                num_classes=args.num_semantic_classes
            )

        if hasattr(args, 'use_range_head') and args.use_range_head:
            # ViT 특징 차원(1792)을 입력으로 받음
            vit_feature_dim = 1792
            self.range_head = RangeHead(
                input_dim=vit_feature_dim
            )

        if hasattr(args, 'use_xyz_head') and args.use_xyz_head:
            # ViT 특징 차원(1792)을 입력으로 받음
            vit_feature_dim = 1792
            self.xyz_head = XYZHead(
                input_dim=vit_feature_dim
            )

        if hasattr(args, 'use_relative_range_head') and args.use_relative_range_head:
            self.depth_diff_head = DepthAwareFeatureFusion(input_dim=1792, use_tanh=True)

            if hasattr(args, 'use_intermediate_relative_range_head') and args.use_intermediate_relative_range_head:
                self.intermediate_depth_diff_head = DepthAwareFeatureFusion(input_dim=384, use_tanh=True)

            if hasattr(args, 'use_language_relative_range_head') and args.use_language_relative_range_head:
                self.language_depth_diff_head = DepthAwareFeatureFusion(input_dim=4096, use_tanh=True)


            if hasattr(args, 'use_relative_xyz_head') and args.use_relative_xyz_head:
                self.x_diff_head = DepthAwareFeatureFusion(input_dim=1792, use_tanh=True)
                self.y_diff_head = DepthAwareFeatureFusion(input_dim=1792, use_tanh=True)
                self.z_diff_head = DepthAwareFeatureFusion(input_dim=1792, use_tanh=True)

                if hasattr(args, 'use_intermediate_relative_xyz_head') and args.use_intermediate_relative_xyz_head:
                    self.intermediate_x_diff_head = DepthAwareFeatureFusion(input_dim=384, use_tanh=True)
                    self.intermediate_y_diff_head = DepthAwareFeatureFusion(input_dim=384, use_tanh=True)
                    self.intermediate_z_diff_head = DepthAwareFeatureFusion(input_dim=384, use_tanh=True)

                if hasattr(args, 'use_language_relative_xyz_head') and args.use_language_relative_xyz_head:
                    self.language_x_diff_head = DepthAwareFeatureFusion(input_dim=4096, use_tanh=True)
                    self.language_y_diff_head = DepthAwareFeatureFusion(input_dim=4096, use_tanh=True)
                    self.language_z_diff_head = DepthAwareFeatureFusion(input_dim=4096, use_tanh=True)



    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CogVLM-finetune', 'CogVLM finetune Configurations')
        group.add_argument('--pre_seq_len', type=int, default=8)
        group.add_argument('--lora_rank', type=int, default=10)
        group.add_argument('--use_ptuning', action="store_true")
        group.add_argument('--use_lora', action="store_true")
        group.add_argument('--use_qlora', action="store_true")
        group.add_argument('--layer_range', nargs='+', type=int, default=None)
        return super().add_model_specific_args(parser)


from sat.model.finetune import PTuningV2Mixin
from sat.model.finetune.lora2 import LoraMixin
class FineTuneTestCogVLMModel(CogVLMModel):
    def __init__(self, args, transformer=None, **kw_args):
        super().__init__(args, transformer=transformer, **kw_args)
        if args.use_ptuning:
            self.add_mixin("ptuning", PTuningV2Mixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.pre_seq_len))
        if args.use_lora:
            self.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, lora_alpha=args.lora_alpha, layer_range=args.layer_range), reinit=True)
            self.get_mixin("eva").vit_model.add_mixin("lora", LoraMixin(args.eva_args['num_layers'], args.lora_rank, lora_alpha=args.lora_alpha, layer_range=args.layer_range), reinit=True)
        elif args.use_qlora:
            self.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, lora_alpha=args.lora_alpha, layer_range=args.layer_range, qlora=True), reinit=True)
        self.args = args

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('CogVLM-finetune', 'CogVLM finetune Configurations')
        group.add_argument('--pre_seq_len', type=int, default=8)
        group.add_argument('--lora_rank', type=int, default=10)
        group.add_argument('--use_ptuning', action="store_true")
        group.add_argument('--use_lora', action="store_true")
        group.add_argument('--use_qlora', action="store_true")
        group.add_argument('--layer_range', nargs='+', type=int, default=None)
        return super().add_model_specific_args(parser)
