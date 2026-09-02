import torch
import torch.nn as nn
import math
import timm
import torch.nn.functional as F

def interpolate_features(descriptors, pts, h, w, normalize=True, patch_size=14, stride=14):
    last_coord_h = ( (h - patch_size) // stride ) * stride + (patch_size / 2)
    last_coord_w = ( (w - patch_size) // stride ) * stride + (patch_size / 2)

    pts = pts - 0.5

    grid_coords = torch.zeros_like(pts)
    grid_coords[..., 0] = 2 * pts[..., 0] / max(w - 1, 1) - 1
    grid_coords[..., 1] = 2 * pts[..., 1] / max(h - 1, 1) - 1

    grid_coords = grid_coords.unsqueeze(1)  # Shape becomes [batch_size, 1, num_keypoints, 2]
    grid_coords = grid_coords.to(descriptors.dtype)

    interpolated_features = F.grid_sample(descriptors, grid_coords, align_corners=False, padding_mode='border', mode='bilinear')

    interpolated_features = interpolated_features.squeeze(2) # shape [batch_size, channels, num_keypoints]

    return F.normalize(interpolated_features, dim=1) if normalize else interpolated_features

# dino patch size is even, so the pixel corner is not really aligned, potential improvements here, borrowed from DINO-Tracker
def interpolate_features_vit(descriptors, pts, h, w, normalize=True, patch_size=14, stride=14):
    last_coord_h = ( (h - patch_size) // stride ) * stride + (patch_size / 2)
    last_coord_w = ( (w - patch_size) // stride ) * stride + (patch_size / 2)
    ah = 2 / (last_coord_h - (patch_size / 2))
    aw = 2 / (last_coord_w - (patch_size / 2))
    bh = 1 - last_coord_h * 2 / ( last_coord_h - ( patch_size / 2 ))
    bw = 1 - last_coord_w * 2 / ( last_coord_w - ( patch_size / 2 ))

    target_dtype = descriptors.dtype
    target_device = descriptors.device

    a = torch.tensor([[aw, ah]], device=target_device, dtype=target_dtype)
    b = torch.tensor([[bw, bh]], device=target_device, dtype=target_dtype)
    keypoints = a * pts.to(device=target_device, dtype=target_dtype) + b

    # Expand dimensions for grid sampling
    keypoints = keypoints.unsqueeze(1)  # Shape becomes [batch_size, 1, num_keypoints, 2]

    # Interpolate using bilinear sampling
    interpolated_features = F.grid_sample(descriptors, keypoints, align_corners=True, padding_mode='border')

    # interpolated_features will have shape [batch_size, channels, 1, num_keypoints]
    interpolated_features = interpolated_features.squeeze(-2)

    return F.normalize(interpolated_features, dim=1) if normalize else interpolated_features

def extract_kp_depth(depth_map, kp, window_size=3):
    B, N, _ = kp.shape

    if not torch.is_tensor(depth_map):
        depth_map = torch.tensor(depth_map, device=kp.device, dtype=torch.float)

    H, W = depth_map.shape[-2:]
    half = window_size // 2

    padded = F.pad(depth_map, (half, half, half, half), mode='replicate')  # shape: (1,1,H+2*half, W+2*half)

    patches = F.unfold(padded, kernel_size=window_size, stride=1)

    # patch_means = patches.mean(dim=1).squeeze(0)  # shape: (H*W)
    patch_means = patches.mean(dim=1) # shape: (B, H*W)

    # indices = kp[0, :, 1] * W + kp[0, :, 0]  # shape: (num_points,)
    indices = kp[..., 1] * W + kp[..., 0]  # shape: (B, num_points,)

    # avg_depths = patch_means[indices.long()]  # shape: (num_points)
    avg_depths = patch_means.gather(dim=1, index=indices.long())  # shape: (B, num_points)

    return avg_depths

# 동일
def pairwise_logistic_ranking_loss(model, pred_scores, gt_depths, depth_threshold=0.0):
    B, N, D = pred_scores.shape

    pred_i = pred_scores.unsqueeze(2).expand(B, N, N, D)  # (B, N, N, D)
    pred_j = pred_scores.unsqueeze(1).expand(B, N, N, D)  # (B, N, N, D)

    depth_i = gt_depths.unsqueeze(2)   # (B, N, 1)
    depth_j = gt_depths.unsqueeze(1)   # (B, 1, N)

    sign_ij = torch.sign(depth_j - depth_i)

    valid_mask = (torch.abs(depth_j - depth_i) > depth_threshold)

    alpha_ij = sign_ij  # (B, N, N)
    score_diff = model((pred_j - pred_i).view(B, -1, D)).view(B, N, N)  # (B, N, N)
    pairwise_loss = torch.log(1.0 + torch.exp(-alpha_ij * score_diff))

    valid_pairwise_loss = pairwise_loss[valid_mask]
    if valid_pairwise_loss.numel() == 0:
        return torch.tensor(0.0, device=pred_scores.device)
    loss = valid_pairwise_loss.mean()
    return loss

# 살펴봐야함
def calculate_intra_depth_loss(
    depth_diff_head: nn.Module,
    final_feature_map: torch.Tensor,
    gt_depth_map: torch.Tensor,
    num_points: int = 256,
    window_size: int = 3,
    depth_threshold: float = 0.05,
    use_vit_interpolation: bool = False,
    xyz_diff_heads: list = None,
    gt_xyz_maps: list = None,
    xyz_threshold: float = 0.05,
) -> torch.Tensor:
    """
    최종 피처 맵과 GT 깊이 맵을 사용하여 intra-depth consistency loss를 계산합니다.

    Args:
        final_feature_map (torch.Tensor): ViT의 최종 피처 맵 (B, C, H_feat, W_feat)
        gt_depth_map (torch.Tensor): 정답 깊이 맵 (B, 1, H_img, W_img)
        num_points (int): 이미지 당 샘플링할 점의 개수
        window_size (int): 깊이 유효성 검사를 위한 윈도우 크기
        depth_threshold (float): 순위 손실 계산 시 무시할 깊이 차이의 임계값
    """
    _, _, H_img, W_img = gt_depth_map.shape
    assert (H_img, W_img) == (490, 490), \
        f"This function is optimized for 490x490 resolution, but got {(H_img, W_img)}"

    use_xyz = (xyz_diff_heads is not None and gt_xyz_maps is not None and len(xyz_diff_heads) == 3 and len(gt_xyz_maps) == 3)

    final_feature_map = final_feature_map.permute(0, 2, 1).reshape(final_feature_map.shape[0], final_feature_map.shape[2], 35, 35)

    # 1. GT 깊이 맵에서 유효한 점들을 샘플링
    # gt_depth_map의 해상도 기준 좌표 (x, y)
    sampled_points_on_gt = sample_valid_points(gt_depth_map, num_points, window_size)

    # 2. 샘플링된 좌표에서 GT 깊이 값 추출
    # extract_kp_depth는 (B, N, 2) 형태의 입력을 처리할 수 있음
    sampled_gt_depths = extract_kp_depth(gt_depth_map, sampled_points_on_gt, window_size=window_size)
    if use_xyz:
        sampled_gt_x = extract_kp_depth(gt_xyz_maps[0], sampled_points_on_gt, window_size=window_size)
        sampled_gt_y = extract_kp_depth(gt_xyz_maps[1], sampled_points_on_gt, window_size=window_size)
        sampled_gt_z = extract_kp_depth(gt_xyz_maps[2], sampled_points_on_gt, window_size=window_size)

    if use_vit_interpolation:
        # 3. DINO 피처 맵에서 특징(feature) 추출
        # 원본 interpolate_features 함수를 그대로 사용합니다.
        # 이 함수는 이미지 좌표계(pts)를 입력받아 내부적으로 DINO의 패치 좌표계로 변환합니다.
        # 따라서 좌표 스케일링이 필요 없습니다.
        sampled_features = interpolate_features_vit(
            descriptors=final_feature_map,
            pts=sampled_points_on_gt,
            h=H_img,  # ★★★ 피처맵 크기가 아닌 '원본 이미지' 크기를 전달
            w=W_img,  # ★★★
            normalize=False, # 모델 피처가 이미 정규화되었다면 False
            patch_size=14, # ★★★ 모델의 패치 크기
            stride=14          # ★★★ 모델의 스트라이드
        ).permute(0, 2, 1) # (B, C, N) -> (B, N, C)
    else:
        # 3. 피처 맵에서 특징(feature) 추출
        # 샘플링된 좌표를 피처 맵의 해상도에 맞게 스케일링
        _, _, H_feat, W_feat = final_feature_map.shape

        scale_factor = torch.tensor([W_feat / W_img, H_feat / H_img], device=final_feature_map.device)
        sampled_points_on_feat = sampled_points_on_gt * scale_factor

        # interpolate_features 함수를 사용하여 최종 피처 맵에서 특징 추출
        # (B, C, H_feat, W_feat) -> (B, C, N) -> (B, N, C)
        sampled_features = interpolate_features(
            final_feature_map,
            sampled_points_on_feat,
            h=H_feat, w=W_feat,
            # interpolate_features 내부에서 정규화를 하지만,
            # DINO 같은 모델은 이미 정규화된 피처를 출력하므로 False로 설정 가능
            normalize=False,
            patch_size=1, # 패치 크기와 스트라이드는 1로 설정하여 일반적인 보간처럼 동작하게 함
            stride=1
        ).permute(0, 2, 1)

    # 4. Pairwise ranking loss 계산
    # 재사용한 pairwise_logistic_ranking_loss 함수는 (B, N, D) 형태의 입력을 기대함
    intra_depth_loss = pairwise_logistic_ranking_loss(
        depth_diff_head,
        sampled_features,
        sampled_gt_depths,
        depth_threshold=depth_threshold
    )

    if use_xyz:
        intra_x_loss = pairwise_logistic_ranking_loss(
            xyz_diff_heads[0],
            sampled_features,
            sampled_gt_x,
            depth_threshold=xyz_threshold
        )
        intra_y_loss = pairwise_logistic_ranking_loss(
            xyz_diff_heads[1],
            sampled_features,
            sampled_gt_y,
            depth_threshold=xyz_threshold
        )
        intra_z_loss = pairwise_logistic_ranking_loss(
            xyz_diff_heads[2],
            sampled_features,
            sampled_gt_z,
            depth_threshold=xyz_threshold
        )
        intra_depth_loss = (intra_depth_loss + intra_x_loss + intra_y_loss + intra_z_loss) / 4.0
        # print(f"🔧 Intra-depth loss: {intra_depth_loss.item():.6f}")
        # print(f"   - Intra-x loss: {intra_x_loss.item():.6f}")
        # print(f"   - Intra-y loss: {intra_y_loss.item():.6f}")
        # print(f"   - Intra-z loss: {intra_z_loss.item():.6f}")

    return intra_depth_loss

# 검증 완료
def sample_valid_points(gt_depth_map: torch.Tensor, num_points: int, window_size: int = 3) -> torch.Tensor:
    """
    GT 깊이 맵에서 유효한 점들을 샘플링합니다.
    유효한 점이란, 해당 픽셀과 주변 윈도우의 깊이 값이 모두 0이 아닌 점을 의미합니다.
    """
    B, _, H, W = gt_depth_map.shape
    device = gt_depth_map.device

    # 깊이 값이 0보다 큰 영역을 마스킹
    valid_mask = (gt_depth_map > 0).float()

    # 윈도우 내 모든 값이 유효한지 확인하기 위해 min_pool 사용
    # 윈도우 내에 0이 하나라도 있으면 결과는 0이 됨
    padding = window_size // 2
    valid_mask_pooled = -F.max_pool2d(-valid_mask, kernel_size=window_size, stride=1, padding=padding)

    # 각 배치별로 유효한 좌표 샘플링
    batch_points = []
    for i in range(B):
        # 유효한 좌표 (y, x) 찾기
        valid_coords = torch.nonzero(valid_mask_pooled[i, 0] > 0) # Shape: [num_valid, 2]

        if valid_coords.shape[0] == 0:
            # 유효한 포인트가 없으면 (0,0) 주변으로 임의의 점 생성 (예외 처리)
            print("Warning: No valid points found for sampling. Returning random points.")
            sampled_coords = torch.randint(low=padding, high=min(H, W) - padding, size=(num_points, 2), device=device)
        else:
            # 유효한 좌표 중에서 num_points 만큼 랜덤으로 선택
            num_valid_points = valid_coords.shape[0]
            perm = torch.randperm(num_valid_points, device=device)
            selected_indices = perm[:min(num_points, num_valid_points)]
            sampled_coords = valid_coords[selected_indices]

        # PyTorch의 grid_sample과 좌표계 순서를 맞추기 위해 (y,x) -> (x,y)로 변경
        batch_points.append(torch.flip(sampled_coords, dims=[1]))

    return torch.stack(batch_points, dim=0).float() # Shape: (B, num_points, 2)

# 동일 함수
class DepthAwareFeatureFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, use_tanh=True):
        super().__init__()
        self.use_tanh = use_tanh

        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        depth_diff = self.fusion_layer(features)

        if self.use_tanh:
            depth_diff = torch.tanh(depth_diff)

        return depth_diff.squeeze(-1)

class XYZEncoder(nn.Module):
    """
    ConvNeXt 아키텍처와 Zero Convolution을 사용하여 (X, Y, Z, Mask) 4채널 맵을 인코딩하는 모듈.
    학습 초기 안정성을 위해 XYZ 정보의 영향을 0에서부터 점진적으로 학습하도록 설계되었습니다.
    """
    def __init__(self,
                 input_channels: int = 4,
                 model_name: str = 'convnext_tiny',
                 vit_hidden_dim: int = 1792,
                 patch_grid_size: int = 35,
                 pretrained: bool = False):
        super().__init__()

        # 1. timm을 통해 ConvNeXt 백본 생성
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=input_channels
        )

        feature_info = self.backbone.feature_info.channels()
        last_stage_channels = feature_info[-1]

        # 2. ConvNeXt 피처 차원을 ViT 히든 차원으로 변환할 프로젝션 헤드
        #    - 첫 번째 Conv: 채널 수를 맞추는 역할
        #    - 두 번째 Conv: Zero Convolution으로, 학습 가능한 제어 신호 역할
        self.projection_head = nn.Conv2d(last_stage_channels, vit_hidden_dim, kernel_size=1)

        self.zero_conv = nn.Conv2d(vit_hidden_dim, vit_hidden_dim, kernel_size=1)

        # 가중치를 0으로 초기화하여, 학습 시작 시점에는 이 레이어가 항등 함수처럼 동작
        # (아무런 정보를 추가하지 않음)하게 만듭니다.
        # 학습이 진행됨에 따라 네트워크는 XYZ 정보를 얼마나 활용할지 점진적으로 학습합니다.
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

        # 3. 공간적 크기를 ViT의 패치 그리드에 맞추기 위한 어댑티브 풀링
        self.pool = nn.AdaptiveAvgPool2d((patch_grid_size, patch_grid_size))

    def forward(self, xyzm_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyzm_map (torch.Tensor): 입력 텐서 (B, 4, H, W)
        Returns:
            torch.Tensor: ViT와 융합 가능한 형태의 텐서 (B, N, D)
                          N = patch_grid_size * patch_grid_size
                          D = vit_hidden_dim
        """
        # ConvNeXt 백본 통과
        last_stage_features = self.backbone(xyzm_map)[-1]

        # 프로젝션 헤드 통과 (차원 맞추기)
        projected_features = self.projection_head(last_stage_features)

        # Zero Convolution 통과 (제어 신호)
        controlled_features = self.zero_conv(projected_features)

        # 풀링을 통해 공간적 크기를 맞춤
        pooled_features = self.pool(controlled_features)

        # ViT 시퀀스(B, N, D) 형태로 변환
        output = pooled_features.flatten(2).transpose(1, 2)

        return output

class XYZHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 간단한 2계층 MLP 구조
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim // 2, 3)

    def forward(self, patch_features):
        # patch_features shape: [Batch, Num_Patches, Input_Dim]
        x = self.fc1(patch_features)
        x = self.relu(x)
        est_xyz = self.fc2(x) # shape: [Batch, Num_Patches, 3]
        return est_xyz

class RangeHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 간단한 2계층 MLP 구조
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim // 2, 1)

    def forward(self, patch_features):
        # patch_features shape: [Batch, Num_Patches, Input_Dim]
        x = self.fc1(patch_features)
        x = self.relu(x)
        est_range = self.fc2(x) # shape: [Batch, Num_Patches, 1]
        return est_range.squeeze(-1)  # [Batch, Num_Patches]


class SemanticHead(nn.Module):
    """
    ViT 패치 특징을 입력받아 시맨틱 클래스 로짓을 예측하는 Auxiliary Head.
    A simple two-layer MLP is often sufficient.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # 간단한 2계층 MLP 구조
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim // 2, num_classes)

    def forward(self, patch_features):
        # patch_features shape: [Batch, Num_Patches, Input_Dim]
        x = self.fc1(patch_features)
        x = self.relu(x)
        logits = self.fc2(x) # shape: [Batch, Num_Patches, Num_Classes]
        return logits


class PairwiseFusion(nn.Module):
    def __init__(self, standard_dim, control_dim, intermediate_dim=None, dropout_p=0.0,
                 name="PairwiseFusion", debug_print=False, detach_standard_grad=False,
                 control_only=False, wo_zero=False, use_concat=False):
        super().__init__()

        self.name = name
        self.debug_print = debug_print
        self.detach_standard_grad = detach_standard_grad
        self.control_only = control_only
        self.wo_zero = wo_zero
        self.use_concat = use_concat

        if intermediate_dim is None:
            intermediate_dim = standard_dim

        if self.control_only:
            self.input_dim = control_dim
        else:
            self.input_dim = standard_dim + control_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.input_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(intermediate_dim, standard_dim)
        )

        if not self.wo_zero:
            nn.init.zeros_(self.fusion_mlp[-1].weight)
            nn.init.zeros_(self.fusion_mlp[-1].bias)

    def forward(self, standard_feature, control_signal):
        # standard_feature: (B, N, Ds) - Ds는 standard_dim
        # control_signal: (B, N, Dc) - Dc는 control_dim

        # control_only 플래그에 따라 MLP 입력 분기 처리
        if self.control_only:
            # 1. Control 신호만 MLP 입력으로 사용
            mlp_input = control_signal
        else:
            # 1. (기존 로직) 두 피처를 채널(dim=-1) 축으로 결합

            # self.detach_standard_grad 값에 따라 standard_feature를 detach할지 결정
            if self.detach_standard_grad:
                processed_standard_feature = standard_feature.detach()
            else:
                processed_standard_feature = standard_feature

            mlp_input = torch.cat([processed_standard_feature, control_signal], dim=-1)

        # 2. MLP를 통과시켜 보강 신호 계산
        refinement_signal = self.fusion_mlp(mlp_input)

        if self.debug_print:
            # refinement_signal 출력
            print(f"[{self.name}] refinement_signal.mean(): {refinement_signal.mean().item():.6f}, min(): {refinement_signal.min().item():.6f}, max(): {refinement_signal.max().item():.6f}")
            # shape 출력
            print(f"[{self.name}] standard_feature.shape: {standard_feature.shape}, control_signal.shape: {control_signal.shape}, mlp_input.shape: {mlp_input.shape}")

        return refinement_signal


class MultiViewFeatureAdapter(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        """
        초기화 메서드.
        args: 모델 전체 설정을 담은 객체.
              use_cnn, mv_cnn_mid_dim, mv_cnn_out_dim 등의 새로운 인자가 필요합니다.
        input_dim: 입력 피처(multi_view_feature)의 채널(C) 차원.
        output_dim: 출력 제어 신호의 차원 (LLaMA 모델의 hidden_size).
        """
        super().__init__()
        self.args = args
        self.use_cnn = hasattr(args, 'use_cnn') and args.use_cnn
        self.normalize_feature = hasattr(args, 'normalize_feature') and args.normalize_feature

        # CNN 사용 여부에 따라 feature_projection 레이어의 입력 차원이 결정됩니다.
        feature_proj_input_dim = input_dim

        if self.use_cnn:
            # 💡 2D Conv 연산을 위한 CNN 레이어 정의
            # 140x140 -> 70x70
            self.cnn1 = nn.Conv2d(
                in_channels=input_dim,
                out_channels=args.mv_cnn_mid_dim,
                kernel_size=3,
                stride=2,
                padding=1
            )
            self.cnn_act = nn.GELU()
            # 70x70 -> 35x35
            self.cnn2 = nn.Conv2d(
                in_channels=args.mv_cnn_mid_dim,
                out_channels=args.mv_cnn_mid_dim,
                kernel_size=3,
                stride=2,
                padding=1
            )
            # feature_projection의 입력 차원은 cnn2의 출력 채널 수가 됩니다.
            feature_proj_input_dim = args.mv_cnn_mid_dim

        if self.normalize_feature:
            # 정규화는 CNN 처리 후, 또는 CNN을 안 쓸 경우 바로 적용됩니다.
            self.feature_norm = nn.LayerNorm(feature_proj_input_dim)

        # 입력 피처(c)와 목표 피처(D_llama)의 차원을 맞춰주기 위한 Linear 레이어
        self.feature_projection = nn.Linear(feature_proj_input_dim, output_dim)

        # Zero Convolution 레이어
        self.zero_conv = nn.Linear(output_dim, output_dim)

        # 가중치와 편향을 0으로 초기화
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, standard_feature, multi_view_feature):
        """
        순전파 메서드.
        multi_view_feature: (B, HxW, C) 형태의 입력 텐서
        """
        x = multi_view_feature

        if self.use_cnn:
            B, N, C = x.shape
            # H, W가 정수인지 확인 (패치 개수가 제곱수여야 함)
            H = W = int(math.sqrt(N))
            assert H * W == N, "CNN을 사용하려면 패치 개수(N)는 완전제곱수여야 합니다."

            # ⚙️ 텐서 재구성: (B, N, C) -> (B, C, H, W)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

            # ➡️ CNN 연산 수행
            x = self.cnn1(x)
            x = self.cnn_act(x)
            x = self.cnn2(x) # 출력 형태: (B, C_out, H_new, W_new)

            # ⚙️ 텐서 재구성: (B, C_out, H_new, W_new) -> (B, N_new, C_out)
            B, C_new, H_new, W_new = x.shape
            x = x.flatten(2).permute(0, 2, 1)

        # 입력 피처 정규화 (선택적)
        if self.normalize_feature:
            x = self.feature_norm(x)

        # 1. 입력 피처를 목표 차원으로 투영
        projected_feature = self.feature_projection(x)
        # 2. Zero Convolution을 통과시켜 제어 신호 생성
        control_signal = self.zero_conv(projected_feature)

        return control_signal
