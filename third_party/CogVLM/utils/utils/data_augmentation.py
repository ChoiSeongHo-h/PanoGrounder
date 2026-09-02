import cv2
import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms
from typing import Optional, Tuple, Union
import uuid
import os
import math

class RotateNorthPole:
    def __init__(self, angle_deg: float, lon_deg: float):
        self.phi = np.deg2rad(angle_deg)
        self.u1 = np.deg2rad(lon_deg)
        self.P0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        H, W = arr.shape[:2]

        # 1) P1: 새 north pole 좌표
        #    위도 v1 = π/2 - φ, 경도 u1 = self.u1
        v1 = np.pi/2 - self.phi
        u1 = self.u1
        x1 = np.sin(u1) * np.cos(v1)
        y1 = np.cos(u1) * np.cos(v1)
        z1 = -np.sin(v1)
        P1 = np.array([x1, y1, z1], dtype=np.float32)

        # 2) 회전축 axis = normalize(P0 × P1)
        axis = np.cross(self.P0, P1)
        axis /= np.linalg.norm(axis)

        # 3) Rodrigues 공식 파라미터
        c, s = np.cos(self.phi), np.sin(self.phi)

        # 4) equirect grid 생성
        u = np.linspace(-np.pi, np.pi, W, endpoint=False, dtype=np.float32)
        v = np.linspace(-np.pi/2, np.pi/2, H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        # 5) 구면→카르테시안
        x = np.sin(uu) * np.cos(vv)
        y = np.cos(uu) * np.cos(vv)
        z = -np.sin(vv)

        # 6) axis–angle 회전 (Rodrigues)
        dot_av = axis[0]*x + axis[1]*y + axis[2]*z
        cx = axis[1]*z - axis[2]*y
        cy = axis[2]*x - axis[0]*z
        cz = axis[0]*y - axis[1]*x

        x_r = x*c + cx*s + axis[0]*dot_av*(1-c)
        y_r = y*c + cy*s + axis[1]*dot_av*(1-c)
        z_r = z*c + cz*s + axis[2]*dot_av*(1-c)

        # 7) 카르테시안→구면
        v_prime = -np.arcsin(np.clip(z_r, -1.0, 1.0))
        u_prime = np.arctan2(x_r, y_r)

        # 8) 픽셀 좌표로 매핑
        map_x = ((u_prime + np.pi) / (2*np.pi)) * W
        map_y = ((v_prime + np.pi/2) / np.pi) * H

        # 9) warp with wrap-around horizontally
        warped = cv2.remap(
            arr,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP
        )
        return Image.fromarray(warped)

class RotateEquirect:
    def __init__(self, shift_frac: float):
        self.shift_frac = shift_frac

    def __call__(self, img: Image.Image) -> Image.Image:
        W, _ = img.size

        # 1. 원래 계산하려 했던 이동값(소수점 포함)을 구합니다.
        ideal_shift = W * self.shift_frac

        # 2. 이 값을 14로 나눈 뒤, 가장 가까운 정수로 반올림합니다.
        # 3. 그 결과에 다시 14를 곱해 '가장 가까운 14의 배수'를 찾습니다.
        shift = int(round(ideal_shift / 14.0) * 14)

        arr = np.array(img)
        arr = np.roll(arr, shift, axis=1)
        return Image.fromarray(arr)

class RotatePatchedFeature:
    """
    패치화된 HxWxC 형태의 Torch 피처 텐서를 회전시킵니다.
    이동량은 원본 이미지 너비를 기준으로 계산됩니다.
    """
    def __init__(self, shift_frac: float, original_img_width: int):
        self.shift_frac = shift_frac
        self.original_img_width = original_img_width

    def __call__(self, feature: torch.Tensor) -> torch.Tensor:
        # 1. 원본 이미지 기준으로 이상적인 이동값(소수점 포함)을 계산합니다.
        ideal_shift = self.original_img_width * self.shift_frac

        # 2. 패치화된 피처의 이동량을 계산합니다. (요청된 공식)
        # 14로 나눈 뒤 가장 가까운 정수로 반올림합니다.
        feature_shift = int(round(ideal_shift / 14.0))

        # 3. torch.roll을 사용하여 너비(dim=1) 축을 기준으로 텐서를 회전시킵니다.
        return torch.roll(feature, shifts=feature_shift, dims=1)

class BlurTopBottom:
    def __init__(self):
        self.blur_h = 150  # pixel
        self.blur_radius = 50  # pixel

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        fade_h = min(self.blur_h, h // 2)
        mid = fade_h / 2

        img_min = img
        img_max = img.filter(ImageFilter.GaussianBlur(self.blur_radius))

        mask = Image.new("L", (w, h), color=0)
        draw = ImageDraw.Draw(mask)

        for y in range(fade_h):
            if y < mid:
                alpha = 255
            else:
                alpha = int((1 - (y - mid) / (fade_h - mid)) * 255)
            draw.line([(0, y), (w, y)], fill=alpha)

        for y in range(h - fade_h, h):
            yy = h - 1 - y
            if yy < mid:
                alpha = 255
            else:
                alpha = int((1 - (yy - mid) / (fade_h - mid)) * 255)
            draw.line([(0, y), (w, y)], fill=alpha)

        blurred_img = Image.composite(img_max, img_min, mask)
        return blurred_img

def create_point_map_batch(range_map_batch, scale=1.0):
    """
    (PyTorch) Equirectangular range map 배치로부터 (B, H, W, 3) 형태의 3D 포인트 맵 배치를 생성합니다.
    모든 연산은 벡터화되어 GPU에서 효율적으로 실행됩니다.

    Args:
        range_map_batch (torch.Tensor): (B, H, W) 형태의 16비트 range map 배치.
        scale (float): 거리 값에 적용할 스케일 팩터.

    Returns:
        torch.Tensor: (B, H, W, 3) 형태의 float32 포인트 맵 배치. 각 픽셀은 (X, Y, Z) 좌표.
    """
    B, H, W = range_map_batch.shape
    device = range_map_batch.device

    # 전체 픽셀에 대한 인덱스 그리드 생성 (GPU에서 직접 생성)
    # indexing='ij'는 NumPy와 동일한 (행, 열) 순서를 보장합니다.
    rows, cols = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # 브로드캐스팅을 위해 배치 차원 추가: (H, W) -> (1, H, W)
    rows = rows.unsqueeze(0)
    cols = cols.unsqueeze(0)

    # 유효한 거리 값을 가진 픽셀 마스크
    valid_mask = range_map_batch > 0

    # 각도 계산 (배치 전체에 대해 한 번에 계산)
    phi = (cols / (W - 1)) * 2 * torch.pi
    theta = (rows / (H - 1)) * torch.pi

    r = range_map_batch * scale

    # 구면 좌표계 -> 직교 좌표계 변환 (벡터화 연산)
    x = r * torch.sin(theta) * torch.cos(phi)
    y = -r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    # (B, H, W, 3) 형태로 결과를 쌓음
    point_map = torch.stack((x, y, z), dim=-1)

    # 마스크를 사용하여 유효하지 않은 픽셀(거리가 0인)의 좌표를 0으로 설정
    # valid_mask (B, H, W) -> (B, H, W, 1)로 확장하여 채널(3)에 대해 브로드캐스팅
    point_map[~valid_mask] = 0

    return point_map

def augment_xyz_z_rotation_batch(xyz_tensor: torch.Tensor, xyz_aug_num: int) -> torch.Tensor:
    """
    배치 단위로 XYZ 텐서를 Z축 기준으로 회전시켜 증강합니다.

    Args:
        xyz_tensor (torch.Tensor): (B, H, W, 3) 모양의 원본 XYZ 텐서.
        xyz_aug_num (int): 생성할 증강의 개수.

    Returns:
        torch.Tensor: (B, xyz_aug_num, H, W, 3) 모양의 증강된 XYZ 텐서.
    """
    # 입력 텐서의 정보 가져오기
    B, H, W, _ = xyz_tensor.shape
    device = xyz_tensor.device

    # 각 증강에 대한 회전 각도 계산
    # angles = [0, 1*2pi/N, 2*2pi/N, ..., (N-1)*2pi/N]
    angles = torch.arange(xyz_aug_num, device=device) * (2 * math.pi / xyz_aug_num)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    # Z축 회전 행렬 생성 (aug_num, 3, 3)
    # R_z(θ) = [[cosθ, -sinθ, 0],
    #           [sinθ,  cosθ, 0],
    #           [0,     0,    1]]
    rotation_matrices = torch.zeros(xyz_aug_num, 3, 3, device=device)
    rotation_matrices[:, 0, 0] = cos_a
    rotation_matrices[:, 0, 1] = -sin_a
    rotation_matrices[:, 1, 0] = sin_a
    rotation_matrices[:, 1, 1] = cos_a
    rotation_matrices[:, 2, 2] = 1.0

    # (B, H, W, 3) -> (B, H*W, 3) 형태로 변환하여 행렬곱 준비
    xyz_flat = xyz_tensor.view(B, H * W, 3)

    # torch.einsum을 사용하여 배치 단위로 효율적인 행렬곱 수행
    # 'bpi,aoi->bapo': (batch, point, coord_in), (aug, coord_out, coord_in) -> (batch, aug, point, coord_out)
    augmented_xyz_flat = torch.einsum('bpi,aoi->bapo', xyz_flat, rotation_matrices)

    # 원래 모양 (B, aug_num, H, W, 3)으로 복원
    augmented_xyz_batch = augmented_xyz_flat.view(B, xyz_aug_num, H, W, 3)

    return augmented_xyz_batch

class FourierFeatureEncoder(torch.nn.Module):
    """
    입력 좌표(예: XYZ)를 푸리에 피처로 인코딩하는 클래스.
    NeRF에서 사용하는 위치 인코딩(Positional Encoding)과 동일한 원리입니다.
    """
    def __init__(self, input_dim: int, fourier_feature_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.fourier_feature_dim = fourier_feature_dim

        # 주파수 밴드를 정의합니다: [2^0, 2^1, ..., 2^(L-1)]
        # register_buffer를 사용하면 모델의 파라미터는 아니지만, state_dict에 저장되고
        # .to(device) 호출 시 함께 이동하는 텐서로 관리됩니다.
        frequencies = 2.0 ** torch.arange(fourier_feature_dim)
        self.register_buffer('frequencies', frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (..., input_dim) 모양의 입력 텐서
        Returns:
            torch.Tensor: (..., output_dim) 모양의 인코딩된 텐서
                          output_dim = input_dim + input_dim * fourier_feature_dim * 2
        """
        # (..., input_dim) -> (..., input_dim, 1) * (fourier_feature_dim) -> (..., input_dim, fourier_feature_dim)
        projected = x.unsqueeze(-1) * self.frequencies

        # sin, cos 함수를 적용합니다.
        # (..., input_dim, fourier_feature_dim) -> (..., input_dim, fourier_feature_dim, 2)
        encoded_features = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

        # 마지막 두 차원을 하나로 펼쳐 피처 벡터를 만듭니다.
        # (..., input_dim, fourier_feature_dim, 2) -> (..., input_dim * fourier_feature_dim * 2)
        encoded_features = encoded_features.flatten(-2)

        # 원본 입력(x)을 인코딩된 피처와 결합합니다. (NeRF 논문에서 사용된 방식)
        output = torch.cat([x, encoded_features], dim=-1)

        return output

class ToPinhole:
    def __init__(self, fov_deg: float, option):
        self.fov_deg = fov_deg
        self.option = option

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        2:1 또는 1:1 equirectangular 파노라마(PIL Image)에서 정면 rectilinear 뷰 추출
        """
        # PIL → NumPy (RGB)
        img_np = np.array(img)
        H, W = img_np.shape[:2]

        ### MODIFIED ###
        # 1:1 비율도 허용하도록 수정
        if not (abs(W / H - 2.0) < 1e-3 or abs(W / H - 1.0) < 1e-3):
            raise ValueError(f"입력 파노라마 비율이 2:1 또는 1:1이 아님. (W={W}, H={H})")

        # 수평 시야각 → 출력 크기 (원본 각해상도 유지)
        fov_rad = math.radians(self.fov_deg)

        ### MODIFIED ###
        # 출력 너비 계산 기준을 W에서 H로 변경
        out_width = int(round(H * (fov_rad / math.pi)))
        out_width = max(8, out_width)
        out_height = out_width  # 정사각형 출력

        # 카메라 초점거리
        f = (out_width / 2.0) / math.tan(fov_rad / 2.0)

        # 출력 좌표계
        xs = np.linspace(-(out_width - 1) / 2, (out_width - 1) / 2, out_width, dtype=np.float32)
        ys = np.linspace(-(out_height - 1) / 2, (out_height - 1) / 2, out_height, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)

        # 방향 벡터 (정면)
        dirs = np.stack([xx, -yy, np.full_like(xx, f)], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        # 구면좌표 변환
        lon = np.arctan2(dirs[..., 0], dirs[..., 2])
        lat = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))

        # equirectangular 좌표 매핑
        u = (lon / (2 * math.pi) + 0.5) * W
        v = (0.5 - lat / math.pi) * H

        u = np.mod(u, W).astype(np.float32)
        v = np.clip(v, 0, H - 1).astype(np.float32)

        # OpenCV remap은 BGR → RGB 변환 필요
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        if self.option == "linear":
            front_bgr = cv2.remap(img_bgr, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        elif self.option == "nearest":
            front_bgr = cv2.remap(img_bgr, u, v, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        else:
            # (option 유효성 검사 추가)
            raise ValueError("option은 'linear' 또는 'nearest'만 허용")

        # 결과 BGR → RGB → PIL
        if self.option == "linear":
            front_img = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
        elif self.option == "nearest":
            front_img = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2GRAY)
        front_pil = Image.fromarray(front_img)
        return front_pil

class ToPinhole4:
    def __init__(self, fov_deg: float, option):
        self.fov_deg = fov_deg            # "linear" 또는 "nearest"
        self.option = option

    def __call__(self, img: Image.Image) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
        img_np = np.array(img)
        H, W = img_np.shape[:2]

        ### MODIFIED ###
        # 1:1 비율도 허용하도록 수정
        if not (abs(W / H - 2.0) < 1e-3 or abs(W / H - 1.0) < 1e-3):
            raise ValueError(f"입력 파노라마 비율이 2:1 또는 1:1이 아님. (W={W}, H={H})")

        fov_rad = math.radians(self.fov_deg)

        ### MODIFIED ###
        # 출력 너비 계산 기준을 W에서 H로 변경
        out_w = max(8, int(round(H * (fov_rad / math.pi))))
        out_h = out_w
        f = (out_w / 2.0) / math.tan(fov_rad / 2.0)

        xs = np.linspace(-(out_w - 1) / 2, (out_w - 1) / 2, out_w, dtype=np.float32)
        ys = np.linspace(-(out_h - 1) / 2, (out_h - 1) / 2, out_h, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)

        dirs = np.stack([xx, -yy, np.full_like(xx, f)], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        lon0 = np.arctan2(dirs[..., 0], dirs[..., 2])
        lat  = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))

        def map_with_yaw(yaw_rad: float):
            lon = lon0 + yaw_rad
            u = (lon / (2 * math.pi) + 0.5) * W
            v = (0.5 - lat / math.pi) * H
            return np.mod(u, W).astype(np.float32), np.clip(v, 0, H - 1).astype(np.float32)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        def render_at(yaw_deg: float) -> Image.Image:
            u, v = map_with_yaw(math.radians(yaw_deg))
            interp = cv2.INTER_LINEAR if self.option == "linear" else cv2.INTER_NEAREST
            bgr = cv2.remap(img_bgr, u, v, interpolation=interp, borderMode=cv2.BORDER_WRAP)
            if self.option == "linear":
                out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif self.option == "nearest":
                out = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError("option은 'linear' 또는 'nearest'만 허용")
            return Image.fromarray(out)

        # (front, left, right, back) 순서의 튜플 반환
        return (
            render_at(0.0),
            render_at(90.0),
            render_at(180.0),
            render_at(-90.0),
        )

def joint_transform(
    img: Optional[Image.Image] = None,
    gt_img: Image.Image = None,
    feature: torch.Tensor = None,
    semantic: torch.Tensor = None,
    gt_range: torch.Tensor = None,
    input_range: Optional[Image.Image] = None,
    is_train: bool = True,
    image_idx: Optional[int] = None,
    max_deg: float = 3.0,
    use_xyz_aug: bool = False,
    xyz_aug_num: int = 8,
    debug_print: bool = False,
    use_fourier_features: bool = False,
    fourier_feature_dim: int = 6,
    use_rangevit: bool = False,
    use_resize_xyz: bool = False,
    use_range_dino: bool = False,
    use_relative_range_head: bool = False,
    use_relative_xyz_head: bool = False,
    num_rot: Optional[int] = None,
    pinhole_mode = None,
) -> Union[Tuple[Image.Image, Image.Image, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Image.Image]:
    W, _ = gt_img.size
    if is_train:
        angle_deg = random.uniform(-max_deg, max_deg)
        lon_deg   = random.uniform(-180.0, 180.0)
        shift_frac = random.random()

        img_transform = transforms.Compose([
            # RotateNorthPole(angle_deg, lon_deg),
            RotateEquirect(shift_frac),
            # BlurTopBottom(),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            ToPinhole4(fov_deg=120.0, option="linear") if pinhole_mode == "gt" else lambda x: x,
        ])

        gt_transform = transforms.Compose([
            # RotateNorthPole(angle_deg, lon_deg),
            RotateEquirect(shift_frac),
            ToPinhole4(fov_deg=120.0, option="nearest") if pinhole_mode == "gt" else lambda x: x,
        ])

        feature_transform = RotatePatchedFeature(shift_frac, W)

    else:
        assert num_rot is not None
        assert image_idx < num_rot
        shift_frac = image_idx / num_rot

        img_transform = transforms.Compose([
            RotateEquirect(shift_frac),
            # BlurTopBottom(),
            ToPinhole(fov_deg=120.0, option="linear") if pinhole_mode == "nways" else lambda x: x,
            ToPinhole4(fov_deg=120.0, option="linear") if pinhole_mode == "gt" else lambda x: x
        ])

        gt_transform = transforms.Compose([
            RotateEquirect(shift_frac),
            ToPinhole(fov_deg=120.0, option="nearest") if pinhole_mode == "nways" else lambda x: x,
            ToPinhole4(fov_deg=120.0, option="nearest") if pinhole_mode == "gt" else lambda x: x,
        ])

        feature_transform = RotatePatchedFeature(shift_frac, W)

    img = img_transform(img) if img is not None else None
    gt_img = gt_transform(gt_img)
    feature = feature_transform(feature) if feature is not None else None
    semantic = feature_transform(semantic) if semantic is not None else None
    gt_range = feature_transform(gt_range) if gt_range is not None else None
    input_range = gt_transform(input_range) if input_range is not None else None

    gt_xyz = None
    if gt_range is not None:
        gt_xyz = create_point_map_batch(gt_range.squeeze(-1).unsqueeze(0), scale=1.0).squeeze(0)

    input_xyz = None
    input_geometry = None
    fourier_encoder = None
    if use_fourier_features:
        fourier_encoder = FourierFeatureEncoder(input_dim=3, fourier_feature_dim=fourier_feature_dim)
    if input_range is not None:
        input_range = torch.from_numpy(np.array(input_range)).to(torch.long)
        input_xyz = create_point_map_batch(input_range.unsqueeze(0), scale=0.001)

        if use_xyz_aug and input_xyz is not None:
            input_xyz = augment_xyz_z_rotation_batch(input_xyz, xyz_aug_num).squeeze(0)

        # 1. 계산을 위해 텐서를 float 타입으로 변환
        input_xyz = input_xyz.float()

        if use_relative_range_head:
            gt_range = input_range.float()

        if use_relative_xyz_head:
            gt_xyz = input_xyz.squeeze(0)

        if use_range_dino:
            input_range = input_range.float()

            # 2. Min-Max 정규화 (0~1 사이 값으로 스케일링)
            min_val = input_range.min()
            max_val = input_range.max()
            # 분모가 0이 되는 것을 방지하기 위해 아주 작은 값(epsilon)을 더해줍니다.
            normalized_geometry = (input_range - min_val) / (max_val - min_val + 1e-8)

            # 3. 3채널로 확장 (H, W) -> (C, H, W)
            # 먼저 채널 차원을 추가 (490, 490) -> (1, 490, 490)
            # 그 후, 채널 차원을 3번 복사 (1, 490, 490) -> (3, 490, 490)
            expanded_geometry = normalized_geometry.unsqueeze(0).repeat(3, 1, 1)

            # 4. ImageNet 통계치로 표준화 (Standardization)
            # 표준화에 사용할 평균(mean)과 표준편차(std)를 텐서로 정의
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])

            # (3, H, W) 형태의 이미지 텐서에 적용할 수 있도록 (3, 1, 1) 형태로 변환
            mean = mean.view(3, 1, 1)
            std = std.view(3, 1, 1)

            # 브로드캐스팅을 통해 각 채널별로 평균을 빼고 표준편차로 나누어 줍니다.
            input_geometry = (expanded_geometry - mean) / std
            # torch.Size([3, 490, 490])인 이미지를 size=(560, 560), mode='nearest'로 리사이즈
            input_geometry = torch.nn.functional.interpolate(
                input_geometry.unsqueeze(0), size=(560, 560), mode='nearest'
            ).permute(0, 2, 3, 1)
        else:
            # 2. 표준화를 위한 평균(mean)과 표준편차(std) 정의
            mean = torch.tensor([0.0064, 0.0064, -0.8361], device=input_xyz.device, dtype=torch.float32)
            std = torch.tensor([1.06445, 1.06445, 0.7361], device=input_xyz.device, dtype=torch.float32)
            mean_range = torch.tensor([1.7476], device=input_range.device, dtype=torch.float32)
            mean_std = torch.tensor([0.7093], device=input_range.device, dtype=torch.float32)

            # 3. 이상치에 대한 마스크 생성
            mask = (~torch.all(input_xyz == 0, dim=-1, keepdim=True)).float()

            # 4. 표준화 적용
            standardized_xyz = (input_xyz - mean) / std
            standardized_range = (input_range.float() - mean_range) / mean_std

            # 5. 마스크를 곱하여 이상치 위치의 값을 다시 0으로 만들기
            standardized_xyz = standardized_xyz * mask
            standardized_range = standardized_range.unsqueeze(-1) * mask

            # 6. use_fourier_features 플래그 값에 따라 분기 처리
            if use_fourier_features:
                # [CASE 1] 푸리에 피처 적용
                # 6-1. 표준화된 XYZ 텐서에 푸리에 피처 적용
                fourier_features = fourier_encoder(standardized_xyz)

                # 6-2. 푸리에 피처가 적용된 텐서와 마스크를 채널 차원 기준으로 결합
                #      (N, H, W, 3+6L) + (N, H, W, 1) -> (N, H, W, 4+6L)
                input_geometry = torch.cat((fourier_features, mask), dim=-1)
            else:
                # [CASE 2] 기존 방식 (푸리에 피처 미적용)
                # 6-1. 표준화된 XYZ 텐서와 마스크를 채널 차원 기준으로 결합
                #      (N, H, W, 3) + (N, H, W, 1) -> (N, H, W, 4)
                input_geometry = torch.cat((standardized_xyz, mask), dim=-1)
                if use_rangevit:
                    input_geometry = torch.cat((input_geometry, standardized_range), dim=-1)
                    if use_resize_xyz:
                        input_geometry = torch.nn.functional.interpolate(
                            input_geometry.permute(0, 3, 1, 2), size=(280, 280), mode='nearest'
                        ).permute(0, 2, 3, 1)

    if img is not None:
        return img, gt_img, feature, semantic, gt_range, gt_xyz, input_geometry
    else:
        return gt_img
