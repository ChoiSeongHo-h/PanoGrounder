import cv2
import numpy as np
import torch
from pathlib import Path
import random
import textwrap
import uuid
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable, Any

# --- 1. 설정 및 상수 ---

# 시각화 결과를 저장할 디렉토리
VISUALIZATION_DIR = Path("visualizations_single_qa_aug")

# "더 멀다"고 판단하기 위한 최소 거리 차이 (mm)
MIN_RANGE_DIFF = 50  # 5cm

# X, Y, Z 좌표 비교를 위한 최소 차이 (meters)
MIN_COORD_DIFF = 0.05 # 5cm

# 윈도우(BBox) 크기 (홀수 권장)
WINDOW_SIZE = 3
HALF_WIN = WINDOW_SIZE // 2 # 윈도우 크기 절반

# 원본 이미지 및 타겟 그리드 해상도
IMG_H, IMG_W = 490, 490
GRID_H, GRID_W = 1000, 1000

# 2D 시각화 시 상단 텍스트 영역 높이
TEXT_AREA_HEIGHT = 100 # 100 픽셀

# QA 쌍 생성을 위한 최대 시도 횟수
MAX_TRIES_PER_CALL = 50
# Hard Z 샘플링을 위한 내부 시도 횟수
MAX_TRIES_HARD_Z_SAMPLING = 20

# --- 1.5. 신규 QA 생성 설정 (요청 5) ---
@dataclass
class QAConfig:
    """QA 생성 옵션 및 확률 설정"""

    # [요청 1] N-ary (K개 BBox) 비교
    n_ary_enabled: bool = True
    n_ary_prob: float = 0.35  # n-ary (K>2) 질문이 생성될 확률
    n_ary_k_choices: List[int] = field(default_factory=lambda: [3, 4]) # 3개 또는 4개 비교

    # [요청 2] BBox 중심 (Relative) 질문
    relative_enabled: bool = True
    relative_prob: float = 0.35 # 기준 BBox 중심 질문이 생성될 확률

    # [요청 3] 관계 조합 (Combined) 질문 (K=2 고정)
    combined_enabled: bool = True
    combined_prob: float = 0.3 # 관계 조합 질문이 생성될 확률

    # [요청 4] Hard Z 로직
    hard_z_enabled: bool = True
    hard_z_prob: float = 0.4 # Hard Z 질문이 생성될 확률 (조건부)


# --- 2. 시각화 색상 정의 ---
COLOR_ANSWER_RGB_O3D = [0.0, 1.0, 0.0]    # 초록색 (정답)
COLOR_OTHER_RGB_O3D = [1.0, 0.0, 0.0]    # 빨간색 (오답)
COLOR_SCENE_RGB_O3D = [0.7, 0.7, 0.7]    # 회색 (씬)
COLOR_CENTER_RGB_O3D = [1.0, 1.0, 0.0]   # 노란색 (기준점)

COLOR_ANSWER_BGR_CV = (0, 255, 0)      # 초록색 (정답)
COLOR_OTHER_BGR_CV = (0, 0, 255)       # 빨간색 (오답)
COLOR_TEXT_CV = (255, 255, 255)      # 흰색 (텍스트)
COLOR_CENTER_BGR_CV = (0, 255, 255)    # 노란색 (기준점)

# --- 3. 회전 증강 (Rotation Augmentation) 정의 ---
# [수정 금지]
# 사용자 좌표계: +X=뒤(Back), +Y=오른쪽(Right), +Z=위(Up)
# 카메라 기본 시점: 원점에서 -X축(앞, Front)을 바라봄.
ROTATION_VIEWS = {
    'default': (
        "",
        ((1, 1), (2, 1), (0, -1)) # NewX=+Y, NewY=+Z, NewZ=-X
    ),
    'back': (
        "From a perspective looking towards the back, ",
        ((1, -1), (2, 1), (0, 1)) # NewX=-Y, NewY=+Z, NewZ=+X
    ),
    'right': (
        "From a perspective looking towards the right, ",
        ((0, 1), (2, 1), (1, 1))  # NewX=+X, NewY=+Z, NewZ=+Y
    ),
    'left': (
        "From a perspective looking towards the left, ",
        ((0, -1), (2, 1), (1, -1)) # NewX=-X, NewY=+Z, NewZ=-Y
    ),
}

def remap_coord(coords: np.ndarray, remap_tuple: tuple) -> float:
    """
    [수정 금지]
    원본 (x, y, z) 좌표를 맵핑 튜플을 사용해 새로운 단일 좌표 값으로 변환합니다.
    """
    axis_idx, sign = remap_tuple
    return coords[axis_idx] * sign

# --- 4. 헬퍼 함수 (좌표 변환) ---
# [수정 없음]
def scale_coordinates(y, x, img_h, img_w, grid_h, grid_w):
    y_scaled = int(round((y / (img_h - 1)) * (grid_h - 1)))
    x_scaled = int(round((x / (img_w - 1)) * (grid_w - 1)))
    return y_scaled, x_scaled

def rescale_coordinates_to_img(x_s, y_s, img_h, img_w, grid_h, grid_w):
    y_img = int(round((y_s / (grid_h - 1)) * (img_h - 1)))
    x_img = int(round((x_s / (grid_w - 1)) * (img_w - 1)))
    return x_img, y_img

def rescale_bbox_to_img(bbox_s, img_h, img_w, grid_h, grid_w):
    x_min_s, y_min_s, x_max_s, y_max_s = bbox_s
    x_min_img, y_min_img = rescale_coordinates_to_img(x_min_s, y_min_s, img_h, img_w, grid_h, grid_w)
    x_max_img, y_max_img = rescale_coordinates_to_img(x_max_s, y_max_s, img_h, img_w, grid_h, grid_w)
    return (x_min_img, y_min_img), (x_max_img, y_max_img)


# --- 4.5. BBox 데이터 클래스 및 시각화 헬퍼 ---

@dataclass
class BBoxData:
    """
    [신규] BBox 관련 모든 정보를 저장하는 데이터 클래스 (요청 6)
    """
    y_c: int  # 490 그리드 기준 중심 y
    x_c: int  # 490 그리드 기준 중심 x

    # 490 그리드 기준 경계
    y_min: int
    y_max: int
    x_min: int
    x_max: int

    # 1000 그리드 기준 경계 (x_min, y_min, x_max, y_max)
    bbox_s_coords: Tuple[int, int, int, int]
    # 1000 그리드 기준 BBox 문자열 "[[xxx,yyy,xxx,yyy]]"
    bbox_str: str

    # 원본 데이터 평균
    mean_range_mm: int
    mean_coords: np.ndarray # 원본 (x, y, z)

    # 리맵핑된 데이터 (x_new, y_new, z_new)
    remapped_coords: Dict[str, float]


def _save_visualizations(
    prompt: str,
    label_bbox_str: str,  # 정답 BBox 문자열
    bboxes: List[BBoxData], # [신규] K개의 BBoxData 리스트 (요청 1)
    point_map: np.ndarray,
    range_map_mm: np.ndarray,
    dataset_name_suffix: str
):
    """
    [수정됨] QA 쌍에 대한 2D 및 3D 시각화 파일을 저장합니다.
    K개의 BBox를 받아 정답은 초록색, 나머지는 빨간색으로 시각화합니다.
    """

    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    unique_hash = uuid.uuid4().hex[:16]
    base_output_name = unique_hash

    # [신규 수정] 기준 BBox (Relative)인지 확인
    is_relative_case = "_Rel" in dataset_name_suffix
    # [신규 수정] Relative 케이스의 경우, bboxes[0]이 항상 기준점임
    # (주의: _generate_qa_minmax에서 center_bbox가 아닌 comparison_bboxes 리스트만
    #  전달하도록 로직이 변경되면 이 부분도 함께 변경되어야 합니다.
    #  현재 로직(generated_bboxes = bboxes_data) 기준으로는 bboxes[0]이 기준점이 맞습니다.)
    center_bbox_str = bboxes[0].bbox_str if is_relative_case and bboxes else None

    # --- 5.6.1. 2D 이미지 시각화 (텍스트 포함) ---
    try:
        vis_img_normalized = np.zeros_like(range_map_mm, dtype=np.float32)
        valid_ranges = range_map_mm[range_map_mm > 0]

        if len(valid_ranges) > 0:
            min_val, max_val = np.min(valid_ranges), np.max(valid_ranges)
            range_val = float(max_val - min_val)
            if range_val > 0:
                mask_valid = range_map_mm > 0
                vis_img_normalized[mask_valid] = (range_map_mm[mask_valid].astype(np.float32) - min_val) / range_val
            else:
                vis_img_normalized[range_map_mm > 0] = 1.0

        vis_img_8u = (vis_img_normalized * 255.0).astype(np.uint8)
        vis_img_bgr = cv2.cvtColor(vis_img_8u, cv2.COLOR_GRAY2BGR)

        # [신규] K개의 BBox 그리기
        for bbox in bboxes:
            is_answer = (bbox.bbox_str == label_bbox_str)
            is_center = (center_bbox_str is not None and bbox.bbox_str == center_bbox_str)
            if is_answer:
                color_cv = COLOR_ANSWER_BGR_CV
            elif is_center:
                color_cv = COLOR_CENTER_BGR_CV # [신규] 기준점 색상
            else:
                color_cv = COLOR_OTHER_BGR_CV

            (x_min_img, y_min_img), (x_max_img, y_max_img) = rescale_bbox_to_img(
                bbox.bbox_s_coords, IMG_H, IMG_W, GRID_H, GRID_W)

            cv2.rectangle(vis_img_bgr, (x_min_img, y_min_img), (x_max_img, y_max_img), color_cv, thickness=1)

        # 텍스트 영역 추가
        final_vis_img = np.zeros((IMG_H + TEXT_AREA_HEIGHT, IMG_W, 3), dtype=np.uint8)
        final_vis_img[TEXT_AREA_HEIGHT:, :, :] = vis_img_bgr

        lines = textwrap.wrap(prompt, width=50)
        y_text_start = 20
        line_height = 20

        for i, line in enumerate(lines):
            if i * line_height > TEXT_AREA_HEIGHT - line_height: break
            y_pos = y_text_start + i * line_height
            cv2.putText(final_vis_img, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, COLOR_TEXT_CV, 1, cv2.LINE_AA)

        save_path_img = VISUALIZATION_DIR / (base_output_name + ".png")
        cv2.imwrite(str(save_path_img), final_vis_img)

    except Exception as e:
        print(f"Warning: failed to save 2D visualization {base_output_name}.png: {e}")

    # --- 5.6.2. 3D PLY 시각화 ---
    try:
        points_flat = point_map.reshape(-1, 3)
        colors_flat = np.full(points_flat.shape, COLOR_SCENE_RGB_O3D, dtype=np.float64)

        # [신규] K개의 BBox 영역 색상 적용
        for bbox in bboxes:
            is_answer = (bbox.bbox_str == label_bbox_str)
            is_center = (center_bbox_str is not None and bbox.bbox_str == center_bbox_str)
            if is_answer:
                color_o3d = COLOR_ANSWER_RGB_O3D
            elif is_center:
                color_o3d = COLOR_CENTER_RGB_O3D # [신규] 기준점 색상
            else:
                color_o3d = COLOR_OTHER_RGB_O3D

            # 490 그리드 기준 좌표 사용
            mask = np.zeros((IMG_H, IMG_W), dtype=bool)
            mask[bbox.y_min:bbox.y_max+1, bbox.x_min:bbox.x_max+1] = True
            colors_flat[mask.flatten()] = color_o3d

        valid_mask_flat = (range_map_mm.flatten() > 0)
        final_points = points_flat[valid_mask_flat]
        final_colors = colors_flat[valid_mask_flat]

        if len(final_points) == 0:
            raise ValueError("유효한 3D 포인트가 없습니다.")

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_points)
        pcd.colors = o3d.utility.Vector3dVector(final_colors)

        save_path_ply = VISUALIZATION_DIR / (base_output_name + ".ply")
        o3d.io.write_point_cloud(str(save_path_ply), pcd)

    except Exception as e:
        print(f"Warning: failed to save 3D visualization {base_output_name}.ply: {e}")

    print(f"Generated QA sample: {dataset_name_suffix} ({unique_hash})")


# --- 5. 신규 QA 생성 헬퍼 함수 (요청 1, 2, 3, 4, 6) ---

def _get_bbox_data(
    center_yx: Tuple[int, int],
    point_map: np.ndarray,
    range_map_mm: np.ndarray,
    remap_tuples: Dict[str, tuple]
) -> BBoxData:
    """
    [신규] BBox 중심점(490)을 받아 BBoxData 객체를 생성하고 반환합니다.
    """
    y_c, x_c = center_yx

    # 5.4.2. BBox 경계 계산 (490x490 이미지 기준)
    y_min, y_max = y_c - HALF_WIN, y_c + HALF_WIN
    x_min, x_max = x_c - HALF_WIN, x_c + HALF_WIN

    # 5.4.3. BBox 내부 데이터 평균 계산
    win_range = range_map_mm[y_min:y_max+1, x_min:x_max+1]
    win_coords = point_map[y_min:y_max+1, x_min:x_max+1, :]

    mean_range_mm = int(np.mean(win_range))
    mean_coords = np.mean(win_coords, axis=(0, 1)) # (old_x, old_y, old_z)

    # 5.4.4. 선택된 시점(View)에 맞게 좌표 리맵핑
    remapped_coords = {
        'x_new': remap_coord(mean_coords, remap_tuples['x']),
        'y_new': remap_coord(mean_coords, remap_tuples['y']),
        'z_new': remap_coord(mean_coords, remap_tuples['z']),
    }

    # 5.4.5. BBox 좌표 스케일링 (1000x1000 그리드 기준)
    y_min_s, x_min_s = scale_coordinates(y_min, x_min, IMG_H, IMG_W, GRID_H, GRID_W)
    y_max_s, x_max_s = scale_coordinates(y_max, x_max, IMG_H, IMG_W, GRID_H, GRID_W)

    bbox_s_coords = (x_min_s, y_min_s, x_max_s, y_max_s)
    bbox_str = f"[[{x_min_s:03d},{y_min_s:03d},{x_max_s:03d},{y_max_s:03d}]]"

    return BBoxData(
        y_c=y_c, x_c=x_c,
        y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max,
        bbox_s_coords=bbox_s_coords,
        bbox_str=bbox_str,
        mean_range_mm=mean_range_mm,
        mean_coords=mean_coords,
        remapped_coords=remapped_coords
    )

def _sample_valid_indices(k: int, valid_indices: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """
    [신규] 유효한 인덱스 K개를 중복 없이 샘플링합니다.
    """
    if len(valid_indices) < k:
        return None
    try:
        indices = random.sample(range(len(valid_indices)), k)
        return [tuple(valid_indices[i]) for i in indices]
    except ValueError:
        return None

def _sample_hard_z_indices(
    valid_indices: np.ndarray,
    point_map: np.ndarray
) -> Optional[List[Tuple[int, int]]]:
    """
    [신규] Hard Z 로직(요청 4)에 맞는 2개의 인덱스를 샘플링합니다.
    (이미지 하단(y큼)에 있지만 Z좌표(z큼)가 더 큰 경우)
    """
    for _ in range(MAX_TRIES_HARD_Z_SAMPLING):
        centers = _sample_valid_indices(2, valid_indices)
        if not centers:
            continue

        (y1_c, x1_c), (y2_c, x2_c) = centers

        # 490 그리드 기준 경계
        y1_min, y1_max = y1_c - HALF_WIN, y1_c + HALF_WIN
        x1_min, x1_max = x1_c - HALF_WIN, x1_c + HALF_WIN
        y2_min, y2_max = y2_c - HALF_WIN, y2_c + HALF_WIN
        x2_min, x2_max = x2_c - HALF_WIN, x2_c + HALF_WIN

        # Z 좌표 (point_map의 3번째 채널, +Z=Up)
        z1 = np.mean(point_map[y1_min:y1_max+1, x1_min:x1_max+1, 2])
        z2 = np.mean(point_map[y2_min:y2_max+1, x2_min:x2_max+1, 2])

        if abs(z1 - z2) < MIN_COORD_DIFF:
            continue

        # Hard Z 조건 검사:
        # (A가 B보다 이미지 하단에 있는데, Z도 더 높음) OR (B가 A보다 하단에 있는데, Z도 더 높음)
        if (y1_c > y2_c and z1 > z2) or (y2_c > y1_c and z2 > z1):
            return centers # 성공

    return None # 실패

def _calculate_relative_data(
    center_bbox: BBoxData,
    comparison_bboxes: List[BBoxData],
    remap_tuples: Dict[str, tuple]
):
    """
    [신규] Relative 로직(요청 2)을 위해 BBox들의 좌표를 기준 BBox 기준으로 재계산합니다.
    (BBoxData 객체를 직접 수정합니다)
    """
    for bbox in comparison_bboxes:
        # 기준점과의 3D 좌표 차이
        delta_coords = bbox.mean_coords - center_bbox.mean_coords

        # 3D 거리 (Range) 재계산
        bbox.mean_range_mm = int(np.linalg.norm(delta_coords) * 1000.0)

        # 리맵핑된 좌표 (X, Y, Z) 재계산
        bbox.remapped_coords = {
            'x_new': remap_coord(delta_coords, remap_tuples['x']),
            'y_new': remap_coord(delta_coords, remap_tuples['y']),
            'z_new': remap_coord(delta_coords, remap_tuples['z']),
        }

def _format_bbox_str_list(bbox_strs: List[str]) -> str:
    """
    [신규] BBox 문자열 리스트를 "A, B, and C" 형식으로 변환합니다. (요청 1)
    """
    if not bbox_strs:
        return ""
    if len(bbox_strs) == 1:
        return bbox_strs[0]
    if len(bbox_strs) == 2:
        return f"{bbox_strs[0]} and {bbox_strs[1]}"

    return ", ".join(bbox_strs[:-1]) + f", and {bbox_strs[-1]}"

def _get_qa_criterion(qa_type: str) -> Dict[str, Any]:
    """
    [신규] QA 유형별 비교 함수, 최소 차이, 질문 템플릿을 반환합니다. (요청 3)
    """
    if qa_type == 'range':
        return {
            'func': lambda b: b.mean_range_mm,
            'min_diff': MIN_RANGE_DIFF,
            'str_comp': 'farther',                # 비교급
            'str_inv_comp': 'closer',             # 비교급 (반대)
            'str_super': 'farthest',              # 최상급
            'str_inv_super': 'closest'            # 최상급 (반대)
        }
    elif qa_type == 'x':
        # X 좌표는 -z_new (NewZ=-X)에 대응
        return {
            'func': lambda b: -b.remapped_coords['z_new'],
            'min_diff': MIN_COORD_DIFF,
            'str_comp': 'has a larger X coordinate',
            'str_inv_comp': 'has a smaller X coordinate',
            'str_super': 'has the largest X coordinate',      # 'a' -> 'the'
            'str_inv_super': 'has the smallest X coordinate'  # 'a' -> 'the'
        }
    elif qa_type == 'y':
        # Y 좌표는 x_new (NewX=+Y)에 대응
        return {
            'func': lambda b: b.remapped_coords['x_new'],
            'min_diff': MIN_COORD_DIFF,
            'str_comp': 'has a larger Y coordinate',
            'str_inv_comp': 'has a smaller Y coordinate',
            'str_super': 'has the largest Y coordinate',
            'str_inv_super': 'has the smallest Y coordinate'
        }
    elif qa_type == 'z':
        # Z 좌표는 y_new (NewY=+Z)에 대응
        return {
            'func': lambda b: b.remapped_coords['y_new'],
            'min_diff': MIN_COORD_DIFF,
            'str_comp': 'has a larger Z coordinate',
            'str_inv_comp': 'has a smaller Z coordinate',
            'str_super': 'has the largest Z coordinate',
            'str_inv_super': 'has the smallest Z coordinate'
        }
    raise ValueError(f"Unknown qa_type: {qa_type}")

def _generate_qa_minmax(
    bboxes: List[BBoxData],
    qa_type: str,
    view_name: str,
    preamble: str,
    center_bbox: Optional[BBoxData] = None, # [요청 2]
    is_hard_z: bool = False # [요청 4]
) -> Optional[Tuple[str, str, str]]:
    """
    [신규] K개의 BBox 중 Min/Max를 찾는 질문(Standard, N-ary, Relative, Hard Z)을 생성합니다.
    (요청 1, 2, 4)
    """

    criterion = _get_qa_criterion(qa_type)
    target_value_func = criterion['func']
    min_diff = criterion['min_diff']

    # --- Hard Z 로직 (요청 4) ---
    if is_hard_z:
        if len(bboxes) != 2 or not qa_type == 'z':
            return None # Hard Z는 2개 BBox, Z타입 전용

        bbox_a, bbox_b = bboxes
        z_a = target_value_func(bbox_a)
        z_b = target_value_func(bbox_b)

        # Hard Z 조건 (이미지 y, 3D z)
        # (A가 하단(y큼) & Z높음) 또는 (B가 하단 & Z높음)
        label_bbox = None
        if bbox_a.y_c > bbox_b.y_c and z_a > z_b:
            label_bbox = bbox_a
        elif bbox_b.y_c > bbox_a.y_c and z_b > z_a:
            label_bbox = bbox_b

        if label_bbox:
            # 질문은 항상 "larger Z"로 고정 (BBox 2개이므로 비교급 사용)
            question_str = criterion['str_comp'] # 'str' -> 'str_comp'
            bbox_strs = _format_bbox_str_list([b.bbox_str for b in bboxes])
            prompt = (
                f"{preamble}Which of {bbox_strs} {question_str}?" # "has the" 중복 제거
            ).strip()
            return prompt, label_bbox.bbox_str, f"Z_hard_{view_name}"
        else:
            return None # 샘플링이 잘못된 경우

    # --- 일반 Min/Max 로직 (Standard, N-ary, Relative) ---

    num_bboxes = len(bboxes) # [신규] BBox 개수 확인

    is_question_inverted = random.random() < 0.5 # 50% 확률로 "smaller/closer"

    # [수정] BBox 개수에 따라 비교급(2개) 또는 최상급(3개+) 선택
    if num_bboxes > 2:
        question_type_str = criterion['str_inv_super'] if is_question_inverted else criterion['str_super']
    else: # 2개 (또는 1개)
        question_type_str = criterion['str_inv_comp'] if is_question_inverted else criterion['str_comp']

    values = [target_value_func(b) for b in bboxes]

    # Min/Max 값과 인덱스 찾기
    if is_question_inverted: # "smaller / closer"
        target_idx = np.argmin(values)
        target_val = values[target_idx]
        # 다른 모든 값들보다 충분히 작은지 확인
        is_valid = all(v - target_val >= min_diff for i, v in enumerate(values) if i != target_idx)
    else: # "larger / farther"
        target_idx = np.argmax(values)
        target_val = values[target_idx]
        # 다른 모든 값들보다 충분히 큰지 확인
        is_valid = all(target_val - v >= min_diff for i, v in enumerate(values) if i != target_idx)

    if not is_valid:
        return None # 유효한 차이를 가진 정답이 없음

    label = bboxes[target_idx].bbox_str

    # --- 프롬프트 생성 (요청 2: Relative 반영) ---
    bbox_strs = _format_bbox_str_list([b.bbox_str for b in bboxes])

    # Relative(중심) 질문 템플릿
    if center_bbox:
        center_preamble = f"From the perspective of {center_bbox.bbox_str}, "

        # 'range' 질문은 템플릿이 다름
        if qa_type == 'range':
            center_str = f"relative to {center_bbox.bbox_str}"
            prompt = f"{' '.join([preamble, center_preamble]).strip()} Which of {bbox_strs} is {question_type_str} {center_str}?"
        else:
            prompt = f"{' '.join([preamble, center_preamble]).strip()} Which of {bbox_strs} {question_type_str}?"

    # Standard / N-ary 질문 템플릿
    else:
        if qa_type == 'range':
            center_str = "from the origin in 3D space"
            prompt = f"{preamble}Which of {bbox_strs} is {question_type_str} {center_str}?"
        else:
            prompt = f"{preamble}Which of {bbox_strs} {question_type_str}?"

    prompt = ' '.join(prompt.split()) # 여백 정리

    # Suffix 생성
    suffix = f"{qa_type.upper()}_{view_name}"
    if center_bbox:
        suffix += "_Rel"
    if len(bboxes) > 2:
        suffix += f"_K{len(bboxes)}"

    return prompt, label, suffix


def _generate_qa_combined(
    bboxes: List[BBoxData],
    qa_type_1: str,
    view_name: str,
    preamble: str
) -> Optional[Tuple[str, str, str]]:
    """
    [신규] 2개의 BBox에 대해 2개의 속성을 조합하는 질문을 생성합니다. (요청 3)
    """
    if len(bboxes) != 2:
        return None

    bbox_a, bbox_b = bboxes

    # 두 번째 QA 속성 랜덤 선택 (첫 번째와 중복되지 않게)
    available_types = [t for t in ['range', 'x', 'y', 'z'] if t != qa_type_1]
    if not available_types:
        return None
    qa_type_2 = random.choice(available_types)

    # 각 속성에 대한 기준(함수, 템플릿) 가져오기
    crit_1 = _get_qa_criterion(qa_type_1)
    crit_2 = _get_qa_criterion(qa_type_2)

    # 각 속성을 반전시킬지(smaller) 여부
    is_inv_1 = random.random() < 0.5
    is_inv_2 = random.random() < 0.5

    # 비교 함수 (less or greater)
    op_1 = np.less if is_inv_1 else np.greater
    op_2 = np.less if is_inv_2 else np.greater

    # 질문 문자열
    str_1 = crit_1['str_inv_comp'] if is_inv_1 else crit_1['str_comp']
    str_2 = crit_2['str_inv_comp'] if is_inv_2 else crit_2['str_comp']

    # 값 계산
    val_a_1 = crit_1['func'](bbox_a)
    val_b_1 = crit_1['func'](bbox_b)
    val_a_2 = crit_2['func'](bbox_a)
    val_b_2 = crit_2['func'](bbox_b)

    # 두 속성 모두 최소 차이 이상인지 확인
    if not (abs(val_a_1 - val_b_1) >= crit_1['min_diff'] and \
            abs(val_a_2 - val_b_2) >= crit_2['min_diff']):
        return None

    # A가 두 조건 모두 만족하는지
    a_wins = op_1(val_a_1, val_b_1) and op_2(val_a_2, val_b_2)
    # B가 두 조건 모두 만족하는지
    b_wins = op_1(val_b_1, val_a_1) and op_2(val_b_2, val_a_2)

    label = None
    if a_wins and not b_wins:
        label = bbox_a.bbox_str
    elif b_wins and not a_wins:
        label = bbox_b.bbox_str
    else:
        return None # 둘 다 이기거나 둘 다 지면 유효하지 않은 질문

    bbox_strs = _format_bbox_str_list([b.bbox_str for b in bboxes])
    prompt = f"{preamble}Which of {bbox_strs} {str_1} AND {str_2}?"
    prompt = ' '.join(prompt.split()) # 여백 정리

    suffix = f"Combined_{view_name}_{qa_type_1}_{qa_type_2}"

    return prompt, label, suffix

# --- 6. 메인 QA 생성 함수 (대폭 수정됨) ---

def generate_single_qa_with_vis_aug(
    xyz_tensor: torch.Tensor,
    config: QAConfig, # [신규] 설정 객체 (요청 5)
    viz_align_data=False
) -> (str, str):
    """
    [수정됨] XYZ 텐서와 QA 설정을 입력받아,
    설정된 확률에 따라 다양한 유형(Standard, N-ary, Relative, Combined, Hard Z)의
    (label, prompt) 문자열 쌍을 생성하고 관련 시각화 파일(.png, .ply)을 저장합니다.
    """

    # --- 6.1. 입력 텐서 검증 및 변환 ---
    if xyz_tensor.shape != (IMG_H, IMG_W, 3):
        err_msg = f"입력 텐서 크기가 잘못되었습니다. (XYZ: {xyz_tensor.shape})"
        print(f"Error: {err_msg}")
        return f"Error: {err_msg}", f"Error: {err_msg}"

    if HALF_WIN == 0:
        print("Warning: window sizes below 3 use a single pixel.")

    try:
        point_map = xyz_tensor.cpu().numpy().astype(np.float32)
        range_map_m = np.linalg.norm(point_map, axis=2)
        range_map_mm = (range_map_m * 1000.0).astype(np.uint16)

    except Exception as e:
        err_msg = f"Tensor conversion or range calculation failed: {e}"
        print(f"Error: {err_msg}")
        return f"Error: {err_msg}", f"Error: {err_msg}"

    # --- 6.2. 유효한 샘플링 위치 찾기 ---
    kernel = np.ones((WINDOW_SIZE, WINDOW_SIZE), dtype=np.uint8)
    valid_mask_8u = (range_map_mm > 0).astype(np.uint8)
    valid_mask_pooled = cv2.erode(valid_mask_8u, kernel, iterations=1,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
    valid_indices = np.argwhere(valid_mask_pooled > 0)

    if len(valid_indices) < 2:
        err_msg = "유효한 BBox 중심점 후보가 2개 미만입니다."
        print(f"Error: {err_msg}")
        return f"Error: {err_msg}", f"Error: {err_msg}"

    # --- 6.3. 유효한 QA 쌍 찾기 (최대 N회 시도) ---
    prompt = None
    label = None
    generated_bboxes: List[BBoxData] = []
    dataset_name_suffix = ""

    for _ in range(MAX_TRIES_PER_CALL):

        # 6.3.1. QA 기본 유형 및 시점(View) 무작위 선택
        qa_type = random.choice(['range', 'x', 'y', 'z'])

        if qa_type == 'range':
            view_name = 'default' # 'range' 질문은 회전과 무관
        else:
            view_name = random.choice(list(ROTATION_VIEWS.keys()))

        preamble, (x_remap, y_remap, z_remap) = ROTATION_VIEWS[view_name]
        remap_tuples = {'x': x_remap, 'y': y_remap, 'z': z_remap}

        # 6.3.2. QA 생성 전략 결정 (대안: random.choices)

        # --- 1. 가능한 모든 전략과 가중치를 리스트업합니다. ---
        strategies_list = []
        weights_list = []

        # K=2 전략
        if config.hard_z_enabled and qa_type == 'z':
            strategies_list.append('hard_z')
            weights_list.append(config.hard_z_prob)

        if config.combined_enabled:
            strategies_list.append('combined')
            weights_list.append(config.combined_prob)

        # K>2 전략
        if config.relative_enabled:
            strategies_list.append('relative')
            weights_list.append(config.relative_prob)

        if config.n_ary_enabled:
            strategies_list.append('n_ary')
            weights_list.append(config.n_ary_prob)

        # 'Standard K=2' 전략 가중치 계산
        total_special_weight = sum(weights_list)
        weight_standard_k2 = max(0.0, 1.0 - total_special_weight)

        strategies_list.append('standard')
        weights_list.append(weight_standard_k2)

        # --- 2. 가중치를 기반으로 전략을 하나 선택합니다. ---
        try:
            # k=1은 strategy 리스트에서 하나를 뽑으라는 의미
            strategy = random.choices(strategies_list, weights=weights_list, k=1)[0]
        except ValueError:
            # weights_list의 합이 0인 경우 (모든 prob=0, 1-total=0)
            strategy = 'standard'

        # --- 3. 선택된 전략에 따라 K, is_relative, is_hard_z 설정 ---
        is_relative = False
        is_hard_z = False

        if strategy == 'hard_z':
            k = 2
            is_hard_z = True
        elif strategy == 'combined':
            k = 2
        elif strategy == 'relative':
            k = random.choice(config.n_ary_k_choices) # 3 or 4
            is_relative = True
        elif strategy == 'n_ary':
            k = random.choice(config.n_ary_k_choices) # 3 or 4
        else: # 'standard'
            k = 2

        # 6.3.3. BBox 중심점 샘플링
        sampled_centers = None
        if is_hard_z:
            sampled_centers = _sample_hard_z_indices(valid_indices, point_map)
        else:
            sampled_centers = _sample_valid_indices(k, valid_indices)

        if not sampled_centers:
            continue # 샘플링 실패, 다음 시도

        # 6.3.4. BBox 데이터 계산
        try:
            bboxes_data = [_get_bbox_data(center, point_map, range_map_mm, remap_tuples)
                           for center in sampled_centers]
        except Exception:
            continue # 데이터 계산 중 오류

        # 6.3.5. QA 유형별 로직 수행
        center_bbox = None
        comparison_bboxes = bboxes_data

        # Relative (요청 2) 로직
        if is_relative:
            center_bbox = bboxes_data[0]      # 첫 번째 BBox를 기준점으로
            comparison_bboxes = bboxes_data[1:] # 나머지 BBox들을 비교 대상으로
            if not comparison_bboxes:
                continue # 비교 대상이 없으면 (K=1이었으면) 스킵

            _calculate_relative_data(center_bbox, comparison_bboxes, remap_tuples)

        # QA 생성 시도
        result = None
        if strategy == 'combined':
            result = _generate_qa_combined(comparison_bboxes, qa_type, view_name, preamble)
        else:
            # Standard, N-ary, Relative, Hard Z 모두 min/max 로직 사용
            result = _generate_qa_minmax(comparison_bboxes, qa_type, view_name,
                                           preamble, center_bbox, is_hard_z)

        # 6.3.6. 생성 성공 시
        if result:
            prompt, label, dataset_name_suffix = result
            generated_bboxes = bboxes_data # 시각화를 위해 저장
            break # 성공, 메인 루프 탈출

    # --- 6.4. QA 생성 실패 처리 ---
    if not prompt or not label:
        err_msg = f"유효한 QA 쌍 생성 실패 (시도 {MAX_TRIES_PER_CALL}회 초과)"
        print(f"Error: {err_msg}")
        return f"Error: {err_msg}", f"Error: {err_msg}"

    # --- 6.5. 시각화 생성 ---
    if viz_align_data:
        _save_visualizations(
            prompt=prompt,
            label_bbox_str=label,
            bboxes=generated_bboxes, # [수정] K개의 BBox 리스트 전달
            point_map=point_map,
            range_map_mm=range_map_mm,
            dataset_name_suffix=dataset_name_suffix
        )

    # --- 6.6. 결과 반환 ---
    return label, prompt
