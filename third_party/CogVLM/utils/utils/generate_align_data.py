import cv2
import numpy as np
import torch
from pathlib import Path
import random
import textwrap
import uuid # 고유 해시 생성을 위해 uuid 임포트

# 1. 시각화 결과를 저장할 디렉토리
VISUALIZATION_DIR = Path("visualizations_single_qa")

# 2. "더 멀다"고 판단하기 위한 최소 거리 차이 (mm)
# 입력 range_tensor가 밀리미터(mm) 단위라고 가정합니다.
MIN_RANGE_DIFF = 50  # 5cm

# 3. X, Y, Z 좌표 비교를 위한 최소 차이 (meters)
# 입력 xyz_tensor가 미터(m) 단위라고 가정합니다.
MIN_COORD_DIFF = 0.05 # 5cm

# 4. 윈도우(BBox) 크기 (홀수 권장)
WINDOW_SIZE = 3

# 5. 원본 이미지 및 타겟 그리드 해상도
IMG_H, IMG_W = 490, 490
GRID_H, GRID_W = 1000, 1000

# 6. 2D 시각화 시 상단 텍스트 영역 높이
TEXT_AREA_HEIGHT = 100 # 100 픽셀

# 7. QA 쌍 생성을 위한 최대 시도 횟수
MAX_TRIES_PER_CALL = 50

# --- 2. 시각화 색상 정의 ---
COLOR_ANSWER_RGB_O3D = [0.0, 1.0, 0.0]   # 초록색 (정답)
COLOR_OTHER_RGB_O3D = [1.0, 0.0, 0.0]    # 빨간색 (오답)
COLOR_SCENE_RGB_O3D = [0.7, 0.7, 0.7]    # 회색 (씬)

COLOR_ANSWER_BGR_CV = (0, 255, 0)      # 초록색 (정답)
COLOR_OTHER_BGR_CV = (0, 0, 255)       # 빨간색 (오답)
COLOR_TEXT_CV = (255, 255, 255)      # 흰색 (텍스트)


# --- 3. 헬퍼 함수 (좌표 변환) ---

def scale_coordinates(y, x, img_h, img_w, grid_h, grid_w):
    """
    (y, x) 좌표를 원본 이미지(490)에서 타겟 그리드(1000)로 스케일링합니다.
    (y_scaled, x_scaled) 반환
    """
    y_scaled = int(round((y / (img_h - 1)) * (grid_h - 1)))
    x_scaled = int(round((x / (img_w - 1)) * (grid_w - 1)))
    return y_scaled, x_scaled

def rescale_coordinates_to_img(x_s, y_s, img_h, img_w, grid_h, grid_w):
    """
    1000x1000 그리드 좌표(x_s, y_s)를 원본 490x490 이미지 좌표(x_img, y_img)로 변환합니다.
    (x, y) 순서로 반환합니다.
    """
    y_img = int(round((y_s / (grid_h - 1)) * (img_h - 1)))
    x_img = int(round((x_s / (grid_w - 1)) * (img_w - 1)))
    return x_img, y_img # (x, y) 순서

def rescale_bbox_to_img(bbox_s, img_h, img_w, grid_h, grid_w):
    """
    1000x1000 BBox 좌표를 490x490 이미지 BBox 좌표(pt1, pt2)로 변환합니다.
    bbox_s: (x_min_s, y_min_s, x_max_s, y_max_s)
    """
    x_min_s, y_min_s, x_max_s, y_max_s = bbox_s

    x_min_img, y_min_img = rescale_coordinates_to_img(x_min_s, y_min_s, img_h, img_w, grid_h, grid_w)
    x_max_img, y_max_img = rescale_coordinates_to_img(x_max_s, y_max_s, img_h, img_w, grid_h, grid_w)

    # (x_min, y_min), (x_max, y_max)
    return (x_min_img, y_min_img), (x_max_img, y_max_img)

# --- 4. 메인 QA 생성 함수 ---

def generate_single_qa_with_vis(range_tensor: torch.Tensor, xyz_tensor: torch.Tensor, viz_align_data=False) -> (str, str):
    """
    Range(거리) 텐서와 XYZ(좌표) 텐서를 입력받아,
    무작위로 하나의 QA 유형(Range, X, Y, Z, Z_Hard)을 선택하여
    (label, prompt) 문자열 쌍을 생성하고 관련 시각화 파일(.png, .ply)을 저장합니다.

    Args:
        range_tensor (torch.Tensor): [490, 490] 크기. 거리 값 (mm 단위, uint16 가정)
        xyz_tensor (torch.Tensor): [490, 490, 3] 크기. 3D 좌표 값 (m 단위, float32 가정)

    Returns:
        (str, str): (label, prompt) 튜플.
                    label: 정답 BBox 문자열 (예: "[[123,456,129,462]]")
                    prompt: 생성된 질문 문자열
                    실패 시: ("Error: ...", "Error: ...")
    """

    # --- 4.1. 입력 텐서 검증 및 변환 ---
    if range_tensor.shape != (IMG_H, IMG_W) or xyz_tensor.shape != (IMG_H, IMG_W, 3):
        err_msg = f"입력 텐서 크기가 잘못되었습니다. (Range: {range_tensor.shape}, XYZ: {xyz_tensor.shape})"
        print(f"Error: {err_msg}")
        return f"Error: {err_msg}", f"Error: {err_msg}"

    try:
        # 텐서를 CPU의 NumPy 배열로 변환
        range_map = range_tensor.cpu().numpy().astype(np.uint16)
        point_map = xyz_tensor.cpu().numpy().astype(np.float32)
    except Exception as e:
        err_msg = f"Tensor conversion failed: {e}"
        print(f"Error: {err_msg}")
        return f"Error: {err_msg}", f"Error: {err_msg}"

    # --- 4.2. 유효한 샘플링 위치 찾기 ---
    half_win = WINDOW_SIZE // 2
    if half_win == 0:
        print("Warning: window sizes below 3 use a single pixel.")
        half_win = 0 # 1x1 픽셀 처리

    kernel = np.ones((WINDOW_SIZE, WINDOW_SIZE), dtype=np.uint8)

    # 거리 값이 0보다 큰(유효한) 픽셀 마스크
    valid_mask_8u = (range_map > 0).astype(np.uint8)

    # 윈도우 전체가 유효한 영역에 속하도록 마스크를 축소(erode)
    valid_mask_pooled = cv2.erode(valid_mask_8u, kernel, iterations=1,
                                borderType=cv2.BORDER_CONSTANT, borderValue=0)
    valid_indices = np.argwhere(valid_mask_pooled > 0)

    if len(valid_indices) < 2:
        err_msg = "유효한 BBox 중심점 후보가 2개 미만입니다."
        print(f"Error: {err_msg}")
        return f"Error: {err_msg}", f"Error: {err_msg}"

    # --- 4.3. QA 유형 무작위 선택 ---
    qa_type = random.choice(['range', 'x', 'y', 'z'])
    is_hard_z_attempt = False
    if qa_type == 'z' and random.random() < 0.5:
        is_hard_z_attempt = True # 50% 확률로 Z-Hard 질문 시도

    prompt = None
    label = None
    dataset_name_suffix = ""

    # 루프 시작 전, 'z' 타입일 경우를 대비한 임시 변수 초기화
    fallback_z_prompt, fallback_z_label, fallback_z_suffix = None, None, None

    # --- 4.4. 유효한 QA 쌍 찾기 (최대 N회 시도) ---
    for _ in range(MAX_TRIES_PER_CALL):
        try:
            # 4.4.1. 유효한 인덱스 2개 랜덤 샘플링
            idx1, idx2 = random.sample(range(len(valid_indices)), 2)
        except ValueError:
            break # 샘플링 실패 (이론상 발생하기 어려움)

        y1_c, x1_c = valid_indices[idx1]
        y2_c, x2_c = valid_indices[idx2]

        # 4.4.2. BBox 경계 계산 (490x490 이미지 기준)
        y1_min, y1_max = y1_c - half_win, y1_c + half_win
        x1_min, x1_max = x1_c - half_win, x1_c + half_win
        y2_min, y2_max = y2_c - half_win, y2_c + half_win
        x2_min, x2_max = x2_c - half_win, x2_c + half_win

        # 4.4.3. BBox 내부 데이터 평균 계산
        win_A_range = range_map[y1_min:y1_max+1, x1_min:x1_max+1]
        win_B_range = range_map[y2_min:y2_max+1, x2_min:x2_max+1]
        range_A = int(np.mean(win_A_range))
        range_B = int(np.mean(win_B_range))

        win_A_coords = point_map[y1_min:y1_max+1, x1_min:x1_max+1, :]
        win_B_coords = point_map[y2_min:y2_max+1, x2_min:x2_max+1, :]
        mean_coords_A = np.mean(win_A_coords, axis=(0, 1))
        mean_coords_B = np.mean(win_B_coords, axis=(0, 1))
        x_A, y_A, z_A = mean_coords_A
        x_B, y_B, z_B = mean_coords_B

        # 4.4.4. BBox 좌표 스케일링 (1000x1000 그리드 기준)
        y1_min_s, x1_min_s = scale_coordinates(y1_min, x1_min, IMG_H, IMG_W, GRID_H, GRID_W)
        y1_max_s, x1_max_s = scale_coordinates(y1_max, x1_max, IMG_H, IMG_W, GRID_H, GRID_W)
        y2_min_s, x2_min_s = scale_coordinates(y2_min, x2_min, IMG_H, IMG_W, GRID_H, GRID_W)
        y2_max_s, x2_max_s = scale_coordinates(y2_max, x2_max, IMG_H, IMG_W, GRID_H, GRID_W)

        # (x_min, y_min, x_max, y_max) 튜플
        bbox_A_s_coords = (x1_min_s, y1_min_s, x1_max_s, y1_max_s)
        bbox_B_s_coords = (x2_min_s, y2_min_s, x2_max_s, y2_max_s)

        bbox_A_str = f"[[{x1_min_s:03d},{y1_min_s:03d},{x1_max_s:03d},{y1_max_s:03d}]]"
        bbox_B_str = f"[[{x2_min_s:03d},{y2_min_s:03d},{x2_max_s:03d},{y2_max_s:03d}]]"

        # 4.4.5. 선택된 QA 유형에 따라 조건 검사 및 생성

        # 50% 확률로 질문 의미 반전 (larger -> smaller)
        is_question_inverted = random.random() < 0.5

        if qa_type == 'range':
            if abs(range_A - range_B) >= MIN_RANGE_DIFF:
                if is_question_inverted: # "closer" (더 가까운)
                    label = bbox_A_str if range_A < range_B else bbox_B_str
                    prompt = (
                        f"Which of {bbox_A_str} and {bbox_B_str} is closer to "
                        "the origin in 3D space?"
                    )
                else: # "farther" (더 먼)
                    label = bbox_A_str if range_A > range_B else bbox_B_str
                    prompt = (
                        f"Which of {bbox_A_str} and {bbox_B_str} is farther away "
                        "from the origin in 3D space?"
                    )
                dataset_name_suffix = 'Range'
                break # 성공, 루프 탈출

        elif qa_type == 'x':
            if abs(x_A - x_B) >= MIN_COORD_DIFF:
                if is_question_inverted: # "smaller" (더 작은 X)
                    label = bbox_A_str if x_A < x_B else bbox_B_str
                    prompt = (
                        f"Which of {bbox_A_str} and {bbox_B_str} has a smaller "
                        "X coordinate in 3D space?"
                    )
                else: # "larger" (더 큰 X)
                    label = bbox_A_str if x_A > x_B else bbox_B_str
                    prompt = (
                        f"Which of {bbox_A_str} and {bbox_B_str} has a larger "
                        "X coordinate in 3D space?"
                    )
                dataset_name_suffix = 'X'
                break # 성공, 루프 탈출

        elif qa_type == 'y':
            if abs(y_A - y_B) >= MIN_COORD_DIFF:
                if is_question_inverted: # "smaller" (더 작은 Y)
                    label = bbox_A_str if y_A < y_B else bbox_B_str
                    prompt = (
                        f"Which of {bbox_A_str} and {bbox_B_str} has a smaller "
                        "Y coordinate in 3D space?"
                    )
                else: # "larger" (더 큰 Y)
                    label = bbox_A_str if y_A > y_B else bbox_B_str
                    prompt = (
                        f"Which of {bbox_A_str} and {bbox_B_str} has a larger "
                        "Y coordinate in 3D space?"
                    )
                dataset_name_suffix = 'Y'
                break # 성공, 루프 탈출

        elif qa_type == 'z':
            # [수정] Z 좌표 차이 조건부터 먼저 확인
            if abs(z_A - z_B) >= MIN_COORD_DIFF:

                # "Hard" Z 조건 계산
                is_hard_case_1 = (z_A > z_B) and (y1_c > y2_c)
                is_hard_case_2 = (z_B > z_A) and (y2_c > y1_c)
                is_current_pair_hard = is_hard_case_1 or is_hard_case_2

                # Z-Hard를 시도했는데 Hard가 아닌 경우
                if is_hard_z_attempt and not is_current_pair_hard:
                    # [수정] 이 쌍을 유효한 '일반 Z' 폴백(fallback)으로 저장
                    if is_question_inverted: # "smaller"
                        fallback_z_label = bbox_A_str if z_A < z_B else bbox_B_str
                        fallback_z_prompt = (
                            f"Which of {bbox_A_str} and {bbox_B_str} has a smaller "
                            "Z coordinate in 3D space?"
                        )
                    else: # "larger"
                        fallback_z_label = bbox_A_str if z_A > z_B else bbox_B_str
                        fallback_z_prompt = (
                            f"Which of {bbox_A_str} and {bbox_B_str} has a larger "
                            "Z coordinate in 3D space?"
                        )
                    fallback_z_suffix = 'Z' # Hard가 아니므로 'Z'
                    continue # Hard 쌍을 찾기 위해 다음 루프 시도

                # Z-Hard 시도가 아니었거나 (is_hard_z_attempt=False),
                # Z-Hard 시도였고 Hard 조건을 만족한 경우 (is_hard_z_attempt=True, is_current_pair_hard=True)

                if is_question_inverted: # "smaller"
                    label = bbox_A_str if z_A < z_B else bbox_B_str
                    prompt = (
                        f"Which of {bbox_A_str} and {bbox_B_str} has a smaller "
                        "Z coordinate in 3D space?"
                    )
                else: # "larger"
                    label = bbox_A_str if z_A > z_B else bbox_B_str
                    prompt = (
                        f"Which of {bbox_A_str} and {bbox_B_str} has a larger "
                        "Z coordinate in 3D space?"
                    )

                dataset_name_suffix = 'Z_Hard' if is_current_pair_hard else 'Z'
                break # 성공, 루프 탈출

    # --- 4.5. QA 생성 실패 처리 ---

    # [수정] 루프 종료 후, prompt가 비어있고(성공 못함)
    #       qa_type이 'z'였으며, 임시 변수에 저장된 값이 있다면
    #       그 값을 최종 결과로 사용합니다.
    if not prompt and qa_type == 'z' and fallback_z_prompt:
            # 필요하다면 조건 미달 쌍이 사용되었음을 알리는 경고를 출력할 수 있습니다.
            # print(f"Warning: Z-Hard QA generation failed to find 'Hard' pair after {MAX_TRIES_PER_CALL} tries. "
            #     f"Falling back to a valid 'Z' pair.")
            prompt = fallback_z_prompt
            label = fallback_z_label
            dataset_name_suffix = fallback_z_suffix

    # [수정] 이 시점에서 prompt가 채워졌으므로 (z타입의 경우)
    #       아래의 에러 리턴을 통과하게 됩니다.
    if not prompt or not label:
        err_msg = f"유효한 QA 쌍 생성 실패 (시도 {MAX_TRIES_PER_CALL}회 초과, 타입: {qa_type})"
        print(f"Error: {err_msg}")
        return f"Error: {err_msg}", f"Error: {err_msg}"

    # --- 4.6. 시각화 생성 ---

    if viz_align_data:
        VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
        # 고유 해시 생성
        unique_hash = uuid.uuid4().hex[:16]
        base_output_name = f"vis_{dataset_name_suffix}_{unique_hash}"

        # 정답 여부에 따른 색상 결정
        is_A_answer = (label == bbox_A_str)
        color_a_o3d, color_b_o3d = (COLOR_ANSWER_RGB_O3D, COLOR_OTHER_RGB_O3D) if is_A_answer else (COLOR_OTHER_RGB_O3D, COLOR_ANSWER_RGB_O3D)
        color_a_cv, color_b_cv = (COLOR_ANSWER_BGR_CV, COLOR_OTHER_BGR_CV) if is_A_answer else (COLOR_OTHER_BGR_CV, COLOR_ANSWER_BGR_CV)

        # 시각화용 BBox 좌표 (490x490 이미지 기준)
        (x1_min_img, y1_min_img), (x1_max_img, y1_max_img) = rescale_bbox_to_img(
            bbox_A_s_coords, IMG_H, IMG_W, GRID_H, GRID_W)
        (x2_min_img, y2_min_img), (x2_max_img, y2_max_img) = rescale_bbox_to_img(
            bbox_B_s_coords, IMG_H, IMG_W, GRID_H, GRID_W)

        # --- 4.6.1. 2D 이미지 시각화 (텍스트 포함) ---
        try:
            # 0 제외 Min-Max 정규화
            vis_img_normalized = np.zeros_like(range_map, dtype=np.float32)
            valid_ranges = range_map[range_map > 0]

            if len(valid_ranges) > 0:
                min_val, max_val = np.min(valid_ranges), np.max(valid_ranges)
                range_val = float(max_val - min_val)
                if range_val > 0:
                    mask_valid = range_map > 0
                    vis_img_normalized[mask_valid] = (range_map[mask_valid].astype(np.float32) - min_val) / range_val
                else:
                    vis_img_normalized[range_map > 0] = 1.0

            vis_img_8u = (vis_img_normalized * 255.0).astype(np.uint8)
            vis_img_bgr = cv2.cvtColor(vis_img_8u, cv2.COLOR_GRAY2BGR)

            # BBox 그리기
            cv2.rectangle(vis_img_bgr, (x1_min_img, y1_min_img), (x1_max_img, y1_max_img), color_a_cv, thickness=1)
            cv2.rectangle(vis_img_bgr, (x2_min_img, y2_min_img), (x2_max_img, y2_max_img), color_b_cv, thickness=1)

            # 텍스트 영역 추가
            final_vis_img = np.zeros((IMG_H + TEXT_AREA_HEIGHT, IMG_W, 3), dtype=np.uint8)
            final_vis_img[TEXT_AREA_HEIGHT:, :, :] = vis_img_bgr

            # 텍스트 래핑 및 추가
            lines = textwrap.wrap(prompt, width=50) # 너비 50자로 줄바꿈
            y_text_start = 20
            line_height = 20

            for i, line in enumerate(lines):
                if i * line_height > TEXT_AREA_HEIGHT - line_height: break
                y_pos = y_text_start + i * line_height
                cv2.putText(final_vis_img, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, COLOR_TEXT_CV, 1, cv2.LINE_AA)

            # 2D 이미지 저장
            save_path_img = VISUALIZATION_DIR / (base_output_name + ".png")
            cv2.imwrite(str(save_path_img), final_vis_img)

        except Exception as e:
            print(f"Warning: failed to save 2D visualization {base_output_name}.png: {e}")

        # --- 4.6.2. 3D PLY 시각화 ---
        try:
            points_flat = point_map.reshape(-1, 3)
            colors_flat = np.full(points_flat.shape, COLOR_SCENE_RGB_O3D, dtype=np.float64)

            # BBox A 영역 색상 적용
            mask_A = np.zeros((IMG_H, IMG_W), dtype=bool)
            mask_A[y1_min_img:y1_max_img+1, x1_min_img:x1_max_img+1] = True
            colors_flat[mask_A.flatten()] = color_a_o3d

            # BBox B 영역 색상 적용
            mask_B = np.zeros((IMG_H, IMG_W), dtype=bool)
            mask_B[y2_min_img:y2_max_img+1, x2_min_img:x2_max_img+1] = True
            colors_flat[mask_B.flatten()] = color_b_o3d

            # 유효한(거리 > 0) 포인트만 필터링
            valid_mask_flat = (range_map.flatten() > 0)
            final_points = points_flat[valid_mask_flat]
            final_colors = colors_flat[valid_mask_flat]

            if len(final_points) == 0:
                raise ValueError("유효한 3D 포인트가 없습니다.")

            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(final_points)
            pcd.colors = o3d.utility.Vector3dVector(final_colors)

            # PLY 파일 저장
            save_path_ply = VISUALIZATION_DIR / (base_output_name + ".ply")
            o3d.io.write_point_cloud(str(save_path_ply), pcd)

        except Exception as e:
            print(f"Warning: failed to save 3D visualization {base_output_name}.ply: {e}")

        # --- 4.7. 결과 반환 ---
        print(f"Generated QA sample: {dataset_name_suffix} ({unique_hash})")

    return label, prompt
