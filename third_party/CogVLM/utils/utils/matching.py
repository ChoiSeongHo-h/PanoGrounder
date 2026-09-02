import os
import re
import csv
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import math
import ast, random

def to_pinhole_view(img: Image.Image, view_deg, fov_deg, option):
    img_np = np.array(img)
    H, W = img_np.shape[:2]

    ### MODIFIED ###
    # 2:1 비율 (W/H=2) 또는 1:1 비율 (W/H=1)을 허용하도록 수정
    if not (abs(W / H - 2.0) < 1e-3 or abs(W / H - 1.0) < 1e-3):
        raise ValueError(f"입력 파노라마 비율이 2:1 또는 1:1이 아님. (W={W}, H={H})")

    fov_rad = math.radians(fov_deg)

    ### MODIFIED ###
    # 출력 너비(out_w) 계산 기준을 W(너비)에서 H(높이)로 변경
    # (W=2H였던 2:1 비율 기준) W * (fov / 2pi) == (2H) * (fov / 2pi) == H * (fov / pi)
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
        # u, v 계산은 W, H 변수에 따라 올바르게 매핑되므로 수정 필요 없음
        u = (lon / (2 * math.pi) + 0.5) * W
        v = (0.5 - lat / math.pi) * H
        return np.mod(u, W).astype(np.float32), np.clip(v, 0, H - 1).astype(np.float32)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    def render_at(yaw_deg: float) -> Image.Image:
        u, v = map_with_yaw(math.radians(yaw_deg))
        interp = cv2.INTER_LINEAR if option == "linear" else cv2.INTER_NEAREST
        bgr = cv2.remap(img_bgr, u, v, interpolation=interp, borderMode=cv2.BORDER_WRAP)
        if option == "linear":
            out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif option == "nearest":
            out = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("option은 'linear' 또는 'nearest'만 허용")
        return Image.fromarray(out)

    return render_at(view_deg)

import pandas as pd

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou


def compute_giou(boxA, boxB):
    boxes1 = boxA.unsqueeze(0)
    boxes2 = boxB.unsqueeze(0)

    x1_int = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1_int = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2_int = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2_int = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_w = (x2_int - x1_int).clamp(min=0)
    inter_h = (y2_int - y1_int).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))

    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)

    x1_c = torch.min(boxes1[:, 0], boxes2[:, 0])
    y1_c = torch.min(boxes1[:, 1], boxes2[:, 1])
    x2_c = torch.max(boxes1[:, 2], boxes2[:, 2])
    y2_c = torch.max(boxes1[:, 3], boxes2[:, 3])

    cw = (x2_c - x1_c).clamp(min=0)
    ch = (y2_c - y1_c).clamp(min=0)
    area_c = cw * ch

    giou = iou - (area_c - union_area) / area_c.clamp(min=1e-6)
    return giou.squeeze()


def find_all_gt_bboxes(gt_img, min_area=0):
    mask = np.array(gt_img)

    H, W = mask.shape
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids != np.iinfo(mask.dtype).max]

    gt_bboxes = {}

    for obj_id in object_ids:
        binary_mask = (mask == obj_id).astype(np.uint8)
        proj_x = np.sum(binary_mask, axis=0) > 0

        change = np.diff(proj_x.astype(np.int8), prepend=0, append=0)
        starts = np.where(change == 1)[0]
        ends   = np.where(change == -1)[0] - 1
        segments = list(zip(starts, ends))
        segments.sort(key=lambda s: s[1] - s[0], reverse=True)
        segments = segments[:2]

        bboxes = []
        for x_start, x_end in segments:
            region = binary_mask[:, x_start:x_end+1]
            ys, xs_local = np.nonzero(region)
            if ys.size == 0:
                continue
            x_min = x_start + xs_local.min()
            x_max = x_start + xs_local.max()
            y_min = ys.min()
            y_max = ys.max()
            area = (x_max - x_min) * (y_max - y_min)
            if area >= min_area:
                bboxes.append((x_min, y_min, x_max, y_max))

        if bboxes:
            gt_bboxes[int(obj_id)] = bboxes

    return gt_bboxes, (H, W)


def match_prediction_to_gt(pred_bbox, gt_bboxes):
    best_iou = 0
    best_object_id = None
    best_bbox = None
    if pred_bbox is not None:
        for obj_id, gt_bbox in gt_bboxes.items():
            for b in gt_bbox:
                iou = compute_iou(pred_bbox, b)
                if iou > best_iou:
                    best_iou = iou
                    best_object_id = obj_id
                    best_bbox = b
    return best_object_id, best_bbox


def parse_cogvlm_output(output_str, shape):
    H, W = shape
    scale_w, scale_h = W / 1000, H / 1000
    pattern = r"\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]"
    match = re.search(pattern, output_str)
    if not match:
        print(f"Failed to parse bounding box: {output_str}")
        return None

    x1, y1, x2, y2 = map(int, match.groups())
    return (round(x1 * scale_w), round(y1 * scale_h), round(x2 * scale_w), round(y2 * scale_h))


def save_results_to_csv(header, row, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    mode = 'a' if file_exists else 'w'

    with open(csv_path, mode=mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def calculate_regression_loss(pred_bbox, correct_gt_bbox):
    if pred_bbox == (-1, -1, -1, -1):
        return None, None

    l1_loss_fn = nn.SmoothL1Loss(reduction='sum')
    num_gt = correct_gt_bbox.shape[0]

    if num_gt == 0:
        # GT가 없는 경우 → 손실 계산할 수 없음
        return None, None

    elif num_gt == 1:
        # 단일 GT → 그대로 계산
        gt = correct_gt_bbox[0]
        l1 = l1_loss_fn(pred_bbox, gt)
        giou = 1 - compute_giou(pred_bbox, gt)
        return l1, giou

    elif num_gt == 2:
        # 2개인 경우 → 두 개에 대해 각각 l1, giou 계산
        gt1 = correct_gt_bbox[0]
        gt2 = correct_gt_bbox[1]

        l1_1 = l1_loss_fn(pred_bbox, gt1)
        l1_2 = l1_loss_fn(pred_bbox, gt2)

        giou_1 = 1 - compute_giou(pred_bbox, gt1)
        giou_2 = 1 - compute_giou(pred_bbox, gt2)

        # 하나만 선택: 더 작은 GIoU loss를 기준으로 선택
        if giou_1 < giou_2:
            return l1_1, giou_1
        else:
            return l1_2, giou_2

    else:
        raise ValueError("correct_gt_bbox must have length <= 2")


def process_cogvlm_output(qid, res, args):
    from utils.utils import joint_transform
    scene_id,cam_id,i,t,c = qid.rsplit('-', 4)
    image_idx, text_idx, correct_gt_id = int(i), int(t), int(c)

    if args.eval:
        dataset_name = args.dataset_name[0]
        dataset_idx = 0
        render_mode = args.render_mode[0]
    else:
        assert len(args.valid_data) == 1
        dataset_name = None
        dataset_idx = -1
        render_mode = None
        for idx, candidate in enumerate(args.dataset_name):
            if candidate in args.valid_data[0]:
                dataset_name = candidate
                dataset_idx = idx
                render_mode = args.render_mode[dataset_idx]
                break

    if dataset_name in ("ScanRefer", "Nr3D", "Sr3D"):
        image_dataset_name = "ScanNetV2"
    elif dataset_name in ("RIORefer"):
        image_dataset_name = "3RScan"
    else:
        image_dataset_name = dataset_name
    gt_path = f"{os.path.dirname(args.train_data[dataset_idx])}/{image_dataset_name}_{render_mode}/idx/{scene_id}-{render_mode}-{cam_id}.png"
    gt_img = Image.open(gt_path)

    if args.pinhole_mode == "none":
        gt_img = joint_transform(gt_img=gt_img, is_train=False, image_idx=image_idx, num_rot=args.num_rot_eval if args.eval else 1)

    elif args.pinhole_mode == "nways":
        gt_img = joint_transform(gt_img=gt_img, is_train=False, image_idx=image_idx, num_rot=args.num_rot_eval if args.eval else 1, pinhole_mode=args.pinhole_mode)

    elif args.pinhole_mode == "gt":
        gt_img = joint_transform(gt_img=gt_img, is_train=False, image_idx=image_idx, num_rot=args.num_rot_eval if args.eval else 1, pinhole_mode=args.pinhole_mode)
        biggest_bbox_i = None
        biggest_bbox_area = 0.0
        for i in range(4):
            gt_bboxes_all, (H, W) = find_all_gt_bboxes(gt_img[i], min_area=100)
            correct_gt_bboxes = gt_bboxes_all.get(correct_gt_id, [(0, 0, 0, 0)])
            # print(correct_gt_bboxes)
            # draw_bboxes_on_image(img[i], f"debug/{qid}_{i}.png", correct_bbox=correct_gt_bboxes)
            biggest_bbox = correct_gt_bboxes[0]
            area = (biggest_bbox[2]-biggest_bbox[0]) * (biggest_bbox[3]-biggest_bbox[1])
            if area > biggest_bbox_area:
                biggest_bbox_i = i
                biggest_bbox_area = area

        if biggest_bbox_i is None or biggest_bbox_area < 100:
            biggest_bbox_i = 0

        gt_img = gt_img[biggest_bbox_i]
    #####################################

    ################################################### seeground
    elif args.pinhole_mode == "seeground":
        numlabels_path = getattr(args, "numlabels_path", None)
        mask3d_root = getattr(args, "mask3d_root", None)
        if not numlabels_path or not mask3d_root:
            raise ValueError("seeground mode requires numlabels_path and mask3d_root")
        numlabels_df = pd.read_csv(numlabels_path)
        scene_id,cam_id,im,t,c = qid.rsplit('-', 4)
        image_idx, text_idx, correct_gt_id = int(im), int(t), int(c)
        mask = (
            (numlabels_df["scene_id"]  == scene_id) &
            (numlabels_df["text_idx"]  == text_idx)
        )
        try:
            matched_labels = ast.literal_eval(numlabels_df.loc[mask, "matched_labels"].item())
        except ValueError:
            matched_labels = []  # 못 찾았을 때
        mask3d_imgs = []
        for lbl in matched_labels:
            path = os.path.join(mask3d_root, f"{scene_id}-{args.render_mode}-{lbl}-{cam_id}.png")
            if not os.path.exists(path):
                path = os.path.join(mask3d_root, f"{scene_id}-{args.render_mode}-otherfurniture-{cam_id}.png")
            if os.path.exists(path):
                mask3d_imgs.append(path)
        final_binary_mask = None # 최종 0/1 마스크를 저장할 변수

        # 1. 처리할 마스크 이미지 리스트가 비어있지 않은지 확인
        if mask3d_imgs:

            valid_mask_paths = [p for p in mask3d_imgs if os.path.exists(p)]

            if valid_mask_paths:
                try:
                    # 2. 첫 번째 이미지를 로드하여 기준 마스크로 사용
                    first_img_pil = Image.open(valid_mask_paths[0]).convert('L')
                    first_img_np = np.array(first_img_pil)

                    # 255(배경)가 아니면 1, 255(배경)이면 0으로 변환
                    # (first_img_np != 255)는 True/False 배열이 됨
                    final_binary_mask = (first_img_np != 255)

                    # 3. 나머지 이미지들을 순회하며 마스크 병합 (Logical OR)
                    for path in valid_mask_paths[1:]:
                        img_pil = Image.open(path).convert('L')
                        img_np = np.array(img_pil)

                        # 현재 이미지에서 255가 아닌 픽셀(객체)을 찾음
                        current_object_mask = (img_np != 255)

                        # np.logical_or:
                        # 둘 중 하나라도 True(객체)이면 최종 결과가 True가 됨
                        # (1 | 0 = 1, 1 | 1 = 1, 0 | 0 = 0)
                        final_binary_mask = np.logical_or(final_binary_mask, current_object_mask)

                    # 4. 최종 결과를 True/False에서 0/1 (uint8)로 변환
                    final_binary_mask = final_binary_mask.astype(np.uint8)

                except FileNotFoundError as e:
                    print(f"Error: Mask file not found. {e}")
                    final_binary_mask = None # 오류 시 None
                except Exception as e:
                    print(f"Error during mask processing: {e}")
                    final_binary_mask = None

            else:
                print(f"Warning: All mask paths for qid {qid} were invalid.")

        else:
            print(f"Warning: No mask images found for qid {qid}.")

        horizontal_projection_mask = None # 최종 1D 마스크

        if final_binary_mask is not None:
            projection_bool = np.any(final_binary_mask, axis=0)
            horizontal_projection_mask = projection_bool.astype(np.uint8)

        else:
            print("Warning: Skipping horizontal projection, final_binary_mask is None.")

        sorted_segments = [] # 최종 결과를 담을 리스트

        # 1. 1D 마스크가 유효한지 확인
        if horizontal_projection_mask is not None and np.any(horizontal_projection_mask):

            # 2. 배열 양 끝에 0을 추가하여 경계면에서의 변화를 감지
            # 예: [1, 1, 0, 1] -> [0, 1, 1, 0, 1, 0]
            padded_mask = np.concatenate(([0], horizontal_projection_mask, [0]))

            # 3. np.diff로 값이 변하는 지점을 찾음
            # 0->1이 되면 1, 1->0이 되면 -1이 됨
            diff = np.diff(padded_mask)

            # 4. 1이 되는 지점(시작점)과 -1이 되는 지점(끝점)의 인덱스를 찾음

            # diff == 1 인 곳이 1이 시작되는 인덱스
            # (padded_mask 기준이므로, horizontal_projection_mask의 실제 인덱스와 동일)
            start_indices = np.where(diff == 1)[0]

            # diff == -1 인 곳이 0이 시작되는 인덱스 (즉, 1이 끝난 직후)
            # 따라서 1을 빼주면 1이 끝나는 마지막 인덱스가 됨
            end_indices = np.where(diff == -1)[0] - 1

            # 5. 각 구간의 길이와 중심 인덱스 계산
            segments = []
            W = horizontal_projection_mask.shape[0] # 1D 마스크의 전체 폭

            # 5-1. 파노라마 (wrap-around) 케이스인지 확인
            # 조건: (1) 0번 인덱스가 1이고 (2) 마지막 인덱스가 1이며
            # (3) 세그먼트가 2개 이상이어야 함 (즉, 중간에 0이 한번이라도 끼어있음)
            is_wrap_around = (horizontal_projection_mask[0] == 1 and
                        horizontal_projection_mask[-1] == 1 and
                        len(start_indices) > 1)

            if is_wrap_around:
                # 5-2. Wrap-around 세그먼트 처리

                # 왼쪽 세그먼트 정보 (start=0 부터)
                left_end = end_indices[0]
                left_length = left_end + 1

                # 오른쪽 세그먼트 정보 (end=W-1 까지)
                right_start = start_indices[-1]
                # (W-1) - right_start + 1 = W - right_start
                right_length = W - right_start

                # 두 세그먼트 길이 합치기
                total_length = left_length + right_length

                # 새 중심점 계산:
                # 0을 기준으로 나뉜 두 세그먼트 사이의 "갭(gap)"의 중심을 찾고,
                # 거기서 W/2 만큼 이동한 지점이 새 중심점이 됨 (모듈러 연산)
                gap_midpoint = (left_end + right_start) / 2.0
                new_center_float = (gap_midpoint + W / 2.0) % W
                new_center = int(round(new_center_float)) % W

                # 합쳐진 세그먼트를 추가
                segments.append((new_center, total_length))

                # 5-3. 중간에 있는 나머지 세그먼트들 추가
                # (첫 번째와 마지막을 제외한 중간 세그먼트들)
                for i in range(1, len(start_indices) - 1):
                    start = start_indices[i]
                    end = end_indices[i]
                    length = end - start + 1
                    center = (start + end) // 2
                    segments.append((center, length))

            else:
                # 5-4. 일반적인 (wrap-around 없는) 케이스
                # (세그먼트가 1개이거나, 양 끝이 1이 아닌 경우 등)
                for start, end in zip(start_indices, end_indices):
                    length = end - start + 1
                    center = (start + end) // 2
                    segments.append((center, length))

            sorted_segments = sorted(segments, key=lambda x: x[1], reverse=True)

        else:
            print("Warning: Skipping segment analysis, horizontal_projection_mask is None or all zeros.")

        # 6. 길이를 기준으로 내림차순 정렬
        TARGET_LENGTH = 4
        # final_segments를 sorted_segments의 복사본으로 초기화
        final_segments = list(sorted_segments)

        # 1. 4개보다 많으면, 상위 4개만 남기고 자르기
        if len(final_segments) > TARGET_LENGTH:
            final_segments = final_segments[:TARGET_LENGTH]

        # 2. 4개보다 적으면, 4개가 될 때까지 채우기
        elif len(final_segments) < TARGET_LENGTH:

            W = 0 # 1D 마스크의 폭(길이)
            if horizontal_projection_mask is not None:
                # .shape[0]으로 1D 배열의 길이를 가져옴
                W = horizontal_projection_mask.shape[0]
            else:
                W = 2000

            # 추가해야 할 세그먼트 개수 계산
            num_to_add = TARGET_LENGTH - len(final_segments)

            if W > 0:
                # W가 유효할 경우, 랜덤 인덱스 생성# 1. qid 문자열을 시드로 하는 '로컬' 랜덤 생성기(RNG) 생성
                # (파일 상단에 'import random' 필요)
                qid_seed = qid
                local_rng = random.Random(qid_seed)

                # 2. 로컬 RNG를 사용하여 랜덤 인덱스 생성
                for _ in range(num_to_add):
                    # 0부터 (W-1) 사이의 랜덤한 중심점 선택
                    # 'random.randint' 대신 'local_rng.randint' 사용
                    random_center = local_rng.randint(0, W - 1)
                    final_segments.append((random_center, 1))
            else:
                # horizontal_projection_mask가 비어있거나(W=0) None인 경우
                print(f"Warning: Cannot generate random segments (W=0). Padding with (0, 1).")
                for _ in range(num_to_add):
                    final_segments.append((0, 1)) # 기본값 (0, 1)로 채우기

        current_view = final_segments[image_idx][0]
        if W > 0:
            view_deg = (current_view - W/2) / W * 360
        else:
            view_deg = 0.0 # 마스크가 없었으므로 기본값 0
        # gt_img.save(f"debug/{qid}_gt_matching.png")

        gt_img = to_pinhole_view(gt_img, view_deg, fov_deg=120, option="nearest")
        # gt_img.save(f"debug/{qid}_gt_pinhole_matching.png")

    ###################################################

    gt_bboxes_all, shape = find_all_gt_bboxes(gt_img)
    pred_bbox = parse_cogvlm_output(res, shape)
    matched_gt_id, matched_gt_bbox = match_prediction_to_gt(pred_bbox, gt_bboxes_all)
    correct_gt_bbox = gt_bboxes_all.get(correct_gt_id, [])  # Occlusion
    success = (matched_gt_id == correct_gt_id) and len(correct_gt_bbox) > 0

    header = [
        "dataset_name",
        "scene_id",
        "cam_id",
        "image_idx",
        "text_idx",
        "success",
        "correct_gt_bbox",
        "matched_gt_bbox",
        "pred_bbox",
        "correct_gt_id",
        "matched_gt_id"
    ]

    result_row = [
        dataset_name,
        scene_id,
        cam_id,
        image_idx,
        text_idx,
        success,
        correct_gt_bbox if correct_gt_bbox else (-1, -1, -1, -1),
        matched_gt_bbox if matched_gt_bbox else (-1, -1, -1, -1),
        pred_bbox if pred_bbox else (-1, -1, -1, -1),
        correct_gt_id,
        matched_gt_id if matched_gt_id is not None else -1
    ]

    if args.eval:
        filename = args.experiment_name
        output_path = f"{args.output_root}/{dataset_name}/rank{args.rank}_{filename}.csv"
        save_results_to_csv(header, result_row, output_path)

    if pred_bbox is not None and len(correct_gt_bbox) > 0:
        l1_loss, giou_loss = calculate_regression_loss(torch.tensor(pred_bbox, dtype=torch.float64), torch.tensor(correct_gt_bbox, dtype=torch.float64))
        return success, l1_loss.item(), giou_loss.item()
    else:
        return success, 0.0, 0.0
