# Modified by the PanoGrounder Authors from CogVLM upstream.
import os
import ast
import csv
import torch
import numpy as np
import logging
import random
import jsonlines
import pandas as pd
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from sat.helpers import print_rank0
from utils.utils.generate_align_data import generate_single_qa_with_vis
from utils.utils.generate_align_data_aug import generate_single_qa_with_vis_aug, QAConfig


def find_all_files(path, suffix=".jpg"):
    target_files = []
    for cur_dir, _, files in os.walk(path, followlinks=True):
        for f in files:
            if f.endswith(suffix):
                target_files.append(os.path.join(cur_dir, f))
    print_rank0(f'find {len(target_files)} files...')
    return target_files

class ItemDataset(Dataset):
    def __init__(self, image_processor, text_processor, args, data_dirs, cross_image_processor=None, **kwargs):
        super().__init__()
        self.data = self.load_data(data_dirs)
        self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor

    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        if self.cross_image_processor:
            img_dict.update({'cross': self.cross_image_processor(img)})
        return img_dict

    def process_text(self, answer, prompt):
        return self.text_processor(answer, prompt)

    def load_data(self, data_dir):
        all_files = find_all_files(data_dir, suffix=".jpg")
        print_rank0(f"find {len(all_files)} samples in all...")
        return all_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        # img
        try:
            img = Image.open(data).convert('RGB')
        except Exception as e:
            print_rank0(e, level=logging.WARNING)
            return {}
        img_dict = self.process_img(img)
        # text
        label = data.split('/')[-1].split('.')[0]
        uni_key = label
        text_dict = self.process_text(label, "CAPTCHA:")
        if text_dict is None:
            print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {data}", level=logging.WARNING)
            return {}
        # other attr
        ret = {**img_dict, **text_dict, "question_id": uni_key}
        return ret

class VisualGroundingDataset(Dataset):
    def __init__(self, image_processor, text_processor, args, dataset_path, **kwargs):
        super().__init__()
        self.scene_limit = args.scene_limit
        self.cam_limit = args.cam_limit
        self.is_train = "train" in dataset_path
        self.eval = args.eval  # train/val - False, test - True
        if self.eval:
            self.num_rot_eval = args.num_rot_eval
        self.ratio_align_data = args.ratio_align_data if hasattr(args, 'ratio_align_data') else 0.0
        self.pretrain_geometry = args.pretrain_geometry if hasattr(args, 'pretrain_geometry') else False

        dataset_idx = -1
        if self.eval:
            dataset_idx = 0
        elif self.is_train:
            dataset_idx = args.train_data.index(dataset_path)
        else:
            dataset_idx = args.valid_data.index(dataset_path)
        self.data = self.load_data(dataset_path)
        self.render_mode = args.render_mode[dataset_idx]
        self.dataset_name = args.dataset_name[dataset_idx]
        image_dataset_name = "ScanNetV2" if self.dataset_name in ("ScanRefer", "Nr3D", "Sr3D") else self.dataset_name
        self.image_root = f"{os.path.dirname(dataset_path)}/{image_dataset_name}_{self.render_mode}"

        self.image_processor = image_processor
        self.text_processor = text_processor

        self.use_multi_view_feature = args.use_multi_view_feature if hasattr(args, 'use_multi_view_feature') else False
        self.use_cnn = args.use_cnn if hasattr(args, 'use_cnn') else False
        self.multi_view_feature_dim = args.use_multi_view_feature and args.multi_view_feature_dim or -1

        self.use_semantic_head = args.use_semantic_head if hasattr(args, 'use_semantic_head') else False
        self.num_semantic_classes = args.num_semantic_classes if hasattr(args, 'num_semantic_classes') else 40
        self.use_range_head = args.use_range_head if hasattr(args, 'use_range_head') else False
        self.use_xyz_head = args.use_xyz_head if hasattr(args, 'use_xyz_head') else False
        self.use_xyz_encoder = args.use_xyz_encoder if hasattr(args, 'use_xyz_encoder') else False
        self.xyz_encoder_dim = args.xyz_encoder_dim if hasattr(args, 'xyz_encoder_dim') else 384
        self.use_xyz_aug = args.use_xyz_aug if hasattr(args, 'use_xyz_aug') else False
        self.xyz_aug_num = args.xyz_aug_num if hasattr(args, 'xyz_aug_num') else 8
        self.debug_print = args.debug_print if hasattr(args, 'debug_print') else False
        self.use_fourier_features = args.use_fourier_features if hasattr(args, 'use_fourier_features') else False
        self.fourier_feature_dim = args.fourier_feature_dim if hasattr(args, 'fourier_feature_dim') else 6
        self.use_rangevit = args.use_rangevit if hasattr(args, 'use_rangevit') else False
        self.use_resize_xyz = args.use_resize_xyz if hasattr(args, 'use_resize_xyz') else False
        self.use_range_dino = args.use_range_dino if hasattr(args, 'use_range_dino') else False
        self.use_relative_range_head = args.use_relative_range_head if hasattr(args, 'use_relative_range_head') else False
        self.use_dinov3 = args.use_dinov3 if hasattr(args, 'use_dinov3') else False
        self.use_relative_xyz_head = args.use_relative_xyz_head if hasattr(args, 'use_relative_xyz_head') else False
        self.augment_align_data = args.augment_align_data if hasattr(args, 'augment_align_data') else False
        self.viz_align_data = args.viz_align_data if hasattr(args, 'viz_align_data') else False

        self.pinhole_mode = args.pinhole_mode if hasattr(args, 'pinhole_mode') else "none"
        self.mask3d_root = getattr(args, "mask3d_root", None)
        if self.pinhole_mode != "none":
            numlabels_path = getattr(args, "numlabels_path", None)
            if self.pinhole_mode == "seeground" and not numlabels_path:
                raise ValueError("seeground mode requires numlabels_path")
            if numlabels_path:
                self.numlabels_df = pd.read_csv(numlabels_path)

    def load_data(self, dataset_path):
        valid_data = []

        # pretrain_geometry일 때, train, val 모두 추가됨
        if self.pretrain_geometry:
            num_dummy_to_add = int(10000)

            print_rank0(f"self.is_train: {self.is_train}")
            print_rank0("Align data ratio: 1.0")

            if num_dummy_to_add > 0:
                # 더미 데이터의 기반이 될 템플릿 정의
                dummy_base_row = {
                    'dataset_name': 'DummyAlign',
                    'scene_id': 'scene0000_00',
                    'cam_id': 'cam_0',
                    'text_idx': '0',
                    'correct_gt_id': '0',
                    'description': 'This is a dummy alignment data sample.'
                    # 'qid'는 루프 내에서 고유하게 생성됩니다.
                }

                new_dummy_list = []

                # 학습 모드: num_dummy_to_add 만큼 단순 추가
                for i in range(num_dummy_to_add):
                    if self.is_train:
                        dummy_qid = f'{dummy_base_row["scene_id"]}-{dummy_base_row["cam_id"]}-{dummy_base_row["text_idx"]}-{dummy_base_row["correct_gt_id"]}'
                        new_dummy_list.append({
                            **dummy_base_row,
                            "qid": dummy_qid
                        })
                    else:
                        num_rot = self.num_rot_eval if self.eval else 1
                        for image_idx in range(num_rot):  # 90 degrees rotation each
                            dummy_qid = f'{dummy_base_row["scene_id"]}-{dummy_base_row["cam_id"]}-{image_idx}-{dummy_base_row["text_idx"]}-{dummy_base_row["correct_gt_id"]}'
                            new_dummy_list.append({
                                **dummy_base_row,
                                "image_idx": image_idx,
                                "qid": dummy_qid
                            })

                valid_data.extend(new_dummy_list)
                print_rank0(f"Added {len(new_dummy_list)} alignment samples; total: {len(valid_data)}")
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        if self.scene_limit != "-1" and row["scene_id"] > self.scene_limit:
                            continue
                        if self.cam_limit != "-1" and row["cam_id"] > self.cam_limit:
                            continue
                        if self.is_train:
                            valid_data.append({
                                **row,
                                "qid": f'{row["scene_id"]}-{row["cam_id"]}-{row["text_idx"]}-{row["correct_gt_id"]}'
                            })
                        else:
                            num_rot = self.num_rot_eval if self.eval else 1
                            for image_idx in range(num_rot):  # 90 degrees rotation each
                                valid_data.append({
                                    **row,
                                    "image_idx": image_idx,
                                    "qid": f'{row["scene_id"]}-{row["cam_id"]}-{image_idx}-{row["text_idx"]}-{row["correct_gt_id"]}'
                                })
                    except Exception as e:
                        print_rank0(f"Error parsing row: {row}\n{e}", level=logging.WARNING)
            print_rank0(f"Loaded {len(valid_data)} valid samples from {dataset_path}")

            # 온라인 샘플 추가 중 순수 train set에서만
            if self.ratio_align_data > 0.0 and self.is_train:
                num_original_samples = len(valid_data)
                num_dummy_to_add = int(num_original_samples * self.ratio_align_data)

                print_rank0(f"Alignment ratio: {self.ratio_align_data}; adding {num_dummy_to_add} samples to {num_original_samples} original samples")

                if num_dummy_to_add > 0:
                    # 더미 데이터의 기반이 될 템플릿 정의
                    dummy_base_row = {
                        'dataset_name': 'DummyAlign',
                        'scene_id': 'scene0000_00',
                        'cam_id': 'cam_0',
                        'text_idx': '0',
                        'correct_gt_id': '0',
                        'description': 'This is a dummy alignment data sample.'
                        # 'qid'는 루프 내에서 고유하게 생성됩니다.
                    }

                    new_dummy_list = []

                    # 학습 모드: num_dummy_to_add 만큼 단순 추가
                    for i in range(num_dummy_to_add):
                        dummy_qid = f'{dummy_base_row["scene_id"]}-{dummy_base_row["cam_id"]}-{dummy_base_row["text_idx"]}-{dummy_base_row["correct_gt_id"]}'
                        new_dummy_list.append({
                            **dummy_base_row,
                            "qid": dummy_qid
                        })

                    valid_data.extend(new_dummy_list)
                    print_rank0(f"Added {len(new_dummy_list)} alignment samples; total: {len(valid_data)}")

        return valid_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        from utils.utils import find_all_gt_bboxes, joint_transform, to_pinhole_view
        item = self.data[index]
        is_dummy_sample = item.get('dataset_name', '') == 'DummyAlign'
        if is_dummy_sample:
            # os.path.join(self.image_root, 'render') 상의 랜덤 png 파일을 선택합니다.
            random_image_dir = os.path.join(self.image_root, 'render')
            random_image_filename = random.choice(os.listdir(random_image_dir))
            # 선택된 파일명은 f"{item['scene_id']}-{self.render_mode}-{item['cam_id']}.png" 형식임
            # 따라서 item의 scene_id와 cam_id를 해당 파일명에서 추출하여 덮어씁니다.
            parts = random_image_filename.split('-')
            item['scene_id'] = parts[0]
            item['cam_id'] = parts[2].replace('.png', '')

        image_filename = f"{item['scene_id']}-{self.render_mode}-{item['cam_id']}.png"
        image_path = os.path.join(self.image_root, 'render', image_filename)
        gt_path = os.path.join(self.image_root, "idx", image_filename)
        if self.use_multi_view_feature:
            feature_path = os.path.join(self.image_root, "features_35", image_filename.replace('.png', '.npy'))
            if self.use_cnn:
                feature_path = os.path.join(self.image_root, "features_140", image_filename.replace('.png', '.npy'))
            if self.use_dinov3:
                feature_path = os.path.join(self.image_root, "features_35_dinov3", image_filename.replace('.png', '.npy'))
        if self.use_semantic_head:
            semantic_path = os.path.join(self.image_root, "semantic_35", image_filename.replace('.png', '.npy'))
        if self.use_range_head or self.use_xyz_head:
            range_path = os.path.join(self.image_root, "range_35", image_filename.replace('.png', '.npy'))
        if (self.use_xyz_encoder or self.use_relative_range_head):
            input_range_path = os.path.join(self.image_root, "range", image_filename)
        valid_box = True

        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print_rank0(f"Error loading image {image_path}: {e}", level=logging.WARNING)
            return {}
        try:
            gt_img = Image.open(gt_path)
        except Exception as e:
            print_rank0(f"Error loading image {gt_path}: {e}", level=logging.WARNING)
            return {}
        multi_view_feature_map = None
        if self.use_multi_view_feature and os.path.exists(feature_path):
            try:
                # npy feature_path 파일을 불러옵니다.
                multi_view_feature_map = torch.from_numpy(np.load(feature_path)).float()
            except Exception as e:
                print_rank0(f"Failed to load multi-view feature: {feature_path}. Error: {e}", level=logging.WARNING)
                return {}

        semantic_map = None
        if self.use_semantic_head and os.path.exists(semantic_path):
            try:
                # npy semantic_path 파일을 불러옵니다.
                semantic_tensor = np.load(semantic_path)
                grid_size = semantic_tensor.shape[0]
                semantic_tensor = semantic_tensor.reshape(grid_size, grid_size, self.num_semantic_classes+1)
                semantic_map = torch.from_numpy(semantic_tensor).float()
            except Exception as e:
                print_rank0(f"Failed to load semantic map: {semantic_path}. Error: {e}", level=logging.WARNING)
                return {}

        range_map = None
        gt_xyz = None
        if (self.use_range_head and os.path.exists(range_path)) or (self.use_xyz_head and os.path.exists(range_path)):
            try:
                # npy range_path 파일을 불러옵니다.
                range_tensor = np.load(range_path)
                grid_size = range_tensor.shape[0]
                range_tensor = range_tensor.reshape(grid_size, grid_size, 1)
                range_map = torch.from_numpy(range_tensor).float()
            except Exception as e:
                print_rank0(f"Failed to load range map: {range_path}. Error: {e}", level=logging.WARNING)
                return {}

        input_range = None
        input_geometry = None
        if (self.use_xyz_encoder or self.use_relative_range_head) and os.path.exists(input_range_path):
            try:
                input_range = Image.open(input_range_path)
            except Exception as e:
                print_rank0(f"Error loading input range {input_range_path}: {e}", level=logging.WARNING)
                return {}

        try:
            if self.is_train:
                if self.pinhole_mode == "none":
                    img, gt_img, feature, semantic, gt_range, gt_xyz, input_geometry = joint_transform(img=img, gt_img=gt_img, feature=multi_view_feature_map, semantic=semantic_map, gt_range=range_map, input_range=input_range, is_train=self.is_train, use_xyz_aug=self.use_xyz_aug, xyz_aug_num=self.xyz_aug_num, debug_print=self.debug_print, use_fourier_features=self.use_fourier_features, fourier_feature_dim=self.fourier_feature_dim, use_rangevit=self.use_rangevit, use_resize_xyz=self.use_resize_xyz, use_range_dino=self.use_range_dino, use_relative_range_head=self.use_relative_range_head, use_relative_xyz_head=self.use_relative_xyz_head)
                    img_dict = {'vision': self.image_processor(img)}

                    if not is_dummy_sample:
                        gt_bboxes_all, (H, W) = find_all_gt_bboxes(gt_img, min_area=100 / (2000 * 1000) * (490 * 490))
                        correct_gt_bboxes = gt_bboxes_all.get(int(item['qid'].split('-')[-1]), [])

                        if len(correct_gt_bboxes) == 0:
                            bbox = (0, 0, 0, 0)
                            valid_box = False

                        elif len(correct_gt_bboxes) == 1:
                            bbox = correct_gt_bboxes[0]

                        elif len(correct_gt_bboxes) == 2:
                            # orig_img, orig_gt_img = img, gt_img
                            # for i in range(3):
                            #     img, gt_img = joint_transform(orig_img, orig_gt_img, self.is_train)
                            #     gt_bboxes_all, _ = find_all_gt_bboxes(gt_img, min_area=100)
                            #     correct_gt_bboxes = gt_bboxes_all.get(int(item['qid'].split('-')[-1]), [])
                            #     if len(correct_gt_bboxes) == 1:
                            #         bbox = correct_gt_bboxes[0]
                            #         break
                            bbox = correct_gt_bboxes[0]

                        if self.dataset_name == "Nr3D" or self.dataset_name == "Sr3D":
                            distractor_gt_ids = ast.literal_eval(item["distractor_gt_id"])
                            distractor_bboxes = []
                            for d_id in distractor_gt_ids:
                                boxes = gt_bboxes_all.get(d_id, [])
                                distractor_bboxes.extend(boxes)
                            all_bboxes = distractor_bboxes + correct_gt_bboxes
                            random.shuffle(all_bboxes)
                            norm_bboxes = [(
                                    int(x1 * 1000 / W),
                                    int(y1 * 1000 / H),
                                    int(x2 * 1000 / W),
                                    int(y2 * 1000 / H)
                                ) for (x1, y1, x2, y2) in all_bboxes]
                            boxes_str = ', '.join(f'[[{x1:03d},{y1:03d},{x2:03d},{y2:03d}]]' for x1, y1, x2, y2 in norm_bboxes)
                            item["description"] = item["description"].replace("PLACEHOLDER", boxes_str)

                        scale_w, scale_h = W / 1000, H / 1000
                        bbox_normalized = (int(bbox[0] / scale_w), int(bbox[1] / scale_h), int(bbox[2] / scale_w), int(bbox[3] / scale_h))

                        prompt = item["description"]
                        label = f'[[{bbox_normalized[0]:03d},{bbox_normalized[1]:03d},{bbox_normalized[2]:03d},{bbox_normalized[3]:03d}]]'
                    else:
                        if self.augment_align_data:
                            config = QAConfig()
                            label, prompt = generate_single_qa_with_vis_aug(
                                gt_xyz,                 # [H, W, 3], float32, m
                                config=config,          # 설정 객체
                                viz_align_data=self.viz_align_data
                            )
                        else:
                            label, prompt = generate_single_qa_with_vis(gt_range, gt_xyz, viz_align_data=self.viz_align_data)

                    text_dict = self.text_processor(label, prompt)
                else:
                    # While training, always use gt pinhole mode
                    img, gt_img, feature, semantic, gt_range, gt_xyz, input_geometry = joint_transform(img=img, gt_img=gt_img, is_train=self.is_train, pinhole_mode="gt")
                    correct_gt_id = int(item['qid'].split('-')[-1])
                    # print(item['qid'])
                    biggest_bbox_i = None
                    biggest_bbox_area = 0.0
                    biggest_gt_bboxes_all = None
                    for i in range(4):
                        gt_bboxes_all, (H, W) = find_all_gt_bboxes(gt_img[i], min_area=100)
                        correct_gt_bboxes = gt_bboxes_all.get(correct_gt_id, [(0, 0, 0, 0)])
                        # print(correct_gt_bboxes)
                        # draw_bboxes_on_image(img[i], f"debug/{item['qid']}_{i}.png", correct_bbox=correct_gt_bboxes)
                        biggest_bbox = correct_gt_bboxes[0]
                        area = (biggest_bbox[2]-biggest_bbox[0]) * (biggest_bbox[3]-biggest_bbox[1])
                        if area > biggest_bbox_area:
                            biggest_bbox_i = i
                            biggest_bbox_area = area
                            biggest_gt_bboxes_all = gt_bboxes_all
                    # print(biggest_bbox_i, biggest_bbox_area)

                    if biggest_bbox_i is None or biggest_bbox_area < 100:
                        return {}

                    img, gt_img = img[biggest_bbox_i], gt_img[biggest_bbox_i]
                    gt_bboxes_all = biggest_gt_bboxes_all

                    ## ----------------------------------

                    correct_gt_id = int(item['qid'].split('-')[-1])
                    correct_gt_bboxes = gt_bboxes_all.get(correct_gt_id, [])

                    if len(correct_gt_bboxes) == 0:
                        bbox = (0, 0, 0, 0)
                        valid_box = False

                    elif len(correct_gt_bboxes) == 1:
                        bbox = correct_gt_bboxes[0]

                    elif len(correct_gt_bboxes) == 2:
                        # orig_img, orig_gt_img = img, gt_img
                        # for i in range(3):
                        #     img, gt_img = joint_transform(orig_img, orig_gt_img, self.is_train)
                        #     gt_bboxes_all, _ = find_all_gt_bboxes(gt_img, min_area=100)
                        #     correct_gt_bboxes = gt_bboxes_all.get(correct_gt_id, [])
                        #     if len(correct_gt_bboxes) == 1:
                        #         bbox = correct_gt_bboxes[0]
                        #         break
                        bbox = correct_gt_bboxes[0]

                    if self.dataset_name == "Nr3D" or self.dataset_name == "Sr3D":
                        distractor_gt_ids = ast.literal_eval(item["distractor_gt_id"])
                        distractor_bboxes = []
                        for d_id in distractor_gt_ids:
                            boxes = gt_bboxes_all.get(d_id, [])
                            distractor_bboxes.extend(boxes)
                        all_bboxes = distractor_bboxes + correct_gt_bboxes
                        random.shuffle(all_bboxes)
                        norm_bboxes = [(
                                int(x1 * 1000 / W),
                                int(y1 * 1000 / H),
                                int(x2 * 1000 / W),
                                int(y2 * 1000 / H)
                            ) for (x1, y1, x2, y2) in all_bboxes]
                        boxes_str = ', '.join(f'[[{x1:03d},{y1:03d},{x2:03d},{y2:03d}]]' for x1, y1, x2, y2 in norm_bboxes)
                        item["description"] = item["description"].replace("PLACEHOLDER", boxes_str)

                    scale_w, scale_h = W / 1000, H / 1000
                    bbox_normalized = (int(bbox[0] / scale_w), int(bbox[1] / scale_h), int(bbox[2] / scale_w), int(bbox[3] / scale_h))

                    img_dict = {'vision': self.image_processor(img)}
                    prompt = item["description"]
                    label = f'[[{bbox_normalized[0]:03d},{bbox_normalized[1]:03d},{bbox_normalized[2]:03d},{bbox_normalized[3]:03d}]]'
                    text_dict = self.text_processor(label, prompt)

            else:
                if self.pinhole_mode == "none":
                    img, gt_img, feature, semantic, gt_range, gt_xyz, input_geometry = joint_transform(img=img, gt_img=gt_img, feature=multi_view_feature_map, semantic=semantic_map, gt_range=range_map, input_range=input_range, is_train=self.is_train, image_idx=item['image_idx'], num_rot=self.num_rot_eval if self.eval else 1, use_xyz_aug=self.use_xyz_aug, xyz_aug_num=self.xyz_aug_num, debug_print=self.debug_print, use_fourier_features=self.use_fourier_features, fourier_feature_dim=self.fourier_feature_dim, use_rangevit=self.use_rangevit, use_resize_xyz=self.use_resize_xyz, use_range_dino=self.use_range_dino, use_relative_range_head=self.use_relative_range_head, use_relative_xyz_head=self.use_relative_xyz_head)

                    if not is_dummy_sample:
                        gt_bboxes_all, (H, W) = find_all_gt_bboxes(gt_img, min_area=100)
                        correct_gt_bboxes = gt_bboxes_all.get(int(item['qid'].split('-')[-1]), [])

                        if self.dataset_name == "Nr3D" or self.dataset_name == "Sr3D":
                            distractor_gt_ids = ast.literal_eval(item["distractor_gt_id"])
                            distractor_gt_ids.append(int(item["correct_gt_id"]))
                            distractor_bboxes = []
                            for d_id in distractor_gt_ids:
                                boxes = gt_bboxes_all.get(d_id, [])
                                distractor_bboxes.extend(boxes)
                            all_bboxes = distractor_bboxes
                            random.shuffle(all_bboxes)
                            norm_bboxes = [(
                                    int(x1 * 1000 / W),
                                    int(y1 * 1000 / H),
                                    int(x2 * 1000 / W),
                                    int(y2 * 1000 / H)
                                ) for (x1, y1, x2, y2) in all_bboxes]
                            boxes_str = ', '.join(f'[[{x1:03d},{y1:03d},{x2:03d},{y2:03d}]]' for x1, y1, x2, y2 in norm_bboxes)
                            item["description"] = item["description"].replace("PLACEHOLDER", boxes_str)

                        img_dict = {'vision': self.image_processor(img)}
                        prompt = item["description"]
                        label = '[[000,000,000,000]]'  # Placeholder
                    else:
                        if self.augment_align_data:
                            config = QAConfig()
                            label, prompt = generate_single_qa_with_vis_aug(
                                gt_xyz,                 # [H, W, 3], float32, m
                                config=config,          # 설정 객체
                                viz_align_data=self.viz_align_data
                            )
                        else:
                            label, prompt = generate_single_qa_with_vis(gt_range, gt_xyz, viz_align_data=self.viz_align_data)

                        img_dict = {'vision': self.image_processor(img)}

                    text_dict = self.text_processor(label, prompt)
                else:
                    # img.save(f"debug/{item['qid']}.png")
                    # gt_img.save(f"debug/{item['qid']}_gt.png")

                    if self.pinhole_mode == "nways":
                        img, gt_img, feature, semantic, gt_range, gt_xyz, input_geometry = joint_transform(img=img, gt_img=gt_img, is_train=self.is_train, image_idx=item['image_idx'], num_rot=self.num_rot_eval if self.eval else 1, pinhole_mode=self.pinhole_mode)

                    ################################################### pinhole4
                    elif self.pinhole_mode == "gt":
                        img, gt_img, feature, semantic, gt_range, gt_xyz, input_geometry = joint_transform(img=img, gt_img=gt_img, is_train=self.is_train, image_idx=item['image_idx'], num_rot=self.num_rot_eval if self.eval else 1, pinhole_mode=self.pinhole_mode)
                        correct_gt_id = int(item['qid'].split('-')[-1])
                        # print(item['qid'])
                        biggest_bbox_i = None
                        biggest_bbox_area = 0.0
                        biggest_gt_bboxes_all = None
                        for i in range(4):
                            gt_bboxes_all, (H, W) = find_all_gt_bboxes(gt_img[i], min_area=100)
                            correct_gt_bboxes = gt_bboxes_all.get(correct_gt_id, [(0, 0, 0, 0)])
                            # print(correct_gt_bboxes)
                            # draw_bboxes_on_image(img[i], f"debug/{item['qid']}_{i}.png", correct_bbox=correct_gt_bboxes)
                            biggest_bbox = correct_gt_bboxes[0]
                            area = (biggest_bbox[2]-biggest_bbox[0]) * (biggest_bbox[3]-biggest_bbox[1])
                            if i == 0:
                                biggest_gt_bboxes_all = gt_bboxes_all
                            if area > biggest_bbox_area:
                                biggest_bbox_i = i
                                biggest_bbox_area = area
                                biggest_gt_bboxes_all = gt_bboxes_all
                        # print(biggest_bbox_i, biggest_bbox_area)

                        if biggest_bbox_i is None or biggest_bbox_area < 100:
                            biggest_bbox_i = 0

                        img, gt_img = img[biggest_bbox_i], gt_img[biggest_bbox_i]
                        gt_bboxes_all = biggest_gt_bboxes_all
                    ###################################################

                    ################################################### seeground
                    elif self.pinhole_mode == "seeground":
                        mask3d_root = self.mask3d_root
                        if not mask3d_root:
                            raise ValueError("seeground mode requires mask3d_root")
                        scene_id,cam_id,im,t,c = item['qid'].rsplit('-', 4)
                        image_idx, text_idx, correct_gt_id = int(im), int(t), int(c)
                        mask = (
                            (self.numlabels_df["scene_id"]  == scene_id) &
                            (self.numlabels_df["text_idx"]  == text_idx)
                        )
                        try:
                            matched_labels = ast.literal_eval(self.numlabels_df.loc[mask, "matched_labels"].item())
                        except ValueError:
                            matched_labels = []  # 못 찾았을 때
                        mask3d_imgs = []
                        for lbl in matched_labels:
                            path = os.path.join(mask3d_root, f"{scene_id}-{self.render_mode}-{lbl}-{cam_id}.png")
                            if not os.path.exists(path):
                                path = os.path.join(mask3d_root, f"{scene_id}-{self.render_mode}-otherfurniture-{cam_id}.png")
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
                                print(f"Warning: All mask paths for qid {item['qid']} were invalid.")

                        else:
                            print(f"Warning: No mask images found for qid {item['qid']}.")

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
                                qid_seed = item['qid']
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

                    # img.save(f"debug/{item['qid']}.png")
                    # gt_img.save(f"debug/{item['qid']}_gt.png")

                        img = to_pinhole_view(img, view_deg, fov_deg=120, option="linear")
                        gt_img = to_pinhole_view(gt_img, view_deg, fov_deg=120, option="nearest")

                        # img.save(f"debug/{item['qid']}_pinhole.png")
                        # gt_img.save(f"debug/{item['qid']}_gt_pinhole.png")
                    ###################################################
                    # img.save(f"debug/{item['qid']}_after.png")
                    # gt_img.save(f"debug/{item['qid']}_gt_after.png")


                    gt_bboxes_all, (H, W) = find_all_gt_bboxes(gt_img, min_area=100)
                    correct_gt_bboxes = gt_bboxes_all.get(int(item['qid'].split('-')[-1]), [])

                    if self.dataset_name == "Nr3D" or self.dataset_name == "Sr3D":
                        distractor_gt_ids = ast.literal_eval(item["distractor_gt_id"])
                        distractor_gt_ids.append(int(item["correct_gt_id"]))
                        distractor_bboxes = []
                        for d_id in distractor_gt_ids:
                            boxes = gt_bboxes_all.get(d_id, [])
                            distractor_bboxes.extend(boxes)
                        all_bboxes = distractor_bboxes
                        random.shuffle(all_bboxes)
                        norm_bboxes = [(
                                int(x1 * 1000 / W),
                                int(y1 * 1000 / H),
                                int(x2 * 1000 / W),
                                int(y2 * 1000 / H)
                            ) for (x1, y1, x2, y2) in all_bboxes]
                        boxes_str = ', '.join(f'[[{x1:03d},{y1:03d},{x2:03d},{y2:03d}]]' for x1, y1, x2, y2 in norm_bboxes)
                        item["description"] = item["description"].replace("PLACEHOLDER", boxes_str)

                    img_dict = {'vision': self.image_processor(img)}
                    prompt = item["description"]
                    label = '[[000,000,000,000]]'  # Placeholder
                    text_dict = self.text_processor(label, prompt)

        except Exception as e:
            print_rank0(f"Text processing failed for {item['qid']}: {e}", level=logging.WARNING)
            return {}

        if self.use_multi_view_feature and feature is None:
            print(f"Missing feature array: {feature_path}")
            return {}

        ret = {**img_dict, **text_dict, "question_id": item["qid"], "valid_box": torch.tensor([valid_box])}
        if self.use_multi_view_feature and feature is not None:
            ret["multi_view_feature"] = feature.view(-1, self.multi_view_feature_dim)
        if self.use_semantic_head and semantic is not None:
            ret["semantic_labels"] = semantic.view(-1, self.num_semantic_classes+1)
        if self.use_xyz_encoder and input_geometry is not None:
            ret["input_geometry"] = input_geometry
        if self.use_xyz_head and gt_xyz is not None:
            ret["xyz_labels"] = gt_xyz.view(-1, 3)
        if self.use_relative_xyz_head and gt_xyz is not None:
            ret["xyz_labels"] = gt_xyz
        if (self.use_range_head or self.use_relative_range_head) and gt_range is not None:
            ret["range_labels"] = gt_range.unsqueeze(0)
        return ret

class GeometryQADataset(Dataset):
    def __init__(self, image_processor, text_processor, args, dataset_path, **kwargs):
        super().__init__()
        self.scene_limit = args.scene_limit
        self.cam_limit = args.cam_limit
        self.is_train = "train" in dataset_path
        self.eval = args.eval  # train/val - False, test - True
        self.data = self.load_data(dataset_path)
        self.render_mode = args.render_mode
        self.dataset_name = args.dataset_name
        image_dataset_name = "ScanNetV2" if args.dataset_name in ("ScanRefer", "Nr3D", "Sr3D") else args.dataset_name
        self.image_root = f"{os.path.dirname(dataset_path)}/{image_dataset_name}_{self.render_mode}"

        self.image_processor = image_processor
        self.text_processor = text_processor

        self.use_multi_view_feature = args.use_multi_view_feature if hasattr(args, 'use_multi_view_feature') else False
        self.use_cnn = args.use_cnn if hasattr(args, 'use_cnn') else False
        self.multi_view_feature_dim = args.use_multi_view_feature and args.multi_view_feature_dim or -1

        self.use_semantic_head = args.use_semantic_head if hasattr(args, 'use_semantic_head') else False
        self.num_semantic_classes = args.num_semantic_classes if hasattr(args, 'num_semantic_classes') else 40
        self.use_range_head = args.use_range_head if hasattr(args, 'use_range_head') else False
        self.use_xyz_head = args.use_xyz_head if hasattr(args, 'use_xyz_head') else False
        self.use_xyz_encoder = args.use_xyz_encoder if hasattr(args, 'use_xyz_encoder') else False
        self.xyz_encoder_dim = args.xyz_encoder_dim if hasattr(args, 'xyz_encoder_dim') else 384
        self.use_xyz_aug = args.use_xyz_aug if hasattr(args, 'use_xyz_aug') else False
        self.xyz_aug_num = args.xyz_aug_num if hasattr(args, 'xyz_aug_num') else 8
        self.debug_print = args.debug_print if hasattr(args, 'debug_print') else False
        self.use_fourier_features = args.use_fourier_features if hasattr(args, 'use_fourier_features') else False
        self.fourier_feature_dim = args.fourier_feature_dim if hasattr(args, 'fourier_feature_dim') else 6
        self.use_rangevit = args.use_rangevit if hasattr(args, 'use_rangevit') else False
        self.use_resize_xyz = args.use_resize_xyz if hasattr(args, 'use_resize_xyz') else False
        self.use_range_dino = args.use_range_dino if hasattr(args, 'use_range_dino') else False
        self.use_relative_range_head = args.use_relative_range_head if hasattr(args, 'use_relative_range_head') else False
        self.use_dinov3 = args.use_dinov3 if hasattr(args, 'use_dinov3') else False
        self.use_relative_xyz_head = args.use_relative_xyz_head if hasattr(args, 'use_relative_xyz_head') else False

    def load_data(self, dataset_path):
        valid_data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if self.scene_limit != "-1":
                        try:
                            # 1. row["scene_id"]에서 숫자 부분('xxxx')을 추출합니다.
                            # 예: "scene0422_00" -> "scene0422" -> "0422"
                            scene_number_str = row["scene_id"].split('_')[0][5:]

                            # 2. 추출한 숫자와 scene_limit을 모두 정수(int)로 변환합니다.
                            scene_number = int(scene_number_str)
                            limit = int(self.scene_limit)

                            # 3. 두 정수를 비교하여 의도한 대로 동작하게 합니다.
                            if scene_number > limit:
                                continue

                        except (ValueError, IndexError):
                            # 만약 "scene0422_00" 형식이 아닌 데이터가 있을 경우 오류가 발생할 수 있습니다.
                            # 예외 처리를 통해 프로그램을 안정적으로 유지합니다.
                            print(f"Warning: 잘못된 형식의 scene_id ('{row['scene_id']}')를 건너뜁니다.")
                            continue
                    if self.cam_limit != "-1" and row["cam_id"] > self.cam_limit:
                        continue
                    if self.is_train:
                        valid_data.append({
                            **row,
                            "qid": f'{row["scene_id"]}-{row["cam_id"]}-{row["text_idx"]}-0'
                        })
                    else:
                        num_rot = 1
                        for image_idx in range(num_rot):  # 90 degrees rotation each
                            valid_data.append({
                                **row,
                                "image_idx": image_idx,
                                "qid": f'{row["scene_id"]}-{row["cam_id"]}-{image_idx}-{row["text_idx"]}-0'
                            })
                except Exception as e:
                    print_rank0(f"Error parsing row: {row}\n{e}", level=logging.WARNING)
        print_rank0(f"Loaded {len(valid_data)} valid samples from {dataset_path}")
        return valid_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        from utils.utils import find_all_gt_bboxes, joint_transform
        item = self.data[index]
        image_filename = f"{item['scene_id']}-{self.render_mode}-{item['cam_id']}.png"
        image_path = os.path.join(self.image_root, 'render', image_filename)
        gt_path = os.path.join(self.image_root, "idx", image_filename)
        if self.use_multi_view_feature:
            feature_path = os.path.join(self.image_root, "features_35", image_filename.replace('.png', '.npy'))
            if self.use_cnn:
                feature_path = os.path.join(self.image_root, "features_140", image_filename.replace('.png', '.npy'))
            if self.use_dinov3:
                feature_path = os.path.join(self.image_root, "features_35_dinov3", image_filename.replace('.png', '.npy'))
        if self.use_semantic_head:
            semantic_path = os.path.join(self.image_root, "semantic_35", image_filename.replace('.png', '.npy'))
        if self.use_range_head or self.use_xyz_head:
            range_path = os.path.join(self.image_root, "range_35", image_filename.replace('.png', '.npy'))
        if (self.use_xyz_encoder or self.use_relative_range_head):
            input_range_path = os.path.join(self.image_root, "range", image_filename)
        valid_box = True

        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print_rank0(f"Error loading image {image_path}: {e}", level=logging.WARNING)
            return {}
        try:
            gt_img = Image.open(gt_path)
        except Exception as e:
            print_rank0(f"Error loading image {gt_path}: {e}", level=logging.WARNING)
            return {}
        multi_view_feature_map = None
        if self.use_multi_view_feature and os.path.exists(feature_path):
            try:
                # npy feature_path 파일을 불러옵니다.
                multi_view_feature_map = torch.from_numpy(np.load(feature_path)).float()
            except Exception as e:
                print_rank0(f"Failed to load multi-view feature: {feature_path}. Error: {e}", level=logging.WARNING)
                return {}

        semantic_map = None
        if self.use_semantic_head and os.path.exists(semantic_path):
            try:
                # npy semantic_path 파일을 불러옵니다.
                semantic_tensor = np.load(semantic_path)
                grid_size = semantic_tensor.shape[0]
                semantic_tensor = semantic_tensor.reshape(grid_size, grid_size, self.num_semantic_classes+1)
                semantic_map = torch.from_numpy(semantic_tensor).float()
            except Exception as e:
                print_rank0(f"Failed to load semantic map: {semantic_path}. Error: {e}", level=logging.WARNING)
                return {}

        range_map = None
        gt_xyz = None
        if (self.use_range_head and os.path.exists(range_path)) or (self.use_xyz_head and os.path.exists(range_path)):
            try:
                # npy range_path 파일을 불러옵니다.
                range_tensor = np.load(range_path)
                grid_size = range_tensor.shape[0]
                range_tensor = range_tensor.reshape(grid_size, grid_size, 1)
                range_map = torch.from_numpy(range_tensor).float()
            except Exception as e:
                print_rank0(f"Failed to load range map: {range_path}. Error: {e}", level=logging.WARNING)
                return {}

        input_range = None
        input_geometry = None
        if (self.use_xyz_encoder or self.use_relative_range_head) and os.path.exists(input_range_path):
            try:
                input_range = Image.open(input_range_path)
            except Exception as e:
                print_rank0(f"Error loading input range {input_range_path}: {e}", level=logging.WARNING)
                return {}

        try:
            if self.is_train:
                img, gt_img, feature, semantic, gt_range, gt_xyz, input_geometry = joint_transform(img=img, gt_img=gt_img, feature=multi_view_feature_map, semantic=semantic_map, gt_range=range_map, input_range=input_range, is_train=self.is_train, use_xyz_aug=self.use_xyz_aug, xyz_aug_num=self.xyz_aug_num, debug_print=self.debug_print, use_fourier_features=self.use_fourier_features, fourier_feature_dim=self.fourier_feature_dim, use_rangevit=self.use_rangevit, use_resize_xyz=self.use_resize_xyz, use_range_dino=self.use_range_dino, use_relative_range_head=self.use_relative_range_head, use_relative_xyz_head=self.use_relative_xyz_head)

                gt_bboxes_all, (H, W) = find_all_gt_bboxes(gt_img, min_area=100 / (2000 * 1000) * (490 * 490))
                correct_gt_bboxes = gt_bboxes_all.get(int(item['qid'].split('-')[-1]), [])

                if len(correct_gt_bboxes) == 0:
                    bbox = (0, 0, 0, 0)
                    valid_box = False

                elif len(correct_gt_bboxes) == 1:
                    bbox = correct_gt_bboxes[0]

                elif len(correct_gt_bboxes) == 2:
                    # orig_img, orig_gt_img = img, gt_img
                    # for i in range(3):
                    #     img, gt_img = joint_transform(orig_img, orig_gt_img, self.is_train)
                    #     gt_bboxes_all, _ = find_all_gt_bboxes(gt_img, min_area=100)
                    #     correct_gt_bboxes = gt_bboxes_all.get(int(item['qid'].split('-')[-1]), [])
                    #     if len(correct_gt_bboxes) == 1:
                    #         bbox = correct_gt_bboxes[0]
                    #         break
                    bbox = correct_gt_bboxes[0]

                # if self.dataset_name == "Nr3D" or self.dataset_name == "Sr3D":
                #     distractor_gt_ids = ast.literal_eval(item["distractor_gt_id"])
                #     distractor_bboxes = []
                #     for d_id in distractor_gt_ids:
                #         boxes = gt_bboxes_all.get(d_id, [])
                #         distractor_bboxes.extend(boxes)
                #     all_bboxes = distractor_bboxes + correct_gt_bboxes
                #     random.shuffle(all_bboxes)
                #     norm_bboxes = [(
                #             int(x1 * 1000 / W),
                #             int(y1 * 1000 / H),
                #             int(x2 * 1000 / W),
                #             int(y2 * 1000 / H)
                #         ) for (x1, y1, x2, y2) in all_bboxes]
                #     boxes_str = ', '.join(f'[[{x1:03d},{y1:03d},{x2:03d},{y2:03d}]]' for x1, y1, x2, y2 in norm_bboxes)
                #     item["description"] = item["description"].replace("PLACEHOLDER", boxes_str)

                scale_w, scale_h = W / 1000, H / 1000
                bbox_normalized = (int(bbox[0] / scale_w), int(bbox[1] / scale_h), int(bbox[2] / scale_w), int(bbox[3] / scale_h))

                img_dict = {'vision': self.image_processor(img)}
                prompt = item["description"]
                label = f'[[{bbox_normalized[0]:03d},{bbox_normalized[1]:03d},{bbox_normalized[2]:03d},{bbox_normalized[3]:03d}]]'
                text_dict = self.text_processor(label, prompt)

            else:
                img, gt_img, feature, semantic, gt_range, gt_xyz, input_geometry = joint_transform(img=img, gt_img=gt_img, feature=multi_view_feature_map, semantic=semantic_map, gt_range=range_map, input_range=input_range, is_train=self.is_train, image_idx=item['image_idx'], use_xyz_aug=self.use_xyz_aug, xyz_aug_num=self.xyz_aug_num, debug_print=self.debug_print, use_fourier_features=self.use_fourier_features, fourier_feature_dim=self.fourier_feature_dim, use_rangevit=self.use_rangevit, use_resize_xyz=self.use_resize_xyz, use_range_dino=self.use_range_dino, use_relative_range_head=self.use_relative_range_head, use_relative_xyz_head=self.use_relative_xyz_head)

                gt_bboxes_all, (H, W) = find_all_gt_bboxes(gt_img, min_area=100)
                correct_gt_bboxes = gt_bboxes_all.get(int(item['qid'].split('-')[-1]), [])

                # if self.dataset_name == "Nr3D" or self.dataset_name == "Sr3D":
                #     distractor_gt_ids = ast.literal_eval(item["distractor_gt_id"])
                #     distractor_gt_ids.append(int(item["correct_gt_id"]))
                #     distractor_bboxes = []
                #     for d_id in distractor_gt_ids:
                #         boxes = gt_bboxes_all.get(d_id, [])
                #         distractor_bboxes.extend(boxes)
                #     all_bboxes = distractor_bboxes
                #     random.shuffle(all_bboxes)
                #     norm_bboxes = [(
                #             int(x1 * 1000 / W),
                #             int(y1 * 1000 / H),
                #             int(x2 * 1000 / W),
                #             int(y2 * 1000 / H)
                #         ) for (x1, y1, x2, y2) in all_bboxes]
                #     boxes_str = ', '.join(f'[[{x1:03d},{y1:03d},{x2:03d},{y2:03d}]]' for x1, y1, x2, y2 in norm_bboxes)
                #     item["description"] = item["description"].replace("PLACEHOLDER", boxes_str)

                img_dict = {'vision': self.image_processor(img)}
                prompt = item["description"]
                label = item["caption"]
                text_dict = self.text_processor(label, prompt)

        except Exception as e:
            print_rank0(f"Text processing failed for {item['qid']}: {e}", level=logging.WARNING)
            return {}

        if self.use_multi_view_feature and feature is None:
            print(f"Missing feature array: {feature_path}")
            return {}

        ret = {**img_dict, **text_dict, "question_id": item["qid"], "valid_box": torch.tensor([valid_box])}
        if self.use_multi_view_feature and feature is not None:
            ret["multi_view_feature"] = feature.view(-1, self.multi_view_feature_dim)
        if self.use_semantic_head and semantic is not None:
            ret["semantic_labels"] = semantic.view(-1, self.num_semantic_classes+1)
        if self.use_xyz_encoder and input_geometry is not None:
            ret["input_geometry"] = input_geometry
        if self.use_xyz_head and gt_xyz is not None:
            ret["xyz_labels"] = gt_xyz.view(-1, 3)
        if self.use_relative_xyz_head and gt_xyz is not None:
            ret["xyz_labels"] = gt_xyz
        if (self.use_range_head or self.use_relative_range_head) and gt_range is not None:
            ret["range_labels"] = gt_range.unsqueeze(0)
        return ret
