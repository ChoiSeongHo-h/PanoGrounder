import os
import torch
import argparse
from functools import partial
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from sat.helpers import print_rank0
from utils.models import FineTuneTestCogVLMModel
from utils.utils import llama2_text_processor, llama2_text_processor_inference, get_image_processor
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import update_mems
import torch.distributed as dist


def data_collator(examples):
    examples = [ex for ex in examples if len(ex) > 0] # drop {}

    if not examples:
        # 빈 리스트라면, 학습 루프에서 건너뛸 수 있도록 빈 딕셔너리를 반환합니다.
        # 또는 None을 반환할 수도 있지만, 학습 코드와의 호환성을 확인해야 합니다.
        print("data_collator: empty examples")
        return {}

    for example in examples:
        for k in example:
            if isinstance(example[k], list):
                example[k] = torch.tensor(example[k])
            elif isinstance(example[k], np.ndarray):
                example[k] = torch.from_numpy(example[k])
    img_args = {}
    tmp_example = examples[0]
    for k in tmp_example['vision']:
        if type(tmp_example['vision'][k]) is torch.Tensor:
            img_args['vision_'+k] = torch.cat([example['vision'][k] for example in examples])
        else:
            img_args['vision_'+k] = example['vision'][k]
    for example in examples:
        example.pop('vision')
        if 'cross' in example:
            example.pop('cross')

    model_args = {}
    tmp_example = examples[0]
    for k in tmp_example:
        if type(tmp_example[k]) is torch.Tensor:
            if k == 'multi_view_feature':
                model_args[k] = torch.stack([example[k] for example in examples], dim=0)
            elif k == 'input_geometry':
                model_args[k] = torch.stack([example[k] for example in examples], dim=0)
            else:
                model_args[k] = torch.cat([example[k] for example in examples])
        else:
            model_args[k] = [example[k] for example in examples]

    if 'multi_view_feature' in tmp_example and 'multi_view_feature' not in model_args:
         model_args['multi_view_feature'] = torch.stack([ex['multi_view_feature'] for ex in examples])

    if 'input_geometry' in tmp_example and 'input_geometry' not in model_args:
        model_args['input_geometry'] = torch.stack([ex['input_geometry'] for ex in examples])

    model_args.update(img_args)
    return model_args

from collections import defaultdict

def broadcast_auto(data_dict):
    type2list = defaultdict(list)
    other = []
    for k in data_dict:
        if type(data_dict[k]) is torch.Tensor:
            type2list[data_dict[k].dtype].append(k)
        else:
            other.append(k)
    new_data = {}
    for k in type2list:
        new_data.update(mpu.broadcast_data(type2list[k], data_dict, k))
    for k in other:
        new_data[k] = data_dict[k]
    return new_data

def get_batch(data_iterator, args, timers):
    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = broadcast_auto(data)
    for k in data_b:
        if type(data_b[k]) is torch.Tensor and data_b[k].dtype is not torch.int32 and data_b[k].dtype is not torch.long:
            if args.fp16:
                data_b[k] = data_b[k].half()
            elif args.bf16:
                data_b[k] = data_b[k].bfloat16()
    return data_b

from torch.nn import CrossEntropyLoss
import numpy as np

from sat.model.mixins import CachedAutoregressiveMixin

def get_masks_and_position_ids_batched(seqs, image_rope_masks, context_lengths):
    """
    배치 입력을 올바르게 처리하여 각 샘플에 맞는 어텐션 마스크와 포지션 ID를 생성합니다.
    각 샘플의 실제 길이(context_lengths)를 고려하여 패딩 영역을 무시합니다.
    """
    batch_size, seq_len = seqs.shape
    device = seqs.device

    # 패딩을 고려한 Attention Mask 생성
    # 1. 각 시퀀스의 실제 길이에 기반한 패딩 마스크 생성
    #    (실제 토큰은 1, 패딩 토큰은 0)
    padding_mask = torch.arange(seq_len, device=device)[None, :] < context_lengths[:, None]

    # 2. Causal 마스크와 패딩 마스크를 결합하여 최종 attention_mask 생성
    #    - padding_mask[:, None, :] -> Query가 패딩 위치에 있는 경우 attention을 못하도록 막음
    #    - padding_mask[:, :, None] -> Key가 패딩 위치에 있는 경우 attention을 못하도록 막음
    #    - 하삼각 행렬(tril) -> Causal 관계를 적용 (미래 토큰 참조 방지)
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    attention_mask = causal_mask * padding_mask[:, None, :] * padding_mask[:, :, None]

    # 3. 모델 입력 형식에 맞게 차원 확장 [B, 1, S, S]
    attention_mask = attention_mask.unsqueeze(1)

    # 2. Position ID 생성
    all_position_ids = []
    for i in range(batch_size):
        current_rope_mask = image_rope_masks[i]
        current_context_len = int(context_lengths[i])
        position_ids = []
        pid = -1

        for j in range(current_context_len):
            is_text = (current_rope_mask[j] == 0)
            is_new_img_group = (j > 0 and current_rope_mask[j] != current_rope_mask[j - 1])
            if is_text or is_new_img_group:
                pid += 1
            position_ids.append(pid)

        num_padding = seq_len - current_context_len
        if num_padding > 0:
            position_ids.extend([0] * num_padding)

        all_position_ids.append(torch.tensor(position_ids, dtype=torch.long, device=device))

    position_ids_tensor = torch.stack(all_position_ids, dim=0)

    return attention_mask, position_ids_tensor


def chat(model, tokenizer, tokens,
         max_length: int = 1800, max_memory_length: int = 100000,
         context_lengths: torch.Tensor = None,
         qids: list = None,
         **kwargs):
    """
    한 번에 한 토큰씩 생성하는 명시적인 디코딩 루프. (Greedy Search)
    배치 처리 시 is_finished를 고려하여 Position ID를 정확하게 계산.
    """
    device = next(model.parameters()).device

    # 1. 입력 마스크 및 토큰 초기화
    initial_vis_mask = kwargs.pop('vision_expert_mask').to(device)
    initial_img_mask = kwargs.pop('image_embed_mask').to(device)
    initial_rope_mask = kwargs.pop('image_rope_mask').to(device)

    cur_tokens = tokens.to(device)
    batch_size, prompt_len = cur_tokens.shape
    is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # =========================================================================
    # 2. Prefill 단계
    # =========================================================================

    prefill_attention_mask, prefill_position_ids = get_masks_and_position_ids_batched(cur_tokens, initial_rope_mask, context_lengths)

    logits, *output_per_layers = model(
        input_ids=cur_tokens,
        attention_mask=prefill_attention_mask,
        position_ids=prefill_position_ids,
        vision_expert_mask=initial_vis_mask,
        image_embed_mask=initial_img_mask,
        mems=None,
        **kwargs
    )

    mem_kv = [o['mem_kv'] for o in output_per_layers]
    mems = update_mems(mem_kv, None, max_memory_length=max_memory_length)

    batch_indices = torch.arange(batch_size, device=device)
    last_token_indices = context_lengths - 1
    next_logits = logits[batch_indices, last_token_indices]

    next_tok_ids = torch.argmax(next_logits, dim=-1)

    cur_tokens = torch.cat([cur_tokens, next_tok_ids.unsqueeze(1)], dim=1)
    is_finished |= (next_tok_ids == tokenizer.eos_token_id)

    last_real_indices = (context_lengths - 1).view(-1, 1).to(device)
    decoding_position_ids = torch.gather(prefill_position_ids, 1, last_real_indices)

    # =========================================================================
    # 3. Decoding 루프
    # =========================================================================
    padded_prompt_len = cur_tokens.shape[1] - 1
    max_context_len = padded_prompt_len # Prefill 시 사용된 최대 길이

    for current_len in range(padded_prompt_len, max_length - 1):
        if is_finished.all():
            break

        tokens_to_process = cur_tokens[:, -1:]

        # 각 샘플의 패딩을 고려한 동적 어텐션 마스크 생성
        # 현재까지의 총 시퀀스 길이 (KV 캐시의 길이)
        # KV 캐시의 길이는 current_len이 아니라 current_len+1이 맞음
        total_kv_len = current_len + 1

        # 1. Prefill 단계의 패딩을 반영하는 마스크를 다시 생성합니다. (길이: max_context_len)
        #    (실제 토큰은 True, 패딩 토큰은 False)
        prefill_padding_mask = torch.arange(max_context_len, device=device)[None, :] < context_lengths[:, None]

        # 2. 현재까지 생성된 토큰(Decoding 부분)은 모두 유효하므로 True로 채웁니다.
        num_generated_tokens = total_kv_len - max_context_len
        decode_mask = torch.ones((batch_size, num_generated_tokens), dtype=torch.bool, device=device)

        # 3. Prefill 마스크와 Decoding 마스크를 합쳐 전체 KV 캐시에 대한 마스크를 만듭니다.
        #    이 마스크는 각 샘플의 실제 과거 정보만을 가리킵니다.
        full_kv_mask = torch.cat([prefill_padding_mask, decode_mask], dim=1)

        # 4. 모델 입력 형식에 맞게 차원 확장 [B, 1, 1, S_kv]
        attention_mask = full_kv_mask.unsqueeze(1).unsqueeze(2)
        # ==========================================================================================

        # 종료된 시퀀스는 Position ID가 증가하지 않도록 마스킹
        decoding_position_ids += (~is_finished).unsqueeze(-1)
        position_ids = decoding_position_ids

        logits, *output_per_layers = model(
            input_ids=tokens_to_process,
            attention_mask=attention_mask,
            position_ids=position_ids,
            vision_expert_mask=None,
            image_embed_mask=None,
            mems=mems,
            **kwargs
        )

        mem_kv = [o['mem_kv'] for o in output_per_layers]
        mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)

        next_logits = logits[:, -1]
        next_tok_ids = torch.argmax(next_logits, dim=-1)

        # 종료된 시퀀스는 pad_token_id를 생성하도록 강제
        next_tok_ids = torch.where(is_finished, tokenizer.pad_token_id, next_tok_ids)
        is_finished |= (next_tok_ids == tokenizer.eos_token_id)

        cur_tokens = torch.cat([cur_tokens, next_tok_ids.unsqueeze(1)], dim=1)

    return cur_tokens

from utils.utils import process_cogvlm_output
def forward_step_eval(data_iterator, model, args, timers):
    def compute_metrics(eval_preds, qids):
        preds, labels, device = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = { "acc": [], "l1_loss": [], "giou_loss": [] }
        for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
            qid = qids[i]
            success, l1_loss, giou_loss = process_cogvlm_output(qid, pred, args)
            print("qid:", qid, "pred:", pred, "success:", success, "l1_loss:", l1_loss, "giou_loss:", giou_loss)
            if success:
                score_dict['acc'].append(1.)
            else:
                score_dict['acc'].append(0.)
            score_dict['l1_loss'].append(l1_loss)
            score_dict['giou_loss'].append(giou_loss)

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    def save_results_to_csv(header, row, csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)
        mode = 'a' if file_exists else 'w'

        with open(csv_path, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    def compute_metrics_qa(eval_preds, qids):
        preds, labels, device = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = { "rel_range_acc": [] }
        for i, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
            success = (pred.strip() == label.strip())
            print("qid:", qids[i], "pred:", pred, "label:", label, "success:", success)
            score_dict['rel_range_acc'].append(1.0 if success else 0.0)
            qid = qids[i]
            scene_id,cam_id,i,t,c = qid.split('-', 4)
            image_idx, text_idx = int(i), int(t)

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
            ]

            result_row = [
                args.dataset_name,
                scene_id,
                cam_id,
                image_idx,
                text_idx,
                success
            ]

            if args.eval:
                filename = args.experiment_name
                output_path = f"{args.output_root}/{args.dataset_name}/rank{args.rank}_geometry_{filename}.csv"
                save_results_to_csv(header, result_row, output_path)

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict


    # Get the batch.
    timers('batch generator').start()
    data_b = get_batch(data_iterator, args, timers)
    timers('batch generator').stop()

    device = next(model.parameters()).device

    # 2. 현재 배치가 유효한지 확인 (0: 유효하지 않음, 1: 유효함)
    is_valid_batch = torch.tensor(1.0 if data_b else 0.0, device=device)

    # 3. 모든 Rank의 배치 유효성 상태를 수집
    # all_gather를 위해 모든 rank에서 동일한 shape의 텐서가 필요
    valid_status_list = [torch.tensor(0.0, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(valid_status_list, is_valid_batch)

    # dist.barrier() # 선택사항: 디버깅 시 모든 프로세스가 여기까지 왔는지 확인

    # 4. 수집된 상태를 확인하여, 하나라도 유효하지 않은 배치가 있다면 다같이 건너뛰기
    if not all(status.item() == 1.0 for status in valid_status_list):
        if args.rank == 0: # 로그는 한 번만 출력
             print("forward_step_eval: At least one rank received an empty batch. Skipping this step for all ranks.")
        # 모든 Rank가 동기적으로 빈 결과를 리턴
        return torch.tensor(0.0, device=device), {}

    # 배치 내 최대 context_len을 기준으로 설정
    max_context_len = int(data_b['context_length'].max())
    context_lengths = data_b['context_length']

    # max_context_len 기준으로 슬라이싱
    tokens = data_b['input_ids'][:, :max_context_len]
    vision_expert_mask = data_b['vision_expert_mask'][:, :max_context_len]
    image_embed_mask = data_b['image_embed_mask'][:, :max_context_len]
    image_rope_mask = data_b['image_rope_mask'][:, :max_context_len]

    # =========================================================================
    # 패딩 영역을 명시적으로 0으로 초기화
    # =========================================================================
    for i in range(tokens.size(0)):
        real_len = int(context_lengths[i])
        if real_len < max_context_len:
            vision_expert_mask[i, real_len:] = 0
            image_embed_mask[i, real_len:] = 0
            image_rope_mask[i, real_len:] = 0

    data_b['vision_expert_mask'] = vision_expert_mask
    data_b['image_embed_mask'] = image_embed_mask
    data_b['image_rope_mask'] = image_rope_mask
    # =========================================================================

    data_b.pop('input_ids')
    data_b.pop('attention_mask', None)
    data_b.pop('position_ids', None)
    labels = data_b.pop('labels')
    qids = data_b.pop('question_id')

    data_b.pop('context_length')

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    outputs = chat(model, tokenizer, tokens, context_lengths=context_lengths, qids=qids, **data_b)

    generated_outputs = []
    generated_labels = []

    # 각 샘플의 실제 길이(real_context_len)가 아닌,
    # 배치의 패딩 기준이 된 `max_context_len`을 기준으로 잘라내야 올바른 생성 부분을 추출할 수 있습니다.
    for i in range(outputs.size(0)):
        generated_part = outputs[i][max_context_len:]
        generated_outputs.append(generated_part)

        # 레이블도 동일한 기준으로 슬라이싱합니다.
        generated_labels.append(labels[i][max_context_len:])

    padded_outputs = torch.nn.utils.rnn.pad_sequence(generated_outputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_labels = torch.nn.utils.rnn.pad_sequence(generated_labels, batch_first=True, padding_value=-100)

    model.del_mixin('auto-regressive')

    if hasattr(args, 'geometry_qa') and args.geometry_qa:
        metrics = compute_metrics_qa(
            (padded_outputs.cpu(), padded_labels.cpu(), padded_outputs.device), qids
        )
    else:
        metrics = compute_metrics(
            (padded_outputs.cpu(), padded_labels.cpu(), padded_outputs.device), qids
        )

    return torch.tensor(0, device=padded_outputs.device), {k: torch.tensor(v, device=padded_outputs.device) for k, v in metrics.items()}


from torch.nn import CrossEntropyLoss
def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    data_b = get_batch(
        data_iterator, args, timers)
    labels = data_b.pop('labels')
    timers('batch generator').stop()
    logits = model(**data_b)[0]
    lm_logits = logits.to(torch.float32)
    # Shift so that tokens < n predict n
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = lm_logits[..., -1-shift_labels.size(-1):-1, :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.to(torch.float32)

    return loss, {'loss': loss}

from utils.utils import VisualGroundingDataset, ItemDataset, GeometryQADataset
def create_dataset_function(image_processor, text_processor, path, args):
    if hasattr(args, 'geometry_qa') and args.geometry_qa:
        dataset = GeometryQADataset(image_processor, text_processor, args, path)
    else:
        dataset = VisualGroundingDataset(image_processor, text_processor, args, path)
    return dataset

import csv
def merge_ranked_csvs(output_dir, output_file):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    merged_path = os.path.join(output_dir, output_file)
    header_written = False
    rank_paths = [
        os.path.join(output_dir, f"rank{rank}_{output_file}")
        for rank in range(world_size)
    ]
    missing_paths = [path for path in rank_paths if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(
            "Missing rank outputs: " + ", ".join(missing_paths)
        )

    with open(merged_path, "w", newline='', encoding="utf-8") as fout:
        writer = csv.writer(fout)

        for rank_path in rank_paths:
            with open(rank_path, "r", encoding="utf-8") as fin:
                reader = csv.reader(fin)
                for i, row in enumerate(reader):
                    if i == 0 and header_written:
                        continue
                    if i == 0 and not header_written:
                        header_written = True
                    writer.writerow(row)
    print(f"Merged CSV saved to: {merged_path}")

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--max_length', type=int)
    py_parser.add_argument('--ignore_pad_token_for_loss', action='store_false')
    py_parser.add_argument("--version", type=str, default="chat", help='version to interact with')
    py_parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat", help='pretrained ckpt')
    py_parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    py_parser.add_argument("--vit_checkpoint_activations", action='store_true')
    py_parser.add_argument("--dataset_name", type=str, nargs='+', choices=["3RScan", "ScanRefer", "Nr3D", "Sr3D", "ScanNetpp", "ARKitScenes", "Structured3D"], required=True)
    py_parser.add_argument("--scene_limit", type=str, default="-1")
    py_parser.add_argument("--cam_limit", type=str, default="-1")
    py_parser.add_argument("--render_mode", type=str, nargs='+', choices=["ransac_nearcam_obs_dim_multi_many", "ransac_nearcam_obs_dim_multi_many_more", "ransac_obs_random10", "ransac_obs_random", "ransac_obs_dim_multi", "ransac_noobs_dim_multi", "back_culled", "back_culled_gscam", "mesh_low", "detection", "center", "slice", "ransac", "ransac_multi", "ransac_multi_padding", "ransac_nearcam_multi", "ransac_nearcam_obs_multi", "ransac_nearcam_obs_dim_multi"], required=True)
    py_parser.add_argument("--pano_pe_types", type=str, nargs="*", default=[], help="List of PE types to use. Options: pano_abs_pe, pano_rel_pe. Can provide 0~2 of them.")
    py_parser.add_argument("--eval_comment", type=str, default="")
    py_parser.add_argument("--eval", action="store_true", help="whether to save csv file")
    py_parser.add_argument("--output_root", type=str, default="")

    py_parser.add_argument('--use_multi_view_feature', action='store_true', help='Whether to use multi-view feature for training.')
    py_parser.add_argument('--multi_view_feature_dim', type=int, default=384, help='Dimension of the multi-view ensemble feature.')
    py_parser.add_argument('--only_use_multi_view_feature', action='store_true', help='Whether to use only multi-view feature for training.')
    py_parser.add_argument('--normalize_feature', action='store_true', help='Whether to normalize the multi-view feature.')
    py_parser.add_argument('--pre_insert_feature', action='store_true', help='Whether to insert the multi-view feature before the text adapter.')
    py_parser.add_argument('--use_cnn', action='store_true', help='Whether to use CNN for multi-view feature extraction.')
    py_parser.add_argument('--mv_cnn_mid_dim', type=int, default=512, help='Dimension of the CNN feature for multi-view feature extraction.')
    py_parser.add_argument('--use_fusion', action='store_true', help='Whether to use fusion for multi-view feature extraction.')
    py_parser.add_argument('--use_deep_feature', action='store_true', help='Whether to use deep combination for multi-view feature extraction.')
    py_parser.add_argument('--use_semantic_head', action='store_true', help='Whether to use aux head.')
    py_parser.add_argument('--num_semantic_classes', type=int, default=-1, help='Number of semantic classes.')
    py_parser.add_argument('--aux_loss_weight', type=float, default=0.5, help='Weight for the auxiliary semantic loss.')
    py_parser.add_argument('--use_range_head', action='store_true', help='Whether to use aux range head.')
    py_parser.add_argument('--aux_loss_weight_range', type=float, default=1.0, help='Weight for the auxiliary range loss.')
    py_parser.add_argument('--use_xyz_head', action='store_true', help='Whether to use aux xyz head.')
    py_parser.add_argument('--aux_loss_weight_xyz', type=float, default=1.0, help='Weight for the auxiliary xyz loss.')
    py_parser.add_argument('--wo_load_pretrained', action='store_true', help='Whether to not load pretrained weights.')
    py_parser.add_argument('--use_xyz_encoder', action='store_true', help='Whether to use xyz encoder.')
    py_parser.add_argument('--xyz_encoder_dim', type=int, default=384, help='Dimension of the xyz encoder.')
    py_parser.add_argument('--debug_print', action='store_true', help='Whether to debug xyz.')
    py_parser.add_argument('--use_deep_xyz', action='store_true', help='Whether to use deep fusion for xyz.')
    py_parser.add_argument('--lora_alpha', type=int, default=1, help='LoRA alpha.')
    py_parser.add_argument('--use_rangevit', action='store_true', help='Whether to use RangeViT for feature extraction.')
    py_parser.add_argument('--use_resize_xyz', action='store_true', help='Whether to resize xyz.')
    py_parser.add_argument('--use_semantic_head_decay', action='store_true', help='Whether to apply decay to semantic head.')
    py_parser.add_argument('--use_dinov3', action='store_true', help='Whether to use dinov3.')
    py_parser.add_argument('--use_range_dino', action='store_true', help='Whether to use range dino.')
    py_parser.add_argument('--use_relative_range_head', action='store_true', help='Whether to use relative range head.')
    py_parser.add_argument('--use_intermediate_relative_range_head', action='store_true', help='Whether to use intermediate relative range head.')
    py_parser.add_argument('--relative_range_loss_weight', type=float, default=1.0, help='Weight for the relative range loss.')
    py_parser.add_argument('--use_vit_interpolation', action='store_true', help='Whether to use vit interpolation.')
    py_parser.add_argument('--use_relative_xyz_head', action='store_true', help='Whether to use relative xyz head.')
    py_parser.add_argument('--use_intermediate_relative_xyz_head', action='store_true', help='Whether to use intermediate relative xyz head.')
    py_parser.add_argument('--geometry_qa', action='store_true', help='Whether to use geometry qa.')
    py_parser.add_argument('--control_only', action='store_true', help='Whether to control only training.')
    py_parser.add_argument('--use_language_relative_range_head', action='store_true', help='Whether to use language relative range head.')
    py_parser.add_argument('--use_language_relative_xyz_head', action='store_true', help='Whether to use language relative xyz head.')
    py_parser.add_argument('--num_rot_eval', type=int, default="4")
    py_parser.add_argument("--pinhole_mode", type=str, choices=["none", "nways", "seeground", "gt"], default="none", help="Pinhole rendering mode.")
    py_parser.add_argument("--wo_zero", action="store_true", help="disable zero conv")
    py_parser.add_argument("--use_concat", action="store_true", help="use concat fusion")



    py_parser = FineTuneTestCogVLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    valid_pe_types = {"pano_abs_pe", "pano_rel_pe"}
    if not set(args.pano_pe_types).issubset(valid_pe_types):
        raise ValueError(f"Invalid --pano_pe_types: {args.pano_pe_types}. Must be subset of {valid_pe_types}")
    if len(set(args.pano_pe_types)) != len(args.pano_pe_types):
        raise ValueError(f"--pano_pe_types contains duplicates: {args.pano_pe_types}")

    if args.use_qlora:
        args.device = 'cpu'

    if hasattr(args, 'wo_load_pretrained') and args.wo_load_pretrained:
        model, args = FineTuneTestCogVLMModel.from_pretrained(args.from_pretrained, args, build_only=True, overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size != 1 else {})
    else:
        model, args = FineTuneTestCogVLMModel.from_pretrained(args.from_pretrained, args, overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size != 1 else {})
    if args.use_qlora and torch.cuda.is_available():
        model = model.to('cuda')
    from utils.utils import llama2_tokenizer
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(args.eva_args["image_size"][0])
    text_processor = llama2_text_processor(tokenizer, args.max_length, args.image_length)

    if len(args.dataset_name) != 1 or len(args.render_mode) != 1:
        raise ValueError("Evaluation accepts exactly one dataset and render mode")
    output_dir = os.path.join(args.output_root, args.dataset_name[0])
    os.makedirs(output_dir, exist_ok=True)

    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=partial(create_dataset_function, image_processor, text_processor), collate_fn=data_collator, forward_step_eval=forward_step_eval)
    if args.rank == 0:
        filename = args.experiment_name
        merge_ranked_csvs(output_dir, f"{filename}.csv")
