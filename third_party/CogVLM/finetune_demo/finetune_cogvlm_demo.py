# Modified by the PanoGrounder Authors from CogVLM upstream.
import os
import torch
import argparse
from functools import partial
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sat import mpu, get_args, get_tokenizer
from deepspeed_training_custom import training_main
from sat.helpers import print_rank0
from utils.models import FineTuneTrainCogVLMModel
from utils.utils import llama2_text_processor, llama2_text_processor_inference, get_image_processor
from utils.models.multiview_feature_model import calculate_intra_depth_loss
import torch.distributed as dist

def disable_untrainable_params(self):
    total_trainable = 0
    enable = [('mlp', 'vit')]

    unfreeze_mlp = hasattr(self.args, 'unfreeze_mlp') and self.args.unfreeze_mlp
    upper_mlp_idx = 63 if unfreeze_mlp else 55
    lower_mlp_idx = 41 if unfreeze_mlp else 45

    freeze_mlp = hasattr(self.args, 'freeze_mlp') and self.args.freeze_mlp
    if freeze_mlp:
        upper_mlp_idx = -1
        lower_mlp_idx = -1

    # 1. 결과를 저장할 리스트를 초기화합니다.
    param_statuses = []

    if hasattr(self.args, 'use_multi_view_feature') and self.args.use_multi_view_feature:
        enable.append('mv_adapter')
        if hasattr(self.args, 'use_deep_feature') and self.args.use_deep_feature:
            enable.append('mv_adapters')
    if hasattr(self.args, 'use_xyz_encoder') and self.args.use_xyz_encoder:
        enable.append('xyz_encoder')
        enable.append('xyz_adapter')
        enable.append('xyz_adapters')
    if hasattr(self.args, 'use_semantic_head') and self.args.use_semantic_head:
        enable.append('semantic_head')
    if hasattr(self.args, 'use_range_head') and self.args.use_range_head:
        enable.append('range_head')
    if hasattr(self.args, 'use_xyz_head') and self.args.use_xyz_head:
        enable.append('xyz_head')
    if hasattr(self.args, 'use_relative_range_head') and self.args.use_relative_range_head:
        enable.append('depth_diff_head')
        if hasattr(self.args, 'use_intermediate_relative_range_head') and self.args.use_intermediate_relative_range_head:
            enable.append('intermediate_depth_diff_head')
        if hasattr(self.args, 'use_language_relative_range_head') and self.args.use_language_relative_range_head:
            enable.append('language_depth_diff_head')

        if hasattr(self.args, 'use_relative_xyz_head') and self.args.use_relative_xyz_head:
            enable.append('x_diff_head')
            enable.append('y_diff_head')
            enable.append('z_diff_head')
            if hasattr(self.args, 'use_intermediate_relative_xyz_head') and self.args.use_intermediate_relative_xyz_head:
                enable.append('intermediate_x_diff_head')
                enable.append('intermediate_y_diff_head')
                enable.append('intermediate_z_diff_head')
            if hasattr(self.args, 'use_language_relative_xyz_head') and self.args.use_language_relative_xyz_head:
                enable.append('language_x_diff_head')
                enable.append('language_y_diff_head')
                enable.append('language_z_diff_head')
    if hasattr(self.args, 'pano_pe_types'):
        enable.extend(self.args.pano_pe_types)  # 'pano_abs_pe', 'pano_rel_pe' are included in learnable param names.
    if self.args.use_ptuning:
        enable.extend(['ptuning'])
    if self.args.use_lora or self.args.use_qlora:
        enable.extend(['matrix_A', 'matrix_B'])
    if hasattr(self.args, 'only_use_multi_view_feature') and self.args.only_use_multi_view_feature:
        enable = ['mv_adapter']
        if hasattr(self.args, 'use_deep_feature') and self.args.use_deep_feature:
            enable.append('mv_adapters')
    if hasattr(self.args, 'only_use_xyz_encoder') and self.args.only_use_xyz_encoder:
        enable = ['xyz_encoder', 'xyz_adapter', 'xyz_adapters']

    for n, p in self.named_parameters():
        flag = False
        for e in enable:
            if type(e) is tuple:
                if 'fusion_mlp' not in n.lower() and e[0].lower() in n.lower() and e[1].lower() in n.lower() and upper_mlp_idx > int(n[:n.find('.mlp')].split('.')[-1]) > lower_mlp_idx:
                    flag = True
                    break
            else:
                if e.lower() in n.lower():
                    flag = True
                    break

        param_statuses.append(f"{flag},{n}\n")

        if not flag:
            p.requires_grad_(False)
        else:
            total_trainable += p.numel()
            print_rank0(n)

    print_rank0("***** Total trainable parameters: "+str(total_trainable)+" *****")

    # 3. 모든 파라미터 확인이 끝난 후, 리스트의 내용을 파일로 저장합니다.
    #    (분산 학습 환경을 고려하여 rank 0 프로세스에서만 파일을 생성하도록 가정)
    if not os.path.exists("param_trainability_status"):
        os.makedirs("param_trainability_status")
    output_filename = "param_trainability_status/" + self.args.experiment_name + ".csv"
    try:
        # print_rank0과 같은 분산 환경 함수가 없다고 가정하고, rank 0일 때만 실행되도록 직접 확인
        from torch.distributed import get_rank
        if get_rank() == 0:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write("is_trainable,parameter_name\n") # 헤더 추가
                f.writelines(param_statuses)
            print_rank0(f"***** Parameter trainability status saved to {output_filename} *****")
    except (ImportError, RuntimeError):
        # 분산 환경이 아닐 경우 그냥 실행
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("is_trainable,parameter_name\n") # 헤더 추가
            f.writelines(param_statuses)
        print(f"***** Parameter trainability status saved to {output_filename} *****")


FineTuneTrainCogVLMModel.disable_untrainable_params = disable_untrainable_params

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
                model_args[k] = torch.stack([example[k] for example in examples], dim=0) # 혹은 torch.cat
            elif k == 'semantic_labels':
                model_args[k] = torch.stack([example[k] for example in examples], dim=0)
            elif k == 'range_labels':
                model_args[k] = torch.stack([example[k] for example in examples], dim=0)
            elif k == 'xyz_labels':
                model_args[k] = torch.stack([example[k] for example in examples], dim=0)
            elif k == 'input_geometry':
                model_args[k] = torch.stack([example[k] for example in examples], dim=0)
            else:
                model_args[k] = torch.cat([example[k] for example in examples])
        else:
            model_args[k] = tmp_example[k]

    # 만약 multi_view_feature가 for 루프에서 처리되지 않았다면,
    # tmp_example에만 존재할 수 있으므로 명시적으로 처리
    if 'multi_view_feature' in tmp_example and 'multi_view_feature' not in model_args:
         model_args['multi_view_feature'] = torch.stack([ex['multi_view_feature'] for ex in examples])

    if 'semantic_labels' in tmp_example and 'semantic_labels' not in model_args:
        model_args['semantic_labels'] = torch.stack([ex['semantic_labels'] for ex in examples])

    if 'range_labels' in tmp_example and 'range_labels' not in model_args:
        model_args['range_labels'] = torch.stack([ex['range_labels'] for ex in examples])

    if 'xyz_labels' in tmp_example and 'xyz_labels' not in model_args:
        model_args['xyz_labels'] = torch.stack([ex['xyz_labels'] for ex in examples])

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
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy


def chat(model, tokenizer, tokens,
         max_length: int = 1800, num_beams=5, top_p=0.95, top_k=0, temperature=0.8, **kwargs):
    inputs = tokens.to(model.parameters().__next__().device)[0]
    seq = torch.cat(
        [inputs, torch.tensor([-1] * (max_length - len(inputs)), device=inputs.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=0.4, top_k=1, end_tokens=[tokenizer.eos_token_id])
    # strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[tokenizer.eos_token_id],
    #                               num_beams=num_beams, consider_end=True)
    get_func = llama2_text_processor_inference.get_func(None, None, image_rope_mask=kwargs['image_rope_mask'])
    output = filling_sequence(
        model, seq,
        batch_size=1,
        strategy=strategy,
        get_masks_and_position_ids=get_func,
        **kwargs
    )[0]  # drop memory

    return output


from utils.utils import process_cogvlm_output
def forward_step_eval(data_iterator, model, args, timers):
    def compute_metrics(eval_preds):
        preds, labels, device = eval_preds
        preds = preds.unsqueeze(0)
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "acc": [],
            "l1_loss": [],
            "giou_loss": [],
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            if args.pretrain_geometry:
                success = pred == label
                l1_loss = 0.0
                giou_loss = 0.0
            else:
                success, l1_loss, giou_loss = process_cogvlm_output(qid, pred, args)

            if args.rank == 0:
                print("success:", success, "l1_loss:", l1_loss, "giou_loss:", giou_loss)
            if success:
                score_dict['acc'].append(1.)
            else:
                score_dict['acc'].append(0.)
            score_dict['l1_loss'].append(l1_loss)
            score_dict['giou_loss'].append(giou_loss)


        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    # Get the batch.
    timers('batch generator').start()
    data_b = get_batch(
        data_iterator, args, timers)
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

    context_len = int(data_b['context_length'][0])
    tokens = data_b['input_ids'][:, :context_len]
    data_b['vision_expert_mask'] = data_b['vision_expert_mask'][:, :context_len]
    data_b['image_embed_mask'] = data_b['image_embed_mask'][:, :context_len]
    data_b['image_rope_mask'] = data_b['image_rope_mask'][:, :context_len]

    data_b.pop('input_ids')
    data_b.pop('attention_mask')
    data_b.pop('position_ids')
    labels = data_b.pop('labels')
    qid = data_b.pop('question_id')

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    outputs = chat(model, tokenizer, tokens, **data_b)[0][context_len:]
    # print(outputs)
    model.del_mixin('auto-regressive')

    return torch.tensor(0, device=outputs.device), {k: torch.tensor(v, device=outputs.device) for k, v in
                                                    compute_metrics(
                                                        (outputs.cpu(), labels.cpu(), outputs.device)).items()}


from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import deepspeed
def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # PE 파라미터 초기화
    pano_abs_pe_alpha = pano_abs_pe_linear_weight = pano_abs_pe_linear_bias = None
    pano_rel_pe_alpha = pano_rel_pe_beta = None

    # PE 파라미터 필요 여부 판단
    use_abs = "pano_abs_pe" in args.pano_pe_types
    use_rel = "pano_rel_pe" in args.pano_pe_types

    if use_abs or use_rel:
        pos_embedding = model.get_mixin("eva").vit_model.get_mixin("pos_embedding")
        xattn = model.get_mixin("eva").vit_model.get_mixin("xattn")

        gather_params = []
        if use_abs:
            gather_params += [
                pos_embedding.pano_abs_pe_alpha,
                pos_embedding.pano_abs_pe_linear.weight,
                pos_embedding.pano_abs_pe_linear.bias
            ]
        if use_rel:
            gather_params += [
                xattn.pano_rel_pe_alpha,
                xattn.pano_rel_pe_beta
            ]

        with deepspeed.zero.GatheredParameters(gather_params, modifier_rank=0):
            if use_abs:
                pano_abs_pe_alpha = pos_embedding.pano_abs_pe_alpha
                pano_abs_pe_linear_weight = pos_embedding.pano_abs_pe_linear.weight
                pano_abs_pe_linear_bias = pos_embedding.pano_abs_pe_linear.bias
            if use_rel:
                pano_rel_pe_alpha = xattn.pano_rel_pe_alpha
                pano_rel_pe_beta = xattn.pano_rel_pe_beta

            # print("CHECKPOINT 1")
            # print("pano_abs_pe_alpha:", pano_abs_pe_alpha.data, pano_abs_pe_alpha.requires_grad)
            # print("pano_abs_pe_linear_weight:", pano_abs_pe_linear_weight.data[:5, :5], pano_abs_pe_linear_weight.requires_grad)
            # print("pano_abs_pe_linear_bias:", pano_abs_pe_linear_bias.data[:5], pano_abs_pe_linear_bias.requires_grad)

    # Hyperparameters
    l1_weight = getattr(args, 'l1_weight', 10.0)
    resolution = 1000.0  # fixed image resolution for normalization

    # Get the batch.
    timers('batch generator').start()
    data_b = get_batch(
        data_iterator, args, timers)
    labels = data_b.pop('labels')
    if hasattr(args, 'use_semantic_head') and args.use_semantic_head:
        semantic_labels = data_b.pop('semantic_labels', None)
    if (hasattr(args, 'use_range_head') and args.use_range_head) or (hasattr(args, 'use_relative_range_head') and args.use_relative_range_head):
        range_labels = data_b.pop('range_labels', None).squeeze(-1) / 1000.0

    x_labels = y_labels = z_labels = None
    if hasattr(args, 'use_relative_xyz_head') and args.use_relative_xyz_head:
        xyz_labels_temp = data_b.pop('xyz_labels', None)
        x_labels = xyz_labels_temp[:, :, :, 0].unsqueeze(1)
        y_labels = xyz_labels_temp[:, :, :, 1].unsqueeze(1)
        z_labels = xyz_labels_temp[:, :, :, 2].unsqueeze(1)
    if hasattr(args, 'use_xyz_head') and args.use_xyz_head:
        xyz_labels = data_b.pop('xyz_labels', None).squeeze(-1) / 1000.0
    timers('batch generator').stop()
    logits, *output_per_layers = model(**data_b)
    language_feat = output_per_layers[-1]['hidden_states']
    lm_logits = logits.to(torch.float32)
    # Shift so that tokens < n predict n
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = lm_logits[..., -1-shift_labels.size(-1):-1, :].contiguous()

    # Filter valid masks
    valid_box = data_b.pop('valid_box').to(torch.bool)
    shift_labels = shift_labels[valid_box].contiguous()
    shift_logits = shift_logits[valid_box].contiguous()

    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.to(torch.float32)

    # 2) Regression L1 loss
    # 2.1) 확률 분포로부터 기대값(expectation) 계산
    probs = F.softmax(shift_logits, dim=-1)  # [batch, seq, vocab]
    # digit 토큰 id들
    digit_tokens = [str(i) for i in range(10)]
    digit_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(digit_tokens),
        device=probs.device
    )  # [10]

    # 각 위치에서 각 digit 확률만 골라내기 -> [batch, seq, 10]
    digit_probs = probs[..., digit_ids]
    # 기대값: sum(d * p_d)
    digit_range = torch.arange(10, device=probs.device, dtype=torch.float32)
    exp_vals = torch.sum(digit_probs * digit_range.view(1,1,10), dim=-1)  # [batch, seq]

    # 2.2) 실제 숫자(target) 계산 (loop 이용)
    batch_size, seq_len = shift_labels.size()
    device = shift_labels.device
    target_tensor = torch.zeros((batch_size, seq_len), device=device, dtype=torch.float32)
    place_tensor = torch.zeros((batch_size, seq_len), device=device, dtype=torch.float32)

    for b in range(batch_size):
        # 토큰 id -> token string, pad(-100) -> pad id
        lab = shift_labels[b].clone()
        lab[lab == -100] = tokenizer.pad_token_id
        toks = tokenizer.convert_ids_to_tokens(lab.tolist(), skip_special_tokens=False)
        # 연속된 digit span 찾기
        spans, start = [], None
        for i, tok in enumerate(toks):
            if tok.isdigit(): start = i if start is None else start
            else:
                if start is not None:
                    spans.append((start, i-1))
                    start = None
        if start is not None:
            spans.append((start, len(toks)-1))

        # span별로 target과 place 구성
        for s, e in spans:
            num = int(''.join(toks[s:e+1]))
            for i in range(s, e+1):
                target_tensor[b, i] = float(num)
                place_tensor[b, i] = 10 ** (e - i)

    # 2.3) L1 Loss 계산 (기존 L1 또는 EMD L1)
    # 실제 정답 자릿수(D_i) 계산
    digit_val = ((target_tensor / place_tensor).floor() % 10)  # [batch, seq]

    # 마스크: 숫자 스팬이면서 유효한 토큰(pad 아님)
    mask = (place_tensor > 0) & (shift_labels != -100)      # [B, S]

    # EMD 사용 여부에 따라 l1_raw 계산 분기
    use_emd = getattr(args, 'use_emd', False)

    if use_emd:
        digit_val = torch.nan_to_num(digit_val, nan=0.0, posinf=0.0, neginf=0.0)
        place_tensor = torch.nan_to_num(place_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        # --- EMD L1 Loss (L_EMD = P_i * Sum[ P(d) * |D_i - d| ]) ---

        # 1. |D_i - d| 계산 (정답 자릿수와 0~9 각 숫자간의 거리)
        # digit_val: [B, S] (D_i) -> [B, S, 1]
        # digit_range: [10] (d) -> [1, 1, 10]
        distances = torch.abs(
            digit_val.unsqueeze(-1) - digit_range.view(1, 1, 10)
        ) # [B, S, 10]

        # 2. Sum[ P(d) * |D_i - d| ] (거리의 기대값)
        # digit_probs: [B, S, 10] (P(d))
        emd_term = torch.sum(digit_probs * distances, dim=-1) # [B, S]

        # 3. L_EMD = P_i * (EMD term)
        # place_tensor: [B, S] (P_i, 자릿값)
        l1_raw = place_tensor * emd_term  # [B, S]

    else:
        # --- 기존 L1 Regression Loss ---
        # 회귀 예측값: exp_vals * place + (실제 수 - 해당 자리 digit)*place 의 truncated 부분
        truncated = target_tensor - digit_val * place_tensor
        regress_tensor = exp_vals * place_tensor + truncated      # [batch, seq]

        # L1 raw
        l1_raw = torch.abs(target_tensor - regress_tensor)      # [B, S]

    # L1 정규화 (Normalization)
    l1_norm = l1_raw / resolution                             # [B, S]

    # 마스크된 위치의 손실값만 평균
    l1_loss = l1_weight * l1_norm[mask].mean()

    if args.loss_types == "ce":
        total_loss = loss
    elif args.loss_types == "l1":
        total_loss = l1_loss
    elif args.loss_types == "ce_l1":
        total_loss = loss + l1_loss

    # # 3) Debug print (no grad)
    # with torch.no_grad():
    #     print(f"\n=== Debug (batch size={batch_size}) ===")
    #     ce_scalar = loss.item()
    #     l1_scalar = l1_loss.item()
    #     print(f"CE loss: {ce_scalar}")
    #     print(f"L1 loss: {l1_scalar}")
    #     print(f"L1 ratio: {(l1_scalar / (ce_scalar + l1_scalar)):.4f}")
    #     print("idx | pred_token | true_token | regress | L1_raw | L1_norm")
    #     preds = shift_logits.argmax(dim=-1)  # [batch, seq]
    #     for b in range(batch_size):
    #         for i in range(seq_len):
    #             if shift_labels[b,i] == -100:
    #                 continue
    #             pt = tokenizer.convert_ids_to_tokens([preds[b,i].item()])[0]
    #             tt = tokenizer.convert_ids_to_tokens([shift_labels[b,i].item()])[0]
    #             print(f"{b},{i} | {pt} | {tt} | "
    #                   f"{regress_tensor[b,i].item():.4f} | "
    #                   f"{l1_raw[b,i].item():.4f} | "
    #                   f"{l1_norm[b,i].item():.6f}")
    #     print("==============================\n")

    # aux_loss 초기화
    aux_loss = torch.tensor(0.0, device=total_loss.device, dtype=torch.float32)

    if hasattr(args, 'use_semantic_head') and args.use_semantic_head:
        vit_features = model.get_mixin("eva").latest_vit_features
        if vit_features is not None:
            semantic_logits = model.semantic_head(vit_features) # (B, N, 40)

            # 1. 아웃라이어가 아닌 정상 패치(inlier) 마스크 생성
            # semantic_labels의 첫 번째 채널이 0인 경우를 정상 패치로 간주합니다.
            # (B, N) 형태의 boolean 마스크가 생성됩니다.
            with torch.no_grad():
                # semantic_labels[:, :, 0] == 0 -> 정상 패치 (True)
                # semantic_labels[:, :, 0] == 1 -> 아웃라이어 (False)
                inlier_mask = (semantic_labels[..., 0] == 0)

            # 2. 정상 패치가 하나도 없는 경우 loss는 0으로 처리
            if not inlier_mask.any():
                aux_loss = torch.tensor(0.0, device=total_loss.device, dtype=torch.float32)
            else:
                # 3. 정상 패치에 해당하는 logits과 soft labels만 필터링
                # filtered_logits: (num_inliers, 40)
                filtered_logits = semantic_logits[inlier_mask]

                # 아웃라이어 채널을 제외한 soft labels를 가져온 후, 마스크를 적용합니다.
                # filtered_soft_labels: (num_inliers, 40)
                filtered_soft_labels = semantic_labels[..., 1:][inlier_mask]

                # [Insight] 만약 semantic_labels의 1~40번 채널의 합이 1임을 보장할 수 없다면,
                # 아래와 같이 재정규화(re-normalization)를 수행하는 것이 더 안정적일 수 있습니다.
                # with torch.no_grad():
                #     filtered_soft_labels = filtered_soft_labels / torch.sum(filtered_soft_labels, dim=-1, keepdim=True)

                # 4. Soft Label에 대한 Cross-Entropy Loss 계산
                # logits에 log_softmax를 적용하고, 타겟(soft_labels)과 곱한 후 합산합니다.
                # 이는 KL-Divergence Loss와 동일한 결과를 내며, soft label 타겟에 대한 표준적인 loss 계산법입니다.
                log_probs = F.log_softmax(filtered_logits, dim=-1)

                # loss = -Σ(target * log_probs)
                # 각 패치(샘플)에 대한 loss를 계산한 후, 전체 패치에 대해 평균을 냅니다.
                aux_loss = -torch.sum(filtered_soft_labels * log_probs, dim=-1).mean()

                # 5. Loss 가중치 적용
                aux_loss_weight = getattr(args, 'aux_loss_weight', 0.5)
                use_semantic_head_decay = getattr(args, 'use_semantic_head_decay', False)
                # 1 - args.iteration / args.train_iters 만큼 aux_loss_weight 사용
                if use_semantic_head_decay and hasattr(args, 'train_iters') and hasattr(args, 'iteration'):
                    aux_loss_weight = aux_loss_weight * (1 - args.iteration / args.train_iters)
                    if aux_loss_weight < 0:
                        aux_loss_weight = 0

                aux_loss *= aux_loss_weight

            print_rank0(f"Auxiliary semantic loss: {aux_loss.item()}")
            total_loss += aux_loss


    aux_loss_range = torch.tensor(0.0, device=total_loss.device, dtype=torch.float32)

    if hasattr(args, 'use_range_head') and args.use_range_head:
        vit_features = model.get_mixin("eva").latest_vit_features
        if vit_features is not None:
            # 모델의 range_head를 통해 각 패치의 range를 추정합니다. (B, N)
            est_range = model.range_head(vit_features)

            # 1. 유효한 range label에 대한 마스크를 생성합니다.
            # range_labels 텐서에서 0보다 큰 값만 유효한(valid) 거리 값으로 간주합니다.
            # (B, N) 형태의 boolean 마스크가 생성됩니다.
            with torch.no_grad():
                valid_range_mask = (range_labels > 0)

            # 2. 배치 내에 유효한 range label이 하나도 없는 경우, range loss는 0으로 처리합니다.
            if not valid_range_mask.any():
                aux_loss_range = torch.tensor(0.0, device=total_loss.device, dtype=torch.float32)
            else:
                # 3. 생성된 마스크를 사용해 예측값(est_range)과 정답(range_labels)에서 유효한 값만 필터링합니다.
                # 이 과정을 통해 (B, N) 텐서가 1차원 텐서 (num_valid_patches,)로 변환됩니다.
                filtered_est_range = est_range[valid_range_mask]
                filtered_range_labels = range_labels[valid_range_mask]

                # 4. 필터링된 값들에 대해 L1 Loss(평균 절대 오차)를 계산합니다.
                # 이는 예측된 거리와 실제 거리 간의 차이의 절댓값에 대한 평균입니다.
                aux_loss_range = F.l1_loss(filtered_est_range, filtered_range_labels)

                # 5. Loss에 가중치를 적용합니다. (args에 aux_loss_weight_range가 정의되어 있다고 가정)
                # 이 가중치는 전체 손실에서 range 예측 작업의 중요도를 조절하는 하이퍼파라미터입니다.
                if hasattr(args, 'aux_loss_weight_range'):
                    aux_loss_range *= args.aux_loss_weight_range

            # 6. 계산된 range loss를 로그로 출력하고 total_loss에 더해줍니다.
            print_rank0(f"Auxiliary range loss: {aux_loss_range.item()}")
            total_loss += aux_loss_range

    relative_range_loss = torch.tensor(0.0, device=total_loss.device, dtype=torch.float32)
    intermediate_relative_range_loss = torch.tensor(0.0, device=total_loss.device, dtype=torch.float32)
    language_relative_range_loss = torch.tensor(0.0, device=total_loss.device, dtype=torch.float32)

    decayed_relative_range_loss_weight = getattr(args, 'relative_range_loss_weight', 1.0)
    relative_range_loss_weight = getattr(args, 'relative_range_loss_weight', 1.0)
    use_last_range_head_decay_only = getattr(args, 'use_last_range_head_decay_only', False)
    use_range_head_decay = getattr(args, 'use_range_head_decay', False)
    # 1 - args.iteration / args.train_iters 만큼 relative_range_loss_weight 사용
    if use_range_head_decay and hasattr(args, 'train_iters') and hasattr(args, 'iteration'):
        decayed_relative_range_loss_weight = decayed_relative_range_loss_weight * (1 - args.iteration / args.train_iters)
        if decayed_relative_range_loss_weight < 0:
            decayed_relative_range_loss_weight = 0

    gt_xyz_maps = None
    if x_labels is not None and y_labels is not None and z_labels is not None:
        gt_xyz_maps = [x_labels, y_labels, z_labels]

    use_vit_interpolation = getattr(args, 'use_vit_interpolation', False)
    if hasattr(args, 'use_relative_range_head') and args.use_relative_range_head:
        vit_features = model.get_mixin("eva").latest_vit_features
        num_sampled_points = getattr(args, 'num_sampled_points', 256)
        if vit_features is not None:
            xyz_diff_heads = None
            if hasattr(args, 'use_relative_xyz_head') and args.use_relative_xyz_head:
                x_diff_head = model.x_diff_head
                y_diff_head = model.y_diff_head
                z_diff_head = model.z_diff_head
                xyz_diff_heads = [x_diff_head, y_diff_head, z_diff_head]

            relative_range_loss = calculate_intra_depth_loss(model.depth_diff_head, vit_features, range_labels, use_vit_interpolation=use_vit_interpolation, xyz_diff_heads=xyz_diff_heads, gt_xyz_maps=gt_xyz_maps, num_points=num_sampled_points)

            relative_range_loss *= decayed_relative_range_loss_weight

            print_rank0(f"Auxiliary relative range loss: {relative_range_loss.item()}")
            total_loss += relative_range_loss

        if hasattr(args, 'use_intermediate_relative_range_head') and args.use_intermediate_relative_range_head:
            range_features = model.get_mixin("eva").latest_range_features
            if range_features is not None:
                intermediate_xyz_diff_heads = None
                if hasattr(args, 'use_intermediate_relative_xyz_head') and args.use_intermediate_relative_xyz_head:
                    intermediate_x_diff_head = model.intermediate_x_diff_head
                    intermediate_y_diff_head = model.intermediate_y_diff_head
                    intermediate_z_diff_head = model.intermediate_z_diff_head
                    intermediate_xyz_diff_heads = [intermediate_x_diff_head, intermediate_y_diff_head, intermediate_z_diff_head]
                intermediate_relative_range_loss = calculate_intra_depth_loss(model.intermediate_depth_diff_head, range_features, range_labels, use_vit_interpolation=use_vit_interpolation, xyz_diff_heads=intermediate_xyz_diff_heads, gt_xyz_maps=gt_xyz_maps, num_points=num_sampled_points)

                if use_last_range_head_decay_only:
                    intermediate_relative_range_loss *= relative_range_loss_weight
                else:
                    intermediate_relative_range_loss *= decayed_relative_range_loss_weight

                print_rank0(f"Auxiliary intermediate relative range loss: {intermediate_relative_range_loss.item()}")
                total_loss += intermediate_relative_range_loss

        if hasattr(args, 'use_language_relative_range_head') and args.use_language_relative_range_head:
            vision_expert_mask = data_b['vision_expert_mask'] = data_b['vision_expert_mask']

            # 1. vision_expert_mask를 boolean 타입으로 변환하여 인덱싱 준비
            vision_mask_bool = vision_expert_mask.bool()

            # 2. 마스크를 사용해 전체 비전 관련 피처(CLS 토큰 포함)를 추출
            # 이 연산의 결과는 (B * 1226, D) 형태의 1차원 텐서가 됩니다.
            all_vision_features_flat = language_feat[vision_mask_bool]

            # 3. 배치 차원을 복원하여 (B, 1226, D) 형태로 재구성
            # 가정: 배치의 모든 샘플은 동일한 개수(1226)의 비전 토큰을 가짐
            B, _, D = language_feat.shape
            # 참고: num_vision_tokens_total = 1226
            all_vision_features = all_vision_features_flat.view(B, -1, D)

            # 4. 첫 번째 토큰(CLS 토큰)을 제외하고 나머지 패치 토큰들을 슬라이싱
            # all_vision_features[:, 1:, :]는 각 배치의 1번 인덱스부터 끝까지의 토큰을 선택
            vision_language_feat = all_vision_features[:, 1:, :]

            language_xyz_diff_heads = None
            if hasattr(args, 'use_language_relative_xyz_head') and args.use_language_relative_xyz_head:
                language_x_diff_head = model.language_x_diff_head
                language_y_diff_head = model.language_y_diff_head
                language_z_diff_head = model.language_z_diff_head
                language_xyz_diff_heads = [language_x_diff_head, language_y_diff_head, language_z_diff_head]
            language_relative_range_loss = calculate_intra_depth_loss(model.language_depth_diff_head, vision_language_feat, range_labels, use_vit_interpolation=use_vit_interpolation, xyz_diff_heads=language_xyz_diff_heads, gt_xyz_maps=gt_xyz_maps, num_points=num_sampled_points)

            language_relative_range_loss *= decayed_relative_range_loss_weight

            print_rank0(f"Auxiliary language relative range loss: {language_relative_range_loss.item()}")
            total_loss += language_relative_range_loss

    # xyz 보조 손실(auxiliary loss)을 위한 변수를 0으로 초기화합니다.
    aux_loss_xyz = torch.tensor(0.0, device=total_loss.device, dtype=torch.float32)

    # xyz 좌표를 예측하는 헤드(head)를 사용하는지 확인합니다.
    if hasattr(args, 'use_xyz_head') and args.use_xyz_head:
        # Vision Transformer의 특징 맵(feature map)을 가져옵니다.
        vit_features = model.get_mixin("eva").latest_vit_features
        if vit_features is not None:
            # 1. 모델의 xyz_head를 통해 각 패치의 xyz 좌표를 추정합니다. (B, N, 3)
            est_xyz = model.xyz_head(vit_features)

            # 2. 유효한 xyz 레이블에 대한 마스크를 생성합니다.
            #    xyz_labels 텐서에서 [0, 0, 0] 벡터가 아닌 값만 유효한(valid) 좌표로 간주합니다.
            #    (B, N) 형태의 boolean 마스크가 생성됩니다.
            with torch.no_grad():
                # 각 패치(dim=-1)의 (x, y, z) 값 중 하나라도 0이 아니면 True가 됩니다.
                valid_xyz_mask = torch.any(xyz_labels != 0, dim=-1)

            # 3. 배치 내에 유효한 xyz 레이블이 하나도 없는 경우, xyz loss는 0으로 처리합니다.
            if not valid_xyz_mask.any():
                aux_loss_xyz = torch.tensor(0.0, device=total_loss.device, dtype=torch.float32)
            else:
                # 4. 생성된 마스크를 사용해 예측값(est_xyz)과 정답(xyz_labels)에서 유효한 값만 필터링합니다.
                #    이 과정을 통해 (B, N, 3) 텐서가 (num_valid_patches, 3) 크기의 텐서로 변환됩니다.
                filtered_est_xyz = est_xyz[valid_xyz_mask]
                filtered_xyz_labels = xyz_labels[valid_xyz_mask]

                # 5. 필터링된 값들에 대해 L1 Loss(평균 절대 오차)를 계산합니다.
                #    예측된 xyz 좌표와 실제 xyz 좌표 간의 오차를 계산합니다.
                aux_loss_xyz = F.l1_loss(filtered_est_xyz, filtered_xyz_labels)

                # 6. Loss에 가중치를 적용합니다. (args에 aux_loss_weight_xyz가 정의되어 있다고 가정)
                #    이 가중치는 전체 손실에서 xyz 예측 작업의 중요도를 조절하는 하이퍼파라미터입니다.
                if hasattr(args, 'aux_loss_weight_xyz'):
                    aux_loss_xyz *= args.aux_loss_weight_xyz

            # 7. 계산된 xyz loss를 로그로 출력하고 total_loss에 더해줍니다.
            print_rank0(f"Auxiliary xyz loss: {aux_loss_xyz.item()}")
            total_loss += aux_loss_xyz

    metrics = { 'loss': total_loss, 'ce_loss': loss, 'l1_loss': l1_loss, 'aux_loss': aux_loss, 'aux_loss_range': aux_loss_range, 'aux_loss_rel_range': relative_range_loss, 'aux_loss_im_rel_range': intermediate_relative_range_loss, 'aux_loss_lang_rel_range': language_relative_range_loss, 'aux_loss_xyz': aux_loss_xyz }

    if use_abs:
        metrics.update({
            'pano_abs_pe_alpha': pano_abs_pe_alpha,
            'pano_abs_pe_linear_weight': pano_abs_pe_linear_weight.norm(),
            'pano_abs_pe_linear_bias': pano_abs_pe_linear_bias.norm()
        })

    if use_rel:
        metrics.update({
            'pano_rel_pe_alpha': pano_rel_pe_alpha.norm(),
            'pano_rel_pe_beta': pano_rel_pe_beta.norm()
        })

    return total_loss, metrics

from utils.utils import VisualGroundingDataset, ItemDataset
def create_dataset_function(image_processor, text_processor, path, args):
    dataset = VisualGroundingDataset(image_processor, text_processor, args, path)
    return dataset

from sat.model.finetune.lora2 import LoraMixin
from sat.model.finetune.prompt_tuning import PTuningV2Mixin

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--max_length', type=int)
    py_parser.add_argument('--ignore_pad_token_for_loss', action='store_false')
    py_parser.add_argument("--version", type=str, default="chat_old", help='version to interact with')
    py_parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat", help='pretrained ckpt')
    py_parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    py_parser.add_argument("--vit_checkpoint_activations", action='store_true')
    py_parser.add_argument("--dataset_name", type=str, nargs='+', choices=["ScanRefer", "Nr3D", "Sr3D", "ScanNetpp", "ARKitScenes", "Structured3D"], required=True)
    py_parser.add_argument("--scene_limit", type=str, default="-1")
    py_parser.add_argument("--cam_limit", type=str, default="-1")
    py_parser.add_argument("--render_mode", type=str, nargs='+', choices=["mesh_low", "ransac_nearcam_obs_dim_multi", "back_culled", "mesh_high", "detection", "center", "slice", "ransac", "ransac_multi", "ransac_multi_padding", "ransac_nearcam_multi", "ransac_nearcam_obs_multi", "ransac_nearcam_obs_dim_multi", "back_culled&back_culled&back_culled"], required=True)
    py_parser.add_argument("--pano_pe_types", type=str, nargs="*", default=[], help="List of PE types to use. Options: pano_abs_pe, pano_rel_pe. Can provide 0~2 of them.")
    py_parser.add_argument("--loss_types", type=str, choices=["ce", "l1", "ce_l1"], required=True)
    py_parser.add_argument("--eval_comment", type=str, default="")
    py_parser.add_argument("--eval", action="store_true", help="whether to save csv file")
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
    py_parser.add_argument('--only_use_xyz_encoder', action='store_true', help='Whether to use only xyz encoder for training.')
    py_parser.add_argument('--lr_scale', type=float, default=10.0, help='Learning rate scaling factor.')
    py_parser.add_argument('--fusion_dropout', type=float, default=0.0, help='Dropout rate for fusion module.')
    py_parser.add_argument('--fusion_intermediate_dim', type=int, default=None, help='Intermediate dimension for fusion module.')
    py_parser.add_argument('--debug_print', action='store_true', help='Whether to debug xyz.')
    py_parser.add_argument('--use_deep_xyz', action='store_true', help='Whether to use deep fusion for xyz.')
    py_parser.add_argument('--use_xyz_aug', action='store_true', help='Whether to use xyz augmentation.')
    py_parser.add_argument('--xyz_aug_num', type=int, default=8, help='Number of augmentations for xyz.')
    py_parser.add_argument('--use_fourier_features', action='store_true', help='Whether to use fourier features for xyz.')
    py_parser.add_argument('--fourier_feature_dim', type=int, default=6, help='Dimension of fourier features for xyz.')
    py_parser.add_argument('--use_rangevit', action='store_true', help='Whether to use RangeViT for feature extraction.')
    py_parser.add_argument('--detach_standard_grad', action='store_true', help='Whether to detach standard gradient.')
    py_parser.add_argument('--lora_alpha', type=int, default=1, help='LoRA alpha.')
    py_parser.add_argument('--use_resize_xyz', action='store_true', help='Whether to resize xyz.')
    py_parser.add_argument('--use_semantic_head_decay', action='store_true', help='Whether to apply decay to semantic head.')
    py_parser.add_argument('--unfreeze_mlp', action='store_true', help='Whether to unfreeze the MLP layers in EVA.')
    py_parser.add_argument('--use_range_dino', action='store_true', help='Whether to use range dino.')
    py_parser.add_argument('--use_relative_range_head', action='store_true', help='Whether to use relative range head.')
    py_parser.add_argument('--relative_range_loss_weight', type=float, default=1.0, help='Weight for the relative range loss.')
    py_parser.add_argument('--use_dinov3', action='store_true', help='Whether to use dinov3.')
    py_parser.add_argument('--use_intermediate_relative_range_head', action='store_true', help='Whether to use intermediate relative range head.')
    py_parser.add_argument('--use_range_head_decay', action='store_true', help='Whether to apply decay to range head.')
    py_parser.add_argument('--use_last_range_head_decay_only', action='store_true', help='Whether to apply decay to last range head.')
    py_parser.add_argument('--use_vit_interpolation', action='store_true', help='Whether to use vit interpolation.')
    py_parser.add_argument('--xyz_adapter_start', type=int, default=21, help='Start layer for xyz adapter.')
    py_parser.add_argument('--xyz_adapter_end', type=int, default=42, help='End layer for xyz adapter.')
    py_parser.add_argument('--use_relative_xyz_head', action='store_true', help='Whether to use relative xyz head.')
    py_parser.add_argument('--use_intermediate_relative_xyz_head', action='store_true', help='Whether to use intermediate relative xyz head.')
    py_parser.add_argument('--use_language_relative_range_head', action='store_true', help='Whether to use language relative range head.')
    py_parser.add_argument('--use_language_relative_xyz_head', action='store_true', help='Whether to use language relative xyz head.')
    py_parser.add_argument('--num_sampled_points', type=int, default=256, help='Number of sampled points.')
    py_parser.add_argument('--control_only', action='store_true', help='Whether to control only training.')
    py_parser.add_argument('--ratio_align_data', type=float, default=0.0, help='Ratio of alignment data to use.')
    py_parser.add_argument('--mvf_adapter_start', type=int, default=42, help='Start layer for mvf adapter.')
    py_parser.add_argument('--mvf_adapter_end', type=int, default=63, help='End layer for mvf adapter.')
    py_parser.add_argument('--use_emd', action='store_true', help='Whether to use emd loss.')
    py_parser.add_argument('--freeze_mlp', action='store_true', help='Whether to freeze the MLP layers in EVA.')
    py_parser.add_argument('--augment_align_data', action='store_true', help='Whether to augment alignment data.')
    py_parser.add_argument('--viz_align_data', action='store_true', help='Whether to visualize alignment data.')
    py_parser.add_argument("--pinhole_mode", type=str, choices=["none", "nways", "seeground", "gt"], default="none", help="Pinhole rendering mode.")
    py_parser.add_argument("--pretrain_geometry", action="store_true", help="Whether to pretrain geometry encoder.")
    py_parser.add_argument("--wo_zero", action="store_true", help="disable zero conv")
    py_parser.add_argument("--use_concat", action="store_true", help="use concat fusion")



    py_parser = FineTuneTrainCogVLMModel.add_model_specific_args(py_parser)
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

    args.force_train = True
    if hasattr(args, 'wo_load_pretrained') and args.wo_load_pretrained:
        model, args = FineTuneTrainCogVLMModel.from_pretrained(args.from_pretrained, args, build_only=True, overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size != 1 else {})
    else:
        model, args = FineTuneTrainCogVLMModel.from_pretrained(args.from_pretrained, args, overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size != 1 else {})
    if args.use_ptuning:
        model.add_mixin("ptuning", PTuningV2Mixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.pre_seq_len))
    if args.use_lora:
        model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, lora_alpha=args.lora_alpha, layer_range=args.layer_range), reinit=True)
        model.get_mixin("eva").vit_model.add_mixin("lora", LoraMixin(args.eva_args['num_layers'], args.lora_rank, lora_alpha=args.lora_alpha, layer_range=args.layer_range), reinit=True)
    elif args.use_qlora:
        model.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, lora_alpha=args.lora_alpha, layer_range=args.layer_range, qlora=True), reinit=True)

    if args.use_qlora and torch.cuda.is_available():
        model = model.to('cuda')
    from utils.utils import llama2_tokenizer
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(args.eva_args["image_size"][0])
    text_processor = llama2_text_processor(tokenizer, args.max_length, args.image_length)
    lr_scaling_target_names = [
        "mixins.eva.vit_model.transformer.mv_adapters.0.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.0.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.0.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.0.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.1.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.1.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.1.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.1.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.2.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.2.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.2.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.2.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.3.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.3.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.3.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.3.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.4.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.4.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.4.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.4.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.5.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.5.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.5.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.5.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.6.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.6.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.6.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.6.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.7.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.7.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.7.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.7.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.8.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.8.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.8.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.8.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.9.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.9.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.9.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.9.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.10.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.10.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.10.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.10.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.11.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.11.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.11.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.11.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.12.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.12.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.12.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.12.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.13.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.13.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.13.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.13.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.14.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.14.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.14.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.14.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.15.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.15.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.15.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.15.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.16.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.16.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.16.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.16.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.17.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.17.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.17.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.17.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.18.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.18.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.18.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.18.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.19.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.19.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.19.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.19.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.20.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.20.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.20.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.20.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.21.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.21.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.21.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.21.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.22.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.22.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.22.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.22.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.23.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.23.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.23.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.23.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.24.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.24.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.24.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.24.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.25.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.25.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.25.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.25.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.26.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.26.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.26.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.26.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.27.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.27.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.27.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.27.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.28.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.28.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.28.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.28.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.29.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.29.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.29.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.29.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.30.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.30.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.30.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.30.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.31.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.31.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.31.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.31.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.32.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.32.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.32.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.32.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.33.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.33.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.33.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.33.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.34.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.34.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.34.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.34.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.35.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.35.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.35.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.35.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.36.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.36.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.36.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.36.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.37.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.37.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.37.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.37.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.38.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.38.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.38.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.38.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.39.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.39.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.39.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.39.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.40.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.40.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.40.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.40.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.41.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.41.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.41.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.41.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.42.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.42.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.42.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.42.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.43.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.43.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.43.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.43.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.44.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.44.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.44.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.44.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.45.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.45.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.45.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.45.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.46.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.46.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.46.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.46.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.47.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.47.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.47.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.47.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.48.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.48.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.48.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.48.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.49.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.49.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.49.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.49.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.50.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.50.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.50.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.50.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.51.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.51.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.51.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.51.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.52.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.52.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.52.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.52.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.53.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.53.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.53.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.53.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.54.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.54.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.54.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.54.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.55.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.55.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.55.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.55.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.56.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.56.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.56.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.56.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.57.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.57.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.57.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.57.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.58.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.58.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.58.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.58.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.59.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.59.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.59.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.59.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.60.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.60.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.60.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.60.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.61.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.61.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.61.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.61.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.62.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.62.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.mv_adapters.62.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.mv_adapters.62.fusion_mlp.3.bias",

        "mixins.eva.mv_adapter.fusion_mlp.0.weight",
        "mixins.eva.mv_adapter.fusion_mlp.0.bias",
        "mixins.eva.mv_adapter.fusion_mlp.3.weight",
        "mixins.eva.mv_adapter.fusion_mlp.3.bias",

        "semantic_head.fc1.weight",
        "semantic_head.fc1.bias",
        "semantic_head.fc2.weight",
        "semantic_head.fc2.bias",

        "xyz_head.fc1.weight",
        "xyz_head.fc1.bias",
        "xyz_head.fc2.weight",
        "xyz_head.fc2.bias",

        "range_head.fc1.weight",
        "range_head.fc1.bias",
        "range_head.fc2.weight",
        "range_head.fc2.bias",

        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.bias",
        "mixins.eva.xyz_encoder.transformer.word_embeddings.weight",
        "mixins.eva.xyz_encoder.transformer.position_embeddings.weight",
        "mixins.eva.xyz_encoder.transformer.layers.0.input_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.0.input_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.0.attention.query_key_value.weight",
        "mixins.eva.xyz_encoder.transformer.layers.0.attention.query_key_value.bias",
        "mixins.eva.xyz_encoder.transformer.layers.0.attention.dense.weight",
        "mixins.eva.xyz_encoder.transformer.layers.0.attention.dense.bias",
        "mixins.eva.xyz_encoder.transformer.layers.0.post_attention_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.0.post_attention_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.0.mlp.dense_h_to_4h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.0.mlp.dense_h_to_4h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.0.mlp.dense_4h_to_h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.0.mlp.dense_4h_to_h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.1.input_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.1.input_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.1.attention.query_key_value.weight",
        "mixins.eva.xyz_encoder.transformer.layers.1.attention.query_key_value.bias",
        "mixins.eva.xyz_encoder.transformer.layers.1.attention.dense.weight",
        "mixins.eva.xyz_encoder.transformer.layers.1.attention.dense.bias",
        "mixins.eva.xyz_encoder.transformer.layers.1.post_attention_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.1.post_attention_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.1.mlp.dense_h_to_4h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.1.mlp.dense_h_to_4h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.1.mlp.dense_4h_to_h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.1.mlp.dense_4h_to_h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.2.input_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.2.input_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.2.attention.query_key_value.weight",
        "mixins.eva.xyz_encoder.transformer.layers.2.attention.query_key_value.bias",
        "mixins.eva.xyz_encoder.transformer.layers.2.attention.dense.weight",
        "mixins.eva.xyz_encoder.transformer.layers.2.attention.dense.bias",
        "mixins.eva.xyz_encoder.transformer.layers.2.post_attention_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.2.post_attention_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.2.mlp.dense_h_to_4h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.2.mlp.dense_h_to_4h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.2.mlp.dense_4h_to_h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.2.mlp.dense_4h_to_h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.3.input_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.3.input_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.3.attention.query_key_value.weight",
        "mixins.eva.xyz_encoder.transformer.layers.3.attention.query_key_value.bias",
        "mixins.eva.xyz_encoder.transformer.layers.3.attention.dense.weight",
        "mixins.eva.xyz_encoder.transformer.layers.3.attention.dense.bias",
        "mixins.eva.xyz_encoder.transformer.layers.3.post_attention_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.3.post_attention_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.3.mlp.dense_h_to_4h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.3.mlp.dense_h_to_4h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.3.mlp.dense_4h_to_h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.3.mlp.dense_4h_to_h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.4.input_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.4.input_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.4.attention.query_key_value.weight",
        "mixins.eva.xyz_encoder.transformer.layers.4.attention.query_key_value.bias",
        "mixins.eva.xyz_encoder.transformer.layers.4.attention.dense.weight",
        "mixins.eva.xyz_encoder.transformer.layers.4.attention.dense.bias",
        "mixins.eva.xyz_encoder.transformer.layers.4.post_attention_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.4.post_attention_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.4.mlp.dense_h_to_4h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.4.mlp.dense_h_to_4h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.4.mlp.dense_4h_to_h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.4.mlp.dense_4h_to_h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.5.input_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.5.input_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.5.attention.query_key_value.weight",
        "mixins.eva.xyz_encoder.transformer.layers.5.attention.query_key_value.bias",
        "mixins.eva.xyz_encoder.transformer.layers.5.attention.dense.weight",
        "mixins.eva.xyz_encoder.transformer.layers.5.attention.dense.bias",
        "mixins.eva.xyz_encoder.transformer.layers.5.post_attention_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.5.post_attention_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.5.mlp.dense_h_to_4h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.5.mlp.dense_h_to_4h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.5.mlp.dense_4h_to_h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.5.mlp.dense_4h_to_h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.6.input_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.6.input_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.6.attention.query_key_value.weight",
        "mixins.eva.xyz_encoder.transformer.layers.6.attention.query_key_value.bias",
        "mixins.eva.xyz_encoder.transformer.layers.6.attention.dense.weight",
        "mixins.eva.xyz_encoder.transformer.layers.6.attention.dense.bias",
        "mixins.eva.xyz_encoder.transformer.layers.6.post_attention_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.6.post_attention_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.6.mlp.dense_h_to_4h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.6.mlp.dense_h_to_4h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.6.mlp.dense_4h_to_h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.6.mlp.dense_4h_to_h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.7.input_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.7.input_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.7.attention.query_key_value.weight",
        "mixins.eva.xyz_encoder.transformer.layers.7.attention.query_key_value.bias",
        "mixins.eva.xyz_encoder.transformer.layers.7.attention.dense.weight",
        "mixins.eva.xyz_encoder.transformer.layers.7.attention.dense.bias",
        "mixins.eva.xyz_encoder.transformer.layers.7.post_attention_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.7.post_attention_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.7.mlp.dense_h_to_4h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.7.mlp.dense_h_to_4h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.7.mlp.dense_4h_to_h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.7.mlp.dense_4h_to_h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.8.input_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.8.input_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.8.attention.query_key_value.weight",
        "mixins.eva.xyz_encoder.transformer.layers.8.attention.query_key_value.bias",
        "mixins.eva.xyz_encoder.transformer.layers.8.attention.dense.weight",
        "mixins.eva.xyz_encoder.transformer.layers.8.attention.dense.bias",
        "mixins.eva.xyz_encoder.transformer.layers.8.post_attention_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.8.post_attention_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.8.mlp.dense_h_to_4h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.8.mlp.dense_h_to_4h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.8.mlp.dense_4h_to_h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.8.mlp.dense_4h_to_h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.9.input_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.9.input_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.9.attention.query_key_value.weight",
        "mixins.eva.xyz_encoder.transformer.layers.9.attention.query_key_value.bias",
        "mixins.eva.xyz_encoder.transformer.layers.9.attention.dense.weight",
        "mixins.eva.xyz_encoder.transformer.layers.9.attention.dense.bias",
        "mixins.eva.xyz_encoder.transformer.layers.9.post_attention_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.9.post_attention_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.9.mlp.dense_h_to_4h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.9.mlp.dense_h_to_4h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.9.mlp.dense_4h_to_h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.9.mlp.dense_4h_to_h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.10.input_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.10.input_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.10.attention.query_key_value.weight",
        "mixins.eva.xyz_encoder.transformer.layers.10.attention.query_key_value.bias",
        "mixins.eva.xyz_encoder.transformer.layers.10.attention.dense.weight",
        "mixins.eva.xyz_encoder.transformer.layers.10.attention.dense.bias",
        "mixins.eva.xyz_encoder.transformer.layers.10.post_attention_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.10.post_attention_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.10.mlp.dense_h_to_4h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.10.mlp.dense_h_to_4h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.10.mlp.dense_4h_to_h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.10.mlp.dense_4h_to_h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.11.input_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.11.input_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.11.attention.query_key_value.weight",
        "mixins.eva.xyz_encoder.transformer.layers.11.attention.query_key_value.bias",
        "mixins.eva.xyz_encoder.transformer.layers.11.attention.dense.weight",
        "mixins.eva.xyz_encoder.transformer.layers.11.attention.dense.bias",
        "mixins.eva.xyz_encoder.transformer.layers.11.post_attention_layernorm.weight",
        "mixins.eva.xyz_encoder.transformer.layers.11.post_attention_layernorm.bias",
        "mixins.eva.xyz_encoder.transformer.layers.11.mlp.dense_h_to_4h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.11.mlp.dense_h_to_4h.bias",
        "mixins.eva.xyz_encoder.transformer.layers.11.mlp.dense_4h_to_h.weight",
        "mixins.eva.xyz_encoder.transformer.layers.11.mlp.dense_4h_to_h.bias",

        "mixins.eva.xyz_adapter.weight",
        "mixins.eva.xyz_adapter.bias",

        "mixins.eva.xyz_adapter.fusion_mlp.0.weight",
        "mixins.eva.xyz_adapter.fusion_mlp.0.bias",
        "mixins.eva.xyz_adapter.fusion_mlp.3.weight",
        "mixins.eva.xyz_adapter.fusion_mlp.3.bias",

        "mixins.eva.vit_model.transformer.xyz_adapters.0.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.0.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.0.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.0.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.1.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.1.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.1.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.1.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.2.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.2.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.2.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.2.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.3.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.3.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.3.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.3.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.4.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.4.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.4.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.4.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.5.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.5.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.5.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.5.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.6.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.6.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.6.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.6.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.7.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.7.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.7.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.7.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.8.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.8.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.8.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.8.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.9.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.9.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.9.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.9.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.10.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.10.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.10.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.10.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.11.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.11.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.11.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.11.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.12.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.12.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.12.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.12.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.13.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.13.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.13.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.13.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.14.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.14.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.14.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.14.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.15.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.15.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.15.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.15.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.16.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.16.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.16.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.16.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.17.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.17.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.17.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.17.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.18.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.18.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.18.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.18.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.19.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.19.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.19.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.19.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.20.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.20.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.20.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.20.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.21.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.21.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.21.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.21.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.22.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.22.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.22.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.22.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.23.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.23.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.23.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.23.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.24.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.24.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.24.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.24.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.25.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.25.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.25.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.25.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.26.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.26.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.26.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.26.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.27.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.27.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.27.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.27.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.28.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.28.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.28.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.28.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.29.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.29.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.29.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.29.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.30.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.30.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.30.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.30.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.31.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.31.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.31.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.31.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.32.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.32.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.32.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.32.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.33.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.33.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.33.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.33.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.34.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.34.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.34.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.34.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.35.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.35.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.35.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.35.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.36.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.36.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.36.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.36.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.37.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.37.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.37.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.37.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.38.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.38.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.38.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.38.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.39.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.39.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.39.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.39.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.40.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.40.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.40.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.40.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.41.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.41.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.41.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.41.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.42.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.42.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.42.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.42.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.43.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.43.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.43.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.43.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.44.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.44.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.44.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.44.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.45.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.45.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.45.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.45.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.46.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.46.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.46.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.46.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.47.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.47.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.47.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.47.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.48.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.48.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.48.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.48.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.49.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.49.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.49.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.49.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.50.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.50.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.50.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.50.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.51.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.51.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.51.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.51.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.52.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.52.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.52.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.52.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.53.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.53.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.53.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.53.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.54.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.54.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.54.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.54.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.55.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.55.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.55.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.55.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.56.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.56.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.56.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.56.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.57.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.57.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.57.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.57.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.58.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.58.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.58.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.58.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.59.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.59.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.59.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.59.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.60.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.60.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.60.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.60.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.61.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.61.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.61.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.61.fusion_mlp.3.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.62.fusion_mlp.0.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.62.fusion_mlp.0.bias",
        "mixins.eva.vit_model.transformer.xyz_adapters.62.fusion_mlp.3.weight",
        "mixins.eva.vit_model.transformer.xyz_adapters.62.fusion_mlp.3.bias",

        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.0.conv1.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.0.conv1.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.0.conv2.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.0.conv2.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.0.bn1.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.0.bn1.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.0.conv3.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.0.conv3.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.0.bn2.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.0.bn2.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.1.conv1.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.1.conv1.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.1.conv2.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.1.conv2.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.1.bn1.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.1.bn1.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.1.conv3.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.1.conv3.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.1.bn2.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.1.bn2.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.2.conv1.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.2.conv1.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.2.conv2.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.2.conv2.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.2.bn1.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.2.bn1.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.2.conv3.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.2.conv3.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.2.bn2.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.2.bn2.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.conv1.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.conv1.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.conv2.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.conv2.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.bn1.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.bn1.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.conv3.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.conv3.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.bn2.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.bn2.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.conv4.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.conv4.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.bn3.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.bn3.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.conv5.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.conv5.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.bn4.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.conv_block.3.bn4.bias",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.proj_block.1.weight",
        "mixins.eva.xyz_encoder.mixins.patch_embedding.proj.proj_block.1.bias",

        "mixins.eva.xyz_encoder.embeddings.cls_token",
        "mixins.eva.xyz_encoder.embeddings.mask_token",
        "mixins.eva.xyz_encoder.embeddings.register_tokens",
        "mixins.eva.xyz_encoder.embeddings.patch_embeddings.weight",
        "mixins.eva.xyz_encoder.embeddings.patch_embeddings.bias",
        "mixins.eva.xyz_encoder.layer.0.norm1.weight",
        "mixins.eva.xyz_encoder.layer.0.norm1.bias",
        "mixins.eva.xyz_encoder.layer.0.attention.k_proj.weight",
        "mixins.eva.xyz_encoder.layer.0.attention.v_proj.weight",
        "mixins.eva.xyz_encoder.layer.0.attention.v_proj.bias",
        "mixins.eva.xyz_encoder.layer.0.attention.q_proj.weight",
        "mixins.eva.xyz_encoder.layer.0.attention.q_proj.bias",
        "mixins.eva.xyz_encoder.layer.0.attention.o_proj.weight",
        "mixins.eva.xyz_encoder.layer.0.attention.o_proj.bias",
        "mixins.eva.xyz_encoder.layer.0.layer_scale1.lambda1",
        "mixins.eva.xyz_encoder.layer.0.norm2.weight",
        "mixins.eva.xyz_encoder.layer.0.norm2.bias",
        "mixins.eva.xyz_encoder.layer.0.mlp.gate_proj.weight",
        "mixins.eva.xyz_encoder.layer.0.mlp.gate_proj.bias",
        "mixins.eva.xyz_encoder.layer.0.mlp.up_proj.weight",
        "mixins.eva.xyz_encoder.layer.0.mlp.up_proj.bias",
        "mixins.eva.xyz_encoder.layer.0.mlp.down_proj.weight",
        "mixins.eva.xyz_encoder.layer.0.mlp.down_proj.bias",
        "mixins.eva.xyz_encoder.layer.0.layer_scale2.lambda1",
        "mixins.eva.xyz_encoder.layer.1.norm1.weight",
        "mixins.eva.xyz_encoder.layer.1.norm1.bias",
        "mixins.eva.xyz_encoder.layer.1.attention.k_proj.weight",
        "mixins.eva.xyz_encoder.layer.1.attention.v_proj.weight",
        "mixins.eva.xyz_encoder.layer.1.attention.v_proj.bias",
        "mixins.eva.xyz_encoder.layer.1.attention.q_proj.weight",
        "mixins.eva.xyz_encoder.layer.1.attention.q_proj.bias",
        "mixins.eva.xyz_encoder.layer.1.attention.o_proj.weight",
        "mixins.eva.xyz_encoder.layer.1.attention.o_proj.bias",
        "mixins.eva.xyz_encoder.layer.1.layer_scale1.lambda1",
        "mixins.eva.xyz_encoder.layer.1.norm2.weight",
        "mixins.eva.xyz_encoder.layer.1.norm2.bias",
        "mixins.eva.xyz_encoder.layer.1.mlp.gate_proj.weight",
        "mixins.eva.xyz_encoder.layer.1.mlp.gate_proj.bias",
        "mixins.eva.xyz_encoder.layer.1.mlp.up_proj.weight",
        "mixins.eva.xyz_encoder.layer.1.mlp.up_proj.bias",
        "mixins.eva.xyz_encoder.layer.1.mlp.down_proj.weight",
        "mixins.eva.xyz_encoder.layer.1.mlp.down_proj.bias",
        "mixins.eva.xyz_encoder.layer.1.layer_scale2.lambda1",
        "mixins.eva.xyz_encoder.layer.2.norm1.weight",
        "mixins.eva.xyz_encoder.layer.2.norm1.bias",
        "mixins.eva.xyz_encoder.layer.2.attention.k_proj.weight",
        "mixins.eva.xyz_encoder.layer.2.attention.v_proj.weight",
        "mixins.eva.xyz_encoder.layer.2.attention.v_proj.bias",
        "mixins.eva.xyz_encoder.layer.2.attention.q_proj.weight",
        "mixins.eva.xyz_encoder.layer.2.attention.q_proj.bias",
        "mixins.eva.xyz_encoder.layer.2.attention.o_proj.weight",
        "mixins.eva.xyz_encoder.layer.2.attention.o_proj.bias",
        "mixins.eva.xyz_encoder.layer.2.layer_scale1.lambda1",
        "mixins.eva.xyz_encoder.layer.2.norm2.weight",
        "mixins.eva.xyz_encoder.layer.2.norm2.bias",
        "mixins.eva.xyz_encoder.layer.2.mlp.gate_proj.weight",
        "mixins.eva.xyz_encoder.layer.2.mlp.gate_proj.bias",
        "mixins.eva.xyz_encoder.layer.2.mlp.up_proj.weight",
        "mixins.eva.xyz_encoder.layer.2.mlp.up_proj.bias",
        "mixins.eva.xyz_encoder.layer.2.mlp.down_proj.weight",
        "mixins.eva.xyz_encoder.layer.2.mlp.down_proj.bias",
        "mixins.eva.xyz_encoder.layer.2.layer_scale2.lambda1",
        "mixins.eva.xyz_encoder.layer.3.norm1.weight",
        "mixins.eva.xyz_encoder.layer.3.norm1.bias",
        "mixins.eva.xyz_encoder.layer.3.attention.k_proj.weight",
        "mixins.eva.xyz_encoder.layer.3.attention.v_proj.weight",
        "mixins.eva.xyz_encoder.layer.3.attention.v_proj.bias",
        "mixins.eva.xyz_encoder.layer.3.attention.q_proj.weight",
        "mixins.eva.xyz_encoder.layer.3.attention.q_proj.bias",
        "mixins.eva.xyz_encoder.layer.3.attention.o_proj.weight",
        "mixins.eva.xyz_encoder.layer.3.attention.o_proj.bias",
        "mixins.eva.xyz_encoder.layer.3.layer_scale1.lambda1",
        "mixins.eva.xyz_encoder.layer.3.norm2.weight",
        "mixins.eva.xyz_encoder.layer.3.norm2.bias",
        "mixins.eva.xyz_encoder.layer.3.mlp.gate_proj.weight",
        "mixins.eva.xyz_encoder.layer.3.mlp.gate_proj.bias",
        "mixins.eva.xyz_encoder.layer.3.mlp.up_proj.weight",
        "mixins.eva.xyz_encoder.layer.3.mlp.up_proj.bias",
        "mixins.eva.xyz_encoder.layer.3.mlp.down_proj.weight",
        "mixins.eva.xyz_encoder.layer.3.mlp.down_proj.bias",
        "mixins.eva.xyz_encoder.layer.3.layer_scale2.lambda1",
        "mixins.eva.xyz_encoder.layer.4.norm1.weight",
        "mixins.eva.xyz_encoder.layer.4.norm1.bias",
        "mixins.eva.xyz_encoder.layer.4.attention.k_proj.weight",
        "mixins.eva.xyz_encoder.layer.4.attention.v_proj.weight",
        "mixins.eva.xyz_encoder.layer.4.attention.v_proj.bias",
        "mixins.eva.xyz_encoder.layer.4.attention.q_proj.weight",
        "mixins.eva.xyz_encoder.layer.4.attention.q_proj.bias",
        "mixins.eva.xyz_encoder.layer.4.attention.o_proj.weight",
        "mixins.eva.xyz_encoder.layer.4.attention.o_proj.bias",
        "mixins.eva.xyz_encoder.layer.4.layer_scale1.lambda1",
        "mixins.eva.xyz_encoder.layer.4.norm2.weight",
        "mixins.eva.xyz_encoder.layer.4.norm2.bias",
        "mixins.eva.xyz_encoder.layer.4.mlp.gate_proj.weight",
        "mixins.eva.xyz_encoder.layer.4.mlp.gate_proj.bias",
        "mixins.eva.xyz_encoder.layer.4.mlp.up_proj.weight",
        "mixins.eva.xyz_encoder.layer.4.mlp.up_proj.bias",
        "mixins.eva.xyz_encoder.layer.4.mlp.down_proj.weight",
        "mixins.eva.xyz_encoder.layer.4.mlp.down_proj.bias",
        "mixins.eva.xyz_encoder.layer.4.layer_scale2.lambda1",
        "mixins.eva.xyz_encoder.layer.5.norm1.weight",
        "mixins.eva.xyz_encoder.layer.5.norm1.bias",
        "mixins.eva.xyz_encoder.layer.5.attention.k_proj.weight",
        "mixins.eva.xyz_encoder.layer.5.attention.v_proj.weight",
        "mixins.eva.xyz_encoder.layer.5.attention.v_proj.bias",
        "mixins.eva.xyz_encoder.layer.5.attention.q_proj.weight",
        "mixins.eva.xyz_encoder.layer.5.attention.q_proj.bias",
        "mixins.eva.xyz_encoder.layer.5.attention.o_proj.weight",
        "mixins.eva.xyz_encoder.layer.5.attention.o_proj.bias",
        "mixins.eva.xyz_encoder.layer.5.layer_scale1.lambda1",
        "mixins.eva.xyz_encoder.layer.5.norm2.weight",
        "mixins.eva.xyz_encoder.layer.5.norm2.bias",
        "mixins.eva.xyz_encoder.layer.5.mlp.gate_proj.weight",
        "mixins.eva.xyz_encoder.layer.5.mlp.gate_proj.bias",
        "mixins.eva.xyz_encoder.layer.5.mlp.up_proj.weight",
        "mixins.eva.xyz_encoder.layer.5.mlp.up_proj.bias",
        "mixins.eva.xyz_encoder.layer.5.mlp.down_proj.weight",
        "mixins.eva.xyz_encoder.layer.5.mlp.down_proj.bias",
        "mixins.eva.xyz_encoder.layer.5.layer_scale2.lambda1",
        "mixins.eva.xyz_encoder.layer.6.norm1.weight",
        "mixins.eva.xyz_encoder.layer.6.norm1.bias",
        "mixins.eva.xyz_encoder.layer.6.attention.k_proj.weight",
        "mixins.eva.xyz_encoder.layer.6.attention.v_proj.weight",
        "mixins.eva.xyz_encoder.layer.6.attention.v_proj.bias",
        "mixins.eva.xyz_encoder.layer.6.attention.q_proj.weight",
        "mixins.eva.xyz_encoder.layer.6.attention.q_proj.bias",
        "mixins.eva.xyz_encoder.layer.6.attention.o_proj.weight",
        "mixins.eva.xyz_encoder.layer.6.attention.o_proj.bias",
        "mixins.eva.xyz_encoder.layer.6.layer_scale1.lambda1",
        "mixins.eva.xyz_encoder.layer.6.norm2.weight",
        "mixins.eva.xyz_encoder.layer.6.norm2.bias",
        "mixins.eva.xyz_encoder.layer.6.mlp.gate_proj.weight",
        "mixins.eva.xyz_encoder.layer.6.mlp.gate_proj.bias",
        "mixins.eva.xyz_encoder.layer.6.mlp.up_proj.weight",
        "mixins.eva.xyz_encoder.layer.6.mlp.up_proj.bias",
        "mixins.eva.xyz_encoder.layer.6.mlp.down_proj.weight",
        "mixins.eva.xyz_encoder.layer.6.mlp.down_proj.bias",
        "mixins.eva.xyz_encoder.layer.6.layer_scale2.lambda1",
        "mixins.eva.xyz_encoder.layer.7.norm1.weight",
        "mixins.eva.xyz_encoder.layer.7.norm1.bias",
        "mixins.eva.xyz_encoder.layer.7.attention.k_proj.weight",
        "mixins.eva.xyz_encoder.layer.7.attention.v_proj.weight",
        "mixins.eva.xyz_encoder.layer.7.attention.v_proj.bias",
        "mixins.eva.xyz_encoder.layer.7.attention.q_proj.weight",
        "mixins.eva.xyz_encoder.layer.7.attention.q_proj.bias",
        "mixins.eva.xyz_encoder.layer.7.attention.o_proj.weight",
        "mixins.eva.xyz_encoder.layer.7.attention.o_proj.bias",
        "mixins.eva.xyz_encoder.layer.7.layer_scale1.lambda1",
        "mixins.eva.xyz_encoder.layer.7.norm2.weight",
        "mixins.eva.xyz_encoder.layer.7.norm2.bias",
        "mixins.eva.xyz_encoder.layer.7.mlp.gate_proj.weight",
        "mixins.eva.xyz_encoder.layer.7.mlp.gate_proj.bias",
        "mixins.eva.xyz_encoder.layer.7.mlp.up_proj.weight",
        "mixins.eva.xyz_encoder.layer.7.mlp.up_proj.bias",
        "mixins.eva.xyz_encoder.layer.7.mlp.down_proj.weight",
        "mixins.eva.xyz_encoder.layer.7.mlp.down_proj.bias",
        "mixins.eva.xyz_encoder.layer.7.layer_scale2.lambda1",
        "mixins.eva.xyz_encoder.layer.8.norm1.weight",
        "mixins.eva.xyz_encoder.layer.8.norm1.bias",
        "mixins.eva.xyz_encoder.layer.8.attention.k_proj.weight",
        "mixins.eva.xyz_encoder.layer.8.attention.v_proj.weight",
        "mixins.eva.xyz_encoder.layer.8.attention.v_proj.bias",
        "mixins.eva.xyz_encoder.layer.8.attention.q_proj.weight",
        "mixins.eva.xyz_encoder.layer.8.attention.q_proj.bias",
        "mixins.eva.xyz_encoder.layer.8.attention.o_proj.weight",
        "mixins.eva.xyz_encoder.layer.8.attention.o_proj.bias",
        "mixins.eva.xyz_encoder.layer.8.layer_scale1.lambda1",
        "mixins.eva.xyz_encoder.layer.8.norm2.weight",
        "mixins.eva.xyz_encoder.layer.8.norm2.bias",
        "mixins.eva.xyz_encoder.layer.8.mlp.gate_proj.weight",
        "mixins.eva.xyz_encoder.layer.8.mlp.gate_proj.bias",
        "mixins.eva.xyz_encoder.layer.8.mlp.up_proj.weight",
        "mixins.eva.xyz_encoder.layer.8.mlp.up_proj.bias",
        "mixins.eva.xyz_encoder.layer.8.mlp.down_proj.weight",
        "mixins.eva.xyz_encoder.layer.8.mlp.down_proj.bias",
        "mixins.eva.xyz_encoder.layer.8.layer_scale2.lambda1",
        "mixins.eva.xyz_encoder.layer.9.norm1.weight",
        "mixins.eva.xyz_encoder.layer.9.norm1.bias",
        "mixins.eva.xyz_encoder.layer.9.attention.k_proj.weight",
        "mixins.eva.xyz_encoder.layer.9.attention.v_proj.weight",
        "mixins.eva.xyz_encoder.layer.9.attention.v_proj.bias",
        "mixins.eva.xyz_encoder.layer.9.attention.q_proj.weight",
        "mixins.eva.xyz_encoder.layer.9.attention.q_proj.bias",
        "mixins.eva.xyz_encoder.layer.9.attention.o_proj.weight",
        "mixins.eva.xyz_encoder.layer.9.attention.o_proj.bias",
        "mixins.eva.xyz_encoder.layer.9.layer_scale1.lambda1",
        "mixins.eva.xyz_encoder.layer.9.norm2.weight",
        "mixins.eva.xyz_encoder.layer.9.norm2.bias",
        "mixins.eva.xyz_encoder.layer.9.mlp.gate_proj.weight",
        "mixins.eva.xyz_encoder.layer.9.mlp.gate_proj.bias",
        "mixins.eva.xyz_encoder.layer.9.mlp.up_proj.weight",
        "mixins.eva.xyz_encoder.layer.9.mlp.up_proj.bias",
        "mixins.eva.xyz_encoder.layer.9.mlp.down_proj.weight",
        "mixins.eva.xyz_encoder.layer.9.mlp.down_proj.bias",
        "mixins.eva.xyz_encoder.layer.9.layer_scale2.lambda1",
        "mixins.eva.xyz_encoder.layer.10.norm1.weight",
        "mixins.eva.xyz_encoder.layer.10.norm1.bias",
        "mixins.eva.xyz_encoder.layer.10.attention.k_proj.weight",
        "mixins.eva.xyz_encoder.layer.10.attention.v_proj.weight",
        "mixins.eva.xyz_encoder.layer.10.attention.v_proj.bias",
        "mixins.eva.xyz_encoder.layer.10.attention.q_proj.weight",
        "mixins.eva.xyz_encoder.layer.10.attention.q_proj.bias",
        "mixins.eva.xyz_encoder.layer.10.attention.o_proj.weight",
        "mixins.eva.xyz_encoder.layer.10.attention.o_proj.bias",
        "mixins.eva.xyz_encoder.layer.10.layer_scale1.lambda1",
        "mixins.eva.xyz_encoder.layer.10.norm2.weight",
        "mixins.eva.xyz_encoder.layer.10.norm2.bias",
        "mixins.eva.xyz_encoder.layer.10.mlp.gate_proj.weight",
        "mixins.eva.xyz_encoder.layer.10.mlp.gate_proj.bias",
        "mixins.eva.xyz_encoder.layer.10.mlp.up_proj.weight",
        "mixins.eva.xyz_encoder.layer.10.mlp.up_proj.bias",
        "mixins.eva.xyz_encoder.layer.10.mlp.down_proj.weight",
        "mixins.eva.xyz_encoder.layer.10.mlp.down_proj.bias",
        "mixins.eva.xyz_encoder.layer.10.layer_scale2.lambda1",
        "mixins.eva.xyz_encoder.layer.11.norm1.weight",
        "mixins.eva.xyz_encoder.layer.11.norm1.bias",
        "mixins.eva.xyz_encoder.layer.11.attention.k_proj.weight",
        "mixins.eva.xyz_encoder.layer.11.attention.v_proj.weight",
        "mixins.eva.xyz_encoder.layer.11.attention.v_proj.bias",
        "mixins.eva.xyz_encoder.layer.11.attention.q_proj.weight",
        "mixins.eva.xyz_encoder.layer.11.attention.q_proj.bias",
        "mixins.eva.xyz_encoder.layer.11.attention.o_proj.weight",
        "mixins.eva.xyz_encoder.layer.11.attention.o_proj.bias",
        "mixins.eva.xyz_encoder.layer.11.layer_scale1.lambda1",
        "mixins.eva.xyz_encoder.layer.11.norm2.weight",
        "mixins.eva.xyz_encoder.layer.11.norm2.bias",
        "mixins.eva.xyz_encoder.layer.11.mlp.gate_proj.weight",
        "mixins.eva.xyz_encoder.layer.11.mlp.gate_proj.bias",
        "mixins.eva.xyz_encoder.layer.11.mlp.up_proj.weight",
        "mixins.eva.xyz_encoder.layer.11.mlp.up_proj.bias",
        "mixins.eva.xyz_encoder.layer.11.mlp.down_proj.weight",
        "mixins.eva.xyz_encoder.layer.11.mlp.down_proj.bias",
        "mixins.eva.xyz_encoder.layer.11.layer_scale2.lambda1",
        "mixins.eva.xyz_encoder.norm.weight",
        "mixins.eva.xyz_encoder.norm.bias",

        "depth_diff_head.fusion_layer.0.weight",
        "depth_diff_head.fusion_layer.0.bias",
        "depth_diff_head.fusion_layer.1.weight",
        "depth_diff_head.fusion_layer.1.bias",
        "depth_diff_head.fusion_layer.3.weight",
        "depth_diff_head.fusion_layer.3.bias",

        "intermediate_depth_diff_head.fusion_layer.0.weight",
        "intermediate_depth_diff_head.fusion_layer.0.bias",
        "intermediate_depth_diff_head.fusion_layer.1.weight",
        "intermediate_depth_diff_head.fusion_layer.1.bias",
        "intermediate_depth_diff_head.fusion_layer.3.weight",
        "intermediate_depth_diff_head.fusion_layer.3.bias",

        "language_depth_diff_head.fusion_layer.0.weight",
        "language_depth_diff_head.fusion_layer.0.bias",
        "language_depth_diff_head.fusion_layer.1.weight",
        "language_depth_diff_head.fusion_layer.1.bias",
        "language_depth_diff_head.fusion_layer.3.weight",
        "language_depth_diff_head.fusion_layer.3.bias",

        "x_diff_head.fusion_layer.0.weight",
        "x_diff_head.fusion_layer.0.bias",
        "x_diff_head.fusion_layer.1.weight",
        "x_diff_head.fusion_layer.1.bias",
        "x_diff_head.fusion_layer.3.weight",
        "x_diff_head.fusion_layer.3.bias",
        "y_diff_head.fusion_layer.0.weight",
        "y_diff_head.fusion_layer.0.bias",
        "y_diff_head.fusion_layer.1.weight",
        "y_diff_head.fusion_layer.1.bias",
        "y_diff_head.fusion_layer.3.weight",
        "y_diff_head.fusion_layer.3.bias",
        "z_diff_head.fusion_layer.0.weight",
        "z_diff_head.fusion_layer.0.bias",
        "z_diff_head.fusion_layer.1.weight",
        "z_diff_head.fusion_layer.1.bias",
        "z_diff_head.fusion_layer.3.weight",
        "z_diff_head.fusion_layer.3.bias",

        "intermediate_x_diff_head.fusion_layer.0.weight",
        "intermediate_x_diff_head.fusion_layer.0.bias",
        "intermediate_x_diff_head.fusion_layer.1.weight",
        "intermediate_x_diff_head.fusion_layer.1.bias",
        "intermediate_x_diff_head.fusion_layer.3.weight",
        "intermediate_x_diff_head.fusion_layer.3.bias",
        "intermediate_y_diff_head.fusion_layer.0.weight",
        "intermediate_y_diff_head.fusion_layer.0.bias",
        "intermediate_y_diff_head.fusion_layer.1.weight",
        "intermediate_y_diff_head.fusion_layer.1.bias",
        "intermediate_y_diff_head.fusion_layer.3.weight",
        "intermediate_y_diff_head.fusion_layer.3.bias",
        "intermediate_z_diff_head.fusion_layer.0.weight",
        "intermediate_z_diff_head.fusion_layer.0.bias",
        "intermediate_z_diff_head.fusion_layer.1.weight",
        "intermediate_z_diff_head.fusion_layer.1.bias",
        "intermediate_z_diff_head.fusion_layer.3.weight",
        "intermediate_z_diff_head.fusion_layer.3.bias",

        "language_x_diff_head.fusion_layer.0.weight",
        "language_x_diff_head.fusion_layer.0.bias",
        "language_x_diff_head.fusion_layer.1.weight",
        "language_x_diff_head.fusion_layer.1.bias",
        "language_x_diff_head.fusion_layer.3.weight",
        "language_x_diff_head.fusion_layer.3.bias",
        "language_y_diff_head.fusion_layer.0.weight",
        "language_y_diff_head.fusion_layer.0.bias",
        "language_y_diff_head.fusion_layer.1.weight",
        "language_y_diff_head.fusion_layer.1.bias",
        "language_y_diff_head.fusion_layer.3.weight",
        "language_y_diff_head.fusion_layer.3.bias",
        "language_z_diff_head.fusion_layer.0.weight",
        "language_z_diff_head.fusion_layer.0.bias",
        "language_z_diff_head.fusion_layer.1.weight",
        "language_z_diff_head.fusion_layer.1.bias",
        "language_z_diff_head.fusion_layer.3.weight",
        "language_z_diff_head.fusion_layer.3.bias",
    ]

    checkpoint_root = args.save
    model = training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=partial(create_dataset_function, image_processor, text_processor), collate_fn=data_collator, forward_step_eval=forward_step_eval, lr_scale=args.lr_scale, lr_scaling_target_names=lr_scaling_target_names)
    if args.use_lora:
        model.get_mixin("lora").merge_lora()
        model.get_mixin("eva").vit_model.get_mixin("lora").merge_lora()
        # args.use_lora = False
        args.save = os.path.join(checkpoint_root, "merged_lora_cogvlm_{}".format(args.experiment_name))
        from sat.training.model_io import save_checkpoint
        save_checkpoint(1, model, None, None, args)
