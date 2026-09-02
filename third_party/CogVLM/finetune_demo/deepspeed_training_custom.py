# coding=utf-8
# Rewrite by Ming Ding, Tsinghua University
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import math
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime
from contextlib import ExitStack

import torch.distributed as dist
import deepspeed
import wandb

from sat.training.learning_rates import AnnealingLR
from sat.training.model_io import load_checkpoint, save_checkpoint

from sat.training.utils import Timers
from sat.training.utils import report_memory
from sat.training.utils import print_args
from sat.training.utils import get_sample_writer
from sat.training.utils import init_wandb_writer

from sat import mpu
from sat.data_utils import make_loaders
from sat.transformer_defaults import NO_WD_MODULES
from sat.helpers import print_rank0, print_all
from sat.model.base_model import get_model

def training_main(args, model_cls, forward_step_function, create_dataset_function, handle_metrics_function=None, init_function=None, collate_fn=None, forward_step_eval=None, lr_scale=10.0, lr_scaling_target_names=None):
    """Main training program."""
    hooks = {
        'forward_step': forward_step_function,
        'init_function': init_function,
        'create_dataset_function': create_dataset_function,
        'handle_metrics': handle_metrics_function,
        'forward_step_eval': forward_step_eval or forward_step_function
    }

    timers = Timers()  # Timer.

    # Experiment Name
    if args.load and args.mode == 'pretrain':  # continue training
        args.experiment_name = os.path.basename(os.path.normpath(args.load))
    else:
        args.experiment_name = args.experiment_name + '-' +datetime.now().strftime("%m-%d-%H-%M")

    # Pytorch distributed. must before seed. ALREADY MOVED TO arguments.py!
    # if isinstance(model_cls, type):
    #     initialize_distributed(args)
    #     set_random_seed(args.seed)  # Random seeds for reproducability.

    # Data stuff.
    train_data, val_data, test_data = make_loaders(args, hooks['create_dataset_function'], collate_fn=collate_fn)
    if args.epochs:
        args.train_iters = len(train_data)
        if args.eval_interval is None:
            args.eval_interval = len(train_data)//args.epochs
        if args.save_interval is None:
            args.save_interval = len(train_data)//args.epochs

    # Build model
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls
        # for given model, make sure all the params are in the correct device, or the sync param will raise error
        correct_device = torch.device(args.device)
        for param in model.parameters():
            if param.device != correct_device:
                param.data = param.data.to(correct_device)
        # register buffer
        for name, buffer in model.named_buffers():
            if buffer.device != correct_device:
                buffer.data = buffer.data.to(correct_device)

    # Config model IO
    if args.load is not None:
        args.iteration = load_checkpoint(model, args)
    else:
        args.iteration = 0
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)
    torch.distributed.barrier()

    # init hook before building deepspeed model and optimizer
    if hooks['init_function'] is not None:
        hooks['init_function'](args, model)

    # training
    iteration = 0
    if args.train_iters > 0:
        # Optimization related things
        model, optimizer = setup_model_untrainable_params_and_optimizer(args, model, lr_scale=lr_scale, lr_scaling_target_names=lr_scaling_target_names)

        # initialize lr scheduler
        lr_scheduler = get_learning_rate_scheduler(optimizer, args.iteration, args)
        assert isinstance(lr_scheduler, AnnealingLR), \
            'must be sat AnnealingLR, or the lr in param_groups will be wrong.'

        summary_writer = None
        if torch.distributed.get_rank() == 0:
            if args.mode == 'pretrain':
                print_rank0('Pretraining or Continuing training the Model...')
            elif args.mode == 'finetune':
                print_rank0('Finetuning Model...')
            print_args(args)
            summary_writer = get_sample_writer(base=args.summary_dir, name=args.experiment_name, iteration=args.iteration)
            if args.wandb:
                init_wandb_writer(args)
        # Resume data loader if necessary.
        if args.resume_dataloader:
            if not args.iterable_dataset:
                if train_data is not None:
                    train_data.batch_sampler.start_iter = args.iteration % len(train_data)
                if val_data is not None:
                    start_iter_val = (args.train_iters // args.save_interval) * args.eval_interval
                    val_data.batch_sampler.start_iter = start_iter_val % len(val_data)
            else:
                print_rank0('Warning: we cannot resume iterable dataloader. skipping...')

        if args.do_train:
            with ExitStack() as stack:
                def save_on_exit(args_, model_, optimizer_, lr_scheduler_):
                    save_checkpoint(args_.iteration, model_, optimizer_, lr_scheduler_, args_)

                # re-sync random seed, or tensor parallel might be broken (dropout, droppath)
                # TODO add rng states for data parallel and wrap drops in main path.
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed(args.seed)
                # ---------

                iteration, skipped = train(model, optimizer,
                    lr_scheduler,
                    train_data,
                    val_data,
                    timers, args, summary_writer=summary_writer,
                    hooks=hooks
                    )

    # final save
    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

    # final testing
    if args.do_test and test_data is not None:
        prefix = 'the end of training for test data'
        test_loss = evaluate_and_print_results(prefix, iter(test_data),
            model, len(test_data) if args.strict_eval else args.eval_iters, args, timers, True, split='test', hooks=hooks)

    return model

def setup_model_untrainable_params_and_optimizer(args, model, config_params=None, lr_scale=10.0, lr_scaling_target_names=None):
    """Setup model and optimizer."""

    if hasattr(model, 'disable_untrainable_params'):
        model.disable_untrainable_params() # mark trainable params
    param_groups = get_optimizer_param_groups(model, lr_scale, lr_scaling_target_names)

    # sync initialized parameters
    # zero3 don't need to sync
    from sat.helpers import check_if_zero3
    if not check_if_zero3(args):
        print_rank0('Syncing initialized parameters...')
        for param_group in param_groups:
            for param in param_group['params']:
                if not param.model_parallel:
                    # We already keep the same random seed for different ranks
                    # However, it is not reliable. Non-model-parallel parameters could be different when initialization.
                    dist.broadcast(
                        param.data,
                        src=0, # group is default group
                    )
                else:
                    dist.broadcast(
                        param.data,
                        src=mpu.get_model_parallel_rank(), # 0 -- mp_size-1
                        group=mpu.get_data_parallel_group() # 1, mp_size + 1, ...
                    )
        print_rank0('Finished syncing initialized parameters.')

    if args.train_data is not None:
        if args.deepspeed:
            from packaging import version
            print_rank0("DeepSpeed is enabled.", level='DEBUG')
            # checking optimizer
            optimizer_name = args.deepspeed_config.get('optimizer',{}).get('type', '')
            if optimizer_name.startswith('sat.'):
                from importlib import import_module
                from functools import partial
                # split and import
                optimizer_callable = getattr(import_module(optimizer_name.rsplit('.', maxsplit=1)[0]), optimizer_name.split('.')[-1])
                optimizer_callable = partial(optimizer_callable, **args.deepspeed_config.get('optimizer', {}).get('params', {}))
                print_rank0(f'Using optimizer {optimizer_name} from sat.')
                del args.deepspeed_config['optimizer']
            else:
                optimizer_callable = None

            model, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=param_groups,
                optimizer=optimizer_callable,
                args=args,
                mpu=mpu,
                dist_init_required=False,
                config_params=args.deepspeed_config
                    if version.parse(deepspeed.version) < version.parse("0.9.0")
                    else None
            )
        else:
            raise ValueError('Currently, we only support training with deepspeed.')
    else:
        optimizer = None

    return model, optimizer


def add_param_by_lr(dic, p, no_weight_decay=False):
    if not hasattr(p, 'lr_scale'):
        dic[None]['params'].append(p)
    else:
        if p.lr_scale not in dic:
            dic[p.lr_scale] = {'params': [], 'lr': p.lr_scale} if not no_weight_decay else {'params': [], 'weight_decay': 0.0, 'lr': p.lr_scale}
        dic[p.lr_scale]['params'].append(p)


def get_params_for_optimization(module, input_lr_scale=10.0, lr_scaling_target_names=None):
    if lr_scaling_target_names is None:
        lr_scaling_target_names = set()

    # 파라미터 그룹을 (lr_scale, has_weight_decay) 튜플을 키로 하여 관리
    # 예: params_groups[(1.0, True)] = {'params': [...], 'weight_decay': 0.01, 'lr': 1.0}
    param_groups = {}

    # 1. 원본 코드 로직 보존: NO_WD_MODULES에 속한 파라미터들의 ID를 미리 저장
    no_wd_param_ids = set()
    print_rank0(f"Modules of type {NO_WD_MODULES} are marked for no_weight_decay.")
    for m in module.modules():
        if isinstance(m, tuple(NO_WD_MODULES)):
            for p in m.parameters():
                no_wd_param_ids.add(id(p))

    # 2. 전체 파라미터를 순회하며 그룹 분류
    for full_name, p in module.named_parameters():
        if not p.requires_grad:
            continue

        # LR 배율 결정
        # lr_scaling_target_names가 비어있으면 lr_scale은 항상 1.0이 됨
        lr_scale = input_lr_scale if full_name in lr_scaling_target_names else 1.0

        # Weight Decay 적용 여부 결정 (원본 코드의 로직과 동일)
        # 조건: 1) bias 이거나, 2) no_weight_decay 속성이 있거나, 3) NO_WD_MODULES 소속일 경우
        is_bias = full_name.endswith('.bias')
        has_no_wd_attr = hasattr(p, 'no_weight_decay') and p.no_weight_decay
        is_in_no_wd_module = id(p) in no_wd_param_ids

        apply_weight_decay = not (is_bias or has_no_wd_attr or is_in_no_wd_module)

        # 해당하는 그룹에 파라미터 추가
        group_key = (lr_scale, apply_weight_decay)
        if group_key not in param_groups:
            param_groups[group_key] = {
                'params': [],
                'lr_scale': lr_scale,  # 옵티마이저에서 이 값을 활용
                'weight_decay': 0.0 if not apply_weight_decay else None # None으로 두면 옵티마이저 기본값 사용
            }
        param_groups[group_key]['params'].append(p)

        # 특정 파라미터 상태 출력 (디버깅용)
        if has_no_wd_attr:
            print_rank0(f"Parameter '{full_name}' has 'no_weight_decay' attr -> No Weight Decay")
        if full_name in lr_scaling_target_names:
            wd_status = "HAS" if apply_weight_decay else "NO"
            print_rank0(f"Parameter '{full_name}' -> {wd_status} Weight Decay, LR_SCALE: {lr_scale}")


    # 3. 최종 옵티마이저 파라미터 리스트 생성
    # 옵티마이저는 'lr' 키를 사용하므로, 'lr_scale'을 'lr'로 변경하여 전달
    ret = []
    for (lr_scale, apply_wd), group in param_groups.items():
        final_group = group.copy()
        final_group['lr'] = final_group.pop('lr_scale')
        if not apply_wd:
             # weight_decay를 0.0으로 명시적으로 설정
            final_group['weight_decay'] = 0.0
        else:
            # weight_decay 키를 제거하여 옵티마이저의 기본값을 따르도록 함
            final_group.pop('weight_decay', None)

        ret.append(final_group)

    return ret


def get_optimizer_param_groups(model, lr_scale=10.0, lr_scaling_target_names=None):
    # Build parameter groups (weight decay and non-decay).
    if hasattr(model, 'module'):
        model = model.module
    param_groups = get_params_for_optimization(model, lr_scale, lr_scaling_target_names)
    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False
    return param_groups


def get_learning_rate_scheduler(optimizer, iteration, args,
                                auto_warmup_steps=100, auto_warmup_rate=0.05):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = max(iteration - auto_warmup_steps, 0)
    if args.mode == 'pretrain' and iteration == 0:
        auto_warmup_steps = 0
    # If init_step <= current_steps <= init_step + auto_warmup_steps,
    # lr = auto_warmup_rate * args.lr.
    # This overrides other rules.
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               decay_ratio=args.lr_decay_ratio,
                               auto_warmup_steps=auto_warmup_steps,
                               auto_warmup_rate=auto_warmup_rate
                               )

    return lr_scheduler


def train(model, optimizer, lr_scheduler,
        train_data, val_data, timers, args,
        summary_writer=None, hooks={}):
    """Train the model."""
    if train_data is not None:
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None
    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0
    total_metrics = defaultdict(float)
    total_metrics_cnt = defaultdict(int)

    # Iterations.
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    while args.iteration < args.train_iters:
        if args.profiling != -1 and args.iteration == args.profiling:
            torch.cuda.cudart().cudaProfilerStart()

        if args.profiling != -1 and args.iteration >= args.profiling:
            torch.cuda.nvtx.range_push("iteration{}".format(args.iteration))
        lm_loss, skipped_iter, metrics = train_step(train_data_iterator,
                                                    model,
                                                    optimizer,
                                                    lr_scheduler,
                                                    args, timers, hooks=hooks)
        skipped_iters += skipped_iter
        if args.profiling != -1 and args.iteration >= args.profiling:
            torch.cuda.nvtx.range_pop()
        args.iteration += 1
        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()
        for name in metrics:
            if not 'eval' in name:
                assert len(metrics[name].shape)==0, 'metrics without eval must be scalar'
                value = metrics[name].data.detach().float().item()
                if value > -99:
                    total_metrics[name] += value
                    total_metrics_cnt[name] += 1

        # Logging.
        if args.iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            # average img & txt loss
            avg_metrics = {}
            for key in total_metrics:
                avg_metrics[key] = total_metrics[key] / total_metrics_cnt[key] # args.log_interval

            elapsed_time = timers('interval time').elapsed()
            report_iteration_metrics(summary_writer, optimizer, learning_rate, avg_lm_loss,
                                     elapsed_time * 1000.0 / args.log_interval, args.iteration, args.train_iters, args,
                                     avg_metrics)
            total_lm_loss = 0.0
            total_metrics = defaultdict(float)
            total_metrics_cnt = defaultdict(int)
            if report_memory_flag:
                report_memory('after {} iterations'.format(args.iteration))
                report_memory_flag = False

            timers.log(['forward', 'backward', 'allreduce', 'optimizer',
                        'batch generator', 'data loader'],
                       normalizer=args.log_interval)
        # Checkpointing
        if args.save and args.save_interval and args.iteration % args.save_interval == 0:
            save_checkpoint(args.iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and args.iteration % args.eval_interval == 0 and args.do_valid:
            if args.strict_eval:
                val_data_iterator = iter(val_data)
                eval_iters = len(val_data)
            else:
                eval_iters = args.eval_iters
            prefix = 'iteration {}'.format(args.iteration)
            evaluate_and_print_results(
                prefix, val_data_iterator, model, eval_iters, args, timers, False, step=args.iteration, split='val', summary_writer=summary_writer, hooks=hooks)

        if args.exit_interval and args.iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print_all('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, args.iteration), flush=True)
            exit()
    if args.profiling != -1:
        torch.cuda.cudart().cudaProfilerStop()

    return args.iteration, skipped_iters


def train_step(data_iterator, model, optimizer, lr_scheduler,
               args, timers, hooks=None, single_step=False, **kwargs):
    """Single training step."""
    if hooks is None:
        hooks = {}
    lm_loss_total, metrics_total, count, metrics_count = 0.0, {}, 0, {}
    forward_step = hooks['forward_step']

    while True:
        profiling_flag = (args.profiling != -1 and args.iteration >= args.profiling)
        # Forward model for one step.
        if profiling_flag:
            torch.cuda.nvtx.range_push("forward")
        timers('forward').start()
        forward_ret = forward_step(data_iterator, model, args, timers, **kwargs)
        if isinstance(forward_ret, tuple):
            lm_loss, metrics = forward_ret
        else:
            lm_loss, metrics = forward_ret, {}
        timers('forward').stop()
        if profiling_flag:
            torch.cuda.nvtx.range_pop()

        # Check nan or inf in forward, preventing it from interfering loss scaler,
        # and all reduce metrics by the way
        if profiling_flag:
            torch.cuda.nvtx.range_push("loss_and_metrics")
        lm_loss_reduced = lm_loss.detach().clone()
        torch.distributed.all_reduce(lm_loss_reduced.data)
        lm_loss_reduced.data = lm_loss_reduced.data / args.world_size

        loss_checker = lm_loss_reduced
        for name in metrics:
            if not 'eval' in name:
                metrics[name] = metrics[name].detach().clone()
                if metrics[name].data.item() == -100:
                    cnt = torch.zeros(1, dtype=torch.int64, device=metrics[name].data.device)
                    metrics[name].data = torch.tensor(0., device=metrics[name].data.device)
                else:
                    cnt = torch.ones(1, dtype=torch.int64, device=metrics[name].data.device)
                torch.distributed.all_reduce(metrics[name].data)
                torch.distributed.all_reduce(cnt)
                if cnt.item() == 0:
                    metrics[name].data = torch.tensor(-100, device=metrics[name].data.device)
                else:
                    metrics[name].data /= cnt.cpu().item() # args.world_size
                loss_checker = loss_checker + metrics[name]
        if loss_checker.isnan().any() or loss_checker.isinf().any():
            print_all('Skipping backward and optimizer step for nan or inf in forwarding metrics/loss!')
            return lm_loss.detach(), 1, metrics

        print_all('backward success!')

        # Accumulate the statistics
        lm_loss_total += lm_loss_reduced
        for name in metrics:
            if name not in metrics_total:
                metrics_total[name] = torch.tensor(0.0, device=metrics[name].data.device)
            if name not in metrics_count:
                metrics_count[name] = 0
            if metrics[name].data.item() != -100:
                metrics_total[name] += metrics[name]
                metrics_count[name] += 1
        count += 1
        if profiling_flag:
            torch.cuda.nvtx.range_pop()

        if profiling_flag:
            torch.cuda.nvtx.range_push("backward")
        # Calculate gradients, reduce across processes, and clip.
        timers('backward').start()
        backward_step(optimizer, model, lm_loss, args, timers)
        timers('backward').stop()
        if profiling_flag:
            torch.cuda.nvtx.range_pop()
        # Update parameters.
        skipped_iter, complete = 0, False
        if profiling_flag:
            torch.cuda.nvtx.range_push("optimizer")
        timers('optimizer').start()
        if args.deepspeed:
            if model.is_gradient_accumulation_boundary():
                model.step()
                complete = True
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()
                else:
                    skipped_iter = 1
            else:
                model.step()
        else:
            raise ValueError('Currently, we only support training with deepspeed.')
        timers('optimizer').stop()
        if profiling_flag:
            torch.cuda.nvtx.range_pop()
        if complete or single_step:
            break
    lm_loss_total /= count
    metrics_total = {key: torch.tensor(-100, device=metrics_total[key].data.device) if metrics_count[key] == 0 else value / metrics_count[key] for key, value in metrics_total.items()}
    return lm_loss_total, skipped_iter, metrics_total


def backward_step(optimizer, model, loss, args, timers):
    """Backward step."""

    # Backward pass.
    if args.deepspeed:
        model.backward(loss)
    else:
        raise ValueError('Currently, we only support training with deepspeed.')

    if args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('allreduce').reset()

    return

def evaluate(data_iterator, model, eval_iters, args, timers, split, verbose=False, has_last=True, hooks={}):
    """Evaluation."""
    forward_step = hooks['forward_step_eval']
    # Turn on evaluation mode which disables dropout.
    model.eval()
    rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    total_lm_loss, metrics_total = 0, {}
    if split=='val':
        last_shape = args.val_last_shape
        drop_number = args.val_drop_number
    else:
        assert split=='test'
        last_shape = args.test_last_shape
        drop_number = args.test_drop_number
    is_scalar = {}
    with torch.no_grad():
        iteration = 0
        while iteration < eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank0('Evaluating iter {}/{}'.format(iteration, eval_iters))
            # Forward evaluation.
            # try:
            lm_loss, metrics = forward_step(data_iterator, model, args, timers)
            '''when contiguous memory optimizations are enabled, the buffers
            allocated by the optimizations are deallocated during backward pass
            in the absence of backward pass the buffers should be reset after each
            forward pass'''
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()
            total_lm_loss += lm_loss.data.detach().float().item()
            is_last = True if iteration == eval_iters and args.strict_eval and len(last_shape)>0 else False
            for name in metrics:
                if name not in metrics_total:
                    metrics_total[name] = []
                is_scalar[name] = True if len(metrics[name].shape)==0 else False
                shape = list(metrics[name].shape)
                if not is_scalar[name] and is_last and metrics[name].shape[0] != last_shape[0]:
                    # pad tensor's first dim to args.batch_size
                    metrics[name] = torch.concat([metrics[name], torch.zeros([last_shape[0]-metrics[name].shape[0]] + shape[1:], dtype=metrics[name].dtype, device=metrics[name].device)])
                if rank==0:
                    metrics_gathered = [torch.zeros_like(metrics[name], dtype=metrics[name].dtype, device=metrics[name].device) for _ in range(args.world_size)]
                else:
                    # metrics_gathered = None
                    metrics_gathered = [torch.zeros_like(metrics[name], dtype=metrics[name].dtype, device=metrics[name].device) for _ in range(args.world_size)]
                # torch.distributed.gather(metrics[name], metrics_gathered, 0)
                torch.distributed.all_gather(metrics_gathered, metrics[name])

                if rank==0:
                    gathered_len = len(metrics_gathered) if not is_last else len(metrics_gathered) - drop_number * args.model_parallel_size
                    for i in range(gathered_len):
                        if is_scalar[name] or not is_last:
                            metrics_total[name].append(metrics_gathered[i].data.cpu())
                        else:
                            metrics_total[name].append(metrics_gathered[i][:last_shape[i]].data.cpu())
    # Move model back to the train mode.
    model.train()

    total_lm_loss /= eval_iters
    # metrics_avg = {key: value / eval_iters for key, value in metrics_total.items()}
    if rank==0:
        for name in metrics_total:
            if is_scalar[name]:
                metrics_total[name] = torch.stack(metrics_total[name], dim=0)
            else:
                metrics_total[name] = torch.concat(metrics_total[name], dim=0)
        if hooks['handle_metrics'] is not None:
            metrics = hooks['handle_metrics'](metrics_total)
        else:
            for name in metrics_total:
                assert is_scalar[name], 'you must return scalar metrics or implement handle_metrics hooks'
            metrics = {key: sum(value.split(1,0))/len(value) for key, value in metrics_total.items()}
    else:
        metrics = None
    return total_lm_loss, metrics

def evaluate_and_print_results(prefix, data_iterator, model, eval_iters,
                            args, timers, has_last, split, verbose=False, step=None, summary_writer=None, hooks={}):
    """Helper function to evaluate and dump results on screen."""
    lm_loss, metrics = evaluate(data_iterator, model, eval_iters, args, timers, split, verbose, has_last, hooks=hooks)
    lm_ppl = math.exp(min(20, lm_loss))
    if torch.distributed.get_rank(group=mpu.get_data_parallel_group())==0:
        report_evaluate_metrics(summary_writer, prefix, lm_loss, lm_ppl, step, args, metrics)
    return lm_loss


def report_iteration_metrics(summary_writer, optimizer, lr, loss, elapsed_time, step, total_step, args, avg_metrics):
    log_string = ' iteration {:8d}/{:8d} |'.format(step, total_step)
    log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time)
    log_string += ' learning rate {:.3E} |'.format(lr)
    log_string += ' total loss {:.6E} |'.format(loss)
    for key in avg_metrics:
        log_string += ' {} {:.6E} |'.format(key, avg_metrics[key])
    if args.fp16:
        log_string += ' loss scale {:.1f} |'.format(
            optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
    log_string += 'speed {:.2f} samples/(min*GPU)'.format(
        (args.gradient_accumulation_steps * args.batch_size / args.model_parallel_size / (elapsed_time / 60000.0)))
    print_rank0(log_string)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/lr', lr, step)
        summary_writer.add_scalar(f'Train/train_loss', loss, step)
        summary_writer.add_scalar(f'Train/elapsed_time', elapsed_time, step)
        for key in avg_metrics:
            summary_writer.add_scalar('Train/'+key, avg_metrics[key], step)
    if args.wandb and torch.distributed.get_rank() == 0:
        log_dict = {
            "Train/lr": lr,
            "Train/train_loss": loss,
            "Train/elapsed_time": elapsed_time
            }
        for key in avg_metrics:
            log_dict["Train/" + key] = avg_metrics[key]
        wandb.log(log_dict, step=step, commit=True)


def report_evaluate_metrics(summary_writer, prefix, loss, ppl, step, args, avg_metrics):
    if step is not None:
        step += 1
    string = ' validation loss at {} | '.format(prefix)
    string += 'loss: {:.6E} | '.format(loss)
    string += 'PPL: {:.6E}'.format(ppl)
    for key in avg_metrics:
        string += ' {} {:.6E} |'.format(key, avg_metrics[key].item())
    length = len(string) + 1
    print_rank0('-' * 100)
    print_rank0('-' * length)
    print_rank0(string)
    print_rank0('-' * length)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/valid_ppl', ppl, step)
        summary_writer.add_scalar(f'Train/valid_loss', loss, step)
        for key in avg_metrics:
            summary_writer.add_scalar('Train/valid_'+key, avg_metrics[key], step)
    if args.wandb and torch.distributed.get_rank() == 0:
        log_dict = {
            "Train/valid_ppl": ppl,
            "Train/valid_loss": loss,
            }
        for key in avg_metrics:
            log_dict["Train/valid_" + key] = avg_metrics[key]
        wandb.log(log_dict, step=step, commit=True)
