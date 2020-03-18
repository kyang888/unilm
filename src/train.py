"""UNILM pretrain and finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
import glob
import math
import json
import argparse
import random
import time
from pathlib import Path
import torch
import numpy as np

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from data_loader import TrainDataLoader, Preprocess4Seq2seq
from tensorboardX import SummaryWriter


import torch.distributed as dist
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None

def _get_checkpont_num(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return 0
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    return sorted(list(both_set), reverse=False)

def main():
    parser = argparse.ArgumentParser()
    # Path parameters
    parser.add_argument("--data_dir", default=None, type=str,required=True,
                        help="The raw data dir.")
    parser.add_argument("--vocab_path", default=None, type=str, required=True,
                        help="bert vocab path")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--model_output_dir",default=None,type=str,required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir", default='',type=str,required=True,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path", default=None, type=str,
                        help="The param init of pretrain or finetune")
    parser.add_argument("--optim_recover_path",default=None,type=str,
                        help="The file of pretraining optimizer.")
    # Data Process Parameters
    parser.add_argument("--max_seq_length",default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                        help="max position embeddings")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    parser.add_argument('--max_len_a', type=int, default=0,
                        help="Truncate_config: maximum length of segment A.")
    parser.add_argument('--max_len_b', type=int, default=0,
                        help="Truncate_config: maximum length of segment B.")
    parser.add_argument('--trunc_seg', default='',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true',
                        help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument("--mask_prob_eos", default=0, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=20,
                        help="Max tokens of prediction.")
    parser.add_argument('--mask_source_words', action='store_true',
                        help="Whether to mask source words for training")
    parser.add_argument('--skipgram_prb', type=float, default=0.0,
                        help='prob of ngram mask')
    parser.add_argument('--skipgram_size', type=int, default=1,
                        help='the max size of ngram mask')
    parser.add_argument('--mask_whole_word', action='store_true',
                        help="Whether masking a whole word.")
    parser.add_argument('--do_l2r_training', action='store_true',
                        help="Whether to do left to right training")
    parser.add_argument('--has_sentence_oracle', action='store_true',
                        help="Whether to have sentence level oracle for training. "
                             "Only useful for summary generation")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="Number of workers for the data loader.")
    # Model Paramters
    parser.add_argument("--sop", action='store_true',
                        help="whether use sop task.")
    parser.add_argument("--train_batch_size",default=32,type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",default=64,type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for hidden states.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for attention probabilities.")
    parser.add_argument('--relax_projection', action='store_true',
                        help="Use different projection layers for tasks.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")

    # Train Eval Test Paramters

    parser.add_argument("--checkpoint_steps", required=True, type=int,
                        help="save model eyery checkpoint_steps")

    parser.add_argument("--total_steps", required=True, type=int,
                        help="all steps of training model")

    parser.add_argument("--max_checkpoint", required=True, type=int,
                        help="max saved model in model_output_dir")

    parser.add_argument("--examples_size_once", type=int, default=1000,
                        help="read how many examples every time in pretrain or finetune")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="process rank in local")
    parser.add_argument("--local_debug", action='store_true',
                        help="whether debug")
    parser.add_argument("--do_train",action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--fine_tune",action='store_true',
                        help="Whether to run fine_tune.")
    parser.add_argument("--do_eval",action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",default=0.01,type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--finetune_decay",action='store_true',
                        help="Weight decay to the original weights.")
    parser.add_argument("--num_train_epochs",default=3.0,type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",default=0.1,type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',type=int,default=1,
                        help="Number of updates   accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp32_embedding', action='store_true',
                        help="Whether to use 32-bit float precision instead of 16-bit for embeddings")
    parser.add_argument('--loss_scale', type=str, default='dynamic',
                        help='(float or str, optional, default=None):  Optional property override.  '
                             'If passed as a string,must be a string representing a number, e.g., "128.0", or the string "dynamic".')
    parser.add_argument('--opt_level', type=str, default='O1',
                        help=' (str, optional, default="O1"):  Pure or mixed precision optimization level.  '
                             'Accepted values are "O0", "O1", "O2", and "O3", explained in detail above.')
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--from_scratch', action='store_true',
                        help="Initialize parameters with random values (i.e., training from scratch).")

    # Other Patameters
    parser.add_argument('--seed',type=int,default=42,
                        help="random seed for initialization")
    parser.add_argument('--rank', type=int, default=0,
                        help="global rank of current process")
    parser.add_argument("--world_size", default=2, type=int,
                        help="Number of process(显卡)")


    args = parser.parse_args()
    cur_env = os.environ
    args.rank = int(cur_env.get('RANK', -1))
    args.world_size = int(cur_env.get('WORLD_SIZE', -1))
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)
    assert args.train_batch_size >= 1, 'batch_size < 1 '

    # 更新一次模型参数需要多少个样本
    examples_per_update = args.world_size * args.train_batch_size * args.gradient_accumulation_steps
    args.examples_size_once = args.examples_size_once // examples_per_update * examples_per_update
    if args.fine_tune:
        args.examples_size_once = examples_per_update

    os.makedirs(args.model_output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.model_output_dir, 'unilm_config.json'), 'w'), sort_keys=True, indent=2)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = torch.cuda.device_count()
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
    logger.info("world_size:{}, rank:{}, device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        args.world_size, args.rank, device, n_gpu, bool(args.world_size > 1), args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if not args.fine_tune and not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=args.do_lower_case)
    if args.max_position_embeddings:
        tokenizer.max_len = args.max_position_embeddings
    if args.local_rank == 0:
        dist.barrier()
    bi_uni_pipeline = [
        Preprocess4Seq2seq(args.max_pred, args.mask_prob, list(tokenizer.vocab.keys()),
        tokenizer.convert_tokens_to_ids, args.max_seq_length, new_segment_ids=args.new_segment_ids,
        truncate_config={'max_len_a': args.max_len_a,  'max_len_b': args.max_len_b, 'trunc_seg': args.trunc_seg,
                         'always_truncate_tail': args.always_truncate_tail},
        mask_source_words=args.mask_source_words, skipgram_prb=args.skipgram_prb, skipgram_size=args.skipgram_size,
        mask_whole_word=args.mask_whole_word, mode="s2s", has_oracle=args.has_sentence_oracle, num_qkv=args.num_qkv,
        s2s_special_token=args.s2s_special_token, s2s_add_segment=args.s2s_add_segment,
        s2s_share_segment=args.s2s_share_segment, pos_shift=args.pos_shift, fine_tune=args.fine_tune)]
    file_oracle = None
    if args.has_sentence_oracle:
        file_oracle = os.path.join(args.data_dir, 'train.oracle')

    # t_total表示模型参数更新的次数
    # t_total = args.train_steps
    # Prepare model
    recover_step = _get_max_epoch_model(args.model_output_dir)
    cls_num_labels = 2
    type_vocab_size = 6 + \
        (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
    num_sentlvl_labels = 2 if args.has_sentence_oracle else 0
    relax_projection = 4 if args.relax_projection else 0
    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    if (recover_step is None) and (args.model_recover_path is None):
        # if _state_dict == {}, the parameters are randomly initialized
        # if _state_dict == None, the parameters are initialized with bert-init
        _state_dict = {} if args.from_scratch else None
        model = BertForPreTrainingLossMask.from_pretrained(args.bert_model, state_dict=_state_dict,num_labels=cls_num_labels, num_rel=0, type_vocab_size=type_vocab_size,
            config_path=args.config_path, task_idx=3, num_sentlvl_labels=num_sentlvl_labels,max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding, relax_projection=relax_projection, new_pos_ids=args.new_pos_ids,ffn_type=args.ffn_type, hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob, num_qkv=args.num_qkv,seg_emb=args.seg_emb, local_debug=args.local_debug)
        global_step = 0
    else:
        if recover_step:
            logger.info("***** Recover model: %d *****", recover_step)
            model_recover = torch.load(os.path.join(args.output_model_dir, "model.{0}.bin".format(recover_step)), map_location='cpu')
            # recover_step == number of epochs
            global_step = math.floor(recover_step * args.checkpoint_step)
        # 预训练时模型的参数初始化，比如使用chinese-bert-base的模型参数进行初始化
        elif args.model_recover_path:
            logger.info("***** Recover model: %s *****", args.model_recover_path)
            model_recover = torch.load(args.model_recover_path, map_location='cpu')
            global_step = 0
        model = BertForPreTrainingLossMask.from_pretrained(state_dict=model_recover,num_labels=cls_num_labels, num_rel=0, type_vocab_size=type_vocab_size,
                config_path=args.config_path, task_idx=3, num_sentlvl_labels=num_sentlvl_labels,
                max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,fp32_embedding=args.fp32_embedding,
                relax_projection=relax_projection, new_pos_ids=args.new_pos_ids,ffn_type=args.ffn_type, hidden_dropout_prob=args.hidden_dropout_prob,
                attention_probs_dropout_prob=args.attention_probs_dropout_prob, num_qkv=args.num_qkv,seg_emb=args.seg_emb, local_debug=args.local_debug)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("模型参数： {}".format(total_trainable_params))
    if args.local_rank == 0:
        dist.barrier()

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion, t_total=args.total_steps)
    if args.amp and args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer,opt_level=args.opt_level,loss_scale=args.loss_scale)
        from apex.parallel import DistributedDataParallel as DDP
        model = DDP(model)
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if recover_step:
        logger.info("** ** * Recover optimizer: %d * ** **", recover_step)
        optim_recover = torch.load(os.path.join(args.model_output_dir, "optim.{0}.bin".format(recover_step)), map_location='cpu')
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.load_state_dict(optim_recover)
        if args.fp16 and args.amp:
            amp_recover = torch.load(os.path.join(args.model_output_dir, "amp.{0}.bin".format(recover_step)), map_location='cpu')
            logger.info("** ** * Recover amp: %d * ** **", recover_step)
            amp.load_state_dict(amp_recover)
    logger.info("** ** * CUDA.empty_cache() * ** **")
    torch.cuda.empty_cache()


    if args.rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)
    logger.info("***** Running training *****")
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Param Update Num = %d", args.total_steps)
    model.train()

    PRE = "rank{},local_rank {},".format(args.rank, args.local_rank)
    step = 1
    start = time.time()
    train_data_loader = TrainDataLoader(bi_uni_pipline=bi_uni_pipeline, examples_size_once=args.examples_size_once,
                                  world_size=args.world_size, train_batch_size=args.train_batch_size,
                                  num_workers=args.num_workers, data_dir=args.data_dir, tokenizer=tokenizer,max_len=args.max_seq_length)
    best_result = -float('inf')
    for global_step,batch in enumerate(train_data_loader, start=global_step):
        batch = [t.to(device) if t is not None else None for t in batch]
        if args.has_sentence_oracle:
            input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, task_idx, sop_label, oracle_pos, oracle_weights, oracle_labels = batch
        else:
            input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, task_idx, sop_label = batch
            oracle_pos, oracle_weights, oracle_labels = None, None, None
        if not args.sop:
            # 不使用sop训练任务
            sop_label = None
        loss_tuple = model(input_ids, segment_ids, input_mask, masked_lm_labels=lm_label_ids,next_sentence_label=sop_label,masked_pos=masked_pos,masked_weights=masked_weights,
                           task_idx=task_idx,masked_pos_2=oracle_pos, masked_weights_2=oracle_weights,masked_labels_2=oracle_labels, mask_qkv=mask_qkv)
        masked_lm_loss, next_sentence_loss = loss_tuple
        # mean() to average on multi-gpu.
        if n_gpu > 1:
            masked_lm_loss = masked_lm_loss.mean()
            next_sentence_loss = next_sentence_loss.mean()
        # ensure that accumlated gradients are normalized
        if args.gradient_accumulation_steps > 1:
            masked_lm_loss = masked_lm_loss / args.gradient_accumulation_steps
            next_sentence_loss = next_sentence_loss / args.gradient_accumulation_steps
        if not args.sop:
            loss = masked_lm_loss
        else:
            loss = masked_lm_loss + next_sentence_loss
        if args.fp16 and args.amp:
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
        else:
            loss.backward()
        if (global_step+1) % args.gradient_accumulation_steps == 0:
            if args.rank == 0:
                writer.add_scalar('unilm/mlm_loss', masked_lm_loss, global_step)
                writer.add_scalar('unilm/sop_loss', next_sentence_loss, global_step)
            lr_this_step = args.learning_rate * warmup_linear(global_step/args.total_steps, args.warmup_proportion)
            if args.fp16:
                # modify learning rate with special warm up BERT uses
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            #global_step += 1
            #更新一次模型参数花费的时间，单位:秒
            cost_time_per_update = time.time() - start
            # 更新完所有参数花费的时间，单位:小时
            need_time = cost_time_per_update * (args.total_steps - global_step) / 3600.0
            cost_time_per_chectpoint = cost_time_per_update * args.checkpoint_steps / 3600.0
            start = time.time()
            if args.local_rank in [-1,0]:
                INFO = PRE + '当前/chcklpoint_steps/total:{}/{}/{},loss{}/{},更新一次参数{}秒,checkpoint_steps {}小时,' \
                             '训练完成{}小时\n'.format(global_step, args.checkpoint_steps, args.total_steps,
                                                 round(masked_lm_loss.item(), 5),
                                                 round(next_sentence_loss.item(), 5), round(cost_time_per_update, 4),
                                                 round(cost_time_per_chectpoint, 3), round(need_time, 3))
                print(INFO)
        # Save a trained model
        if (global_step+1) % args.checkpoint_steps == 0:
            checkpoint_index = (global_step+1) % args.checkpoint_steps
            if args.rank >= 0:
                train_data_loader.train_sampler.set_epoch(checkpoint_index)
            # if args.eval:
            #     # 如果是pretrain，验证MLM；如果微调，验证评价指标
            #     result = None
            #if best_result < result and _get_checkpont_num(args.model_output_num):
            if args.rank in [0,-1]:
                logger.info("** ** * Saving  model and optimizer * ** **")

                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.model_output_dir, "model.{0}.bin".format(checkpoint_index))
                torch.save(model_to_save.state_dict(), output_model_file)
                output_optim_file = os.path.join(args.model_output_dir, "optim.{0}.bin".format(checkpoint_index))
                torch.save(optimizer.state_dict(), output_optim_file)
                if args.fp16 and args.amp:
                    logger.info("** ** * Saving  amp state  * ** **")
                    output_amp_file = os.path.join(args.model_output_dir, "amp.{0}.bin".format( checkpoint_index))
                    torch.save(amp.state_dict(), output_amp_file)
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
    if args.rank == 0:
        writer.close()
        print('** ** * train finished * ** **')
if __name__ == "__main__":
    main()
