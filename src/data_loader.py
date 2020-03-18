from random import randint, shuffle, choice
from random import random as rand
import math
import torch
import json
import os
import random
from loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.

TOTAL_SPLIT_TOKENS = ['。', '？', '!', '！', '?', ',', '，', ';', '；']
BIG_SPLIT_TOKENS = ['。', '？', '!', '！', '?']
SMALL_SPLIT_TOKENS = [',', '，', ';', '；']



def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b


class Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, file_oracle=None,short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[], ex_list=[]):
        super().__init__()
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.sent_reverse_order = sent_reverse_order
        assert len(ex_list) > 0, "ex_list 应该大于 0"
        self.ex_list = ex_list

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choice(self.bi_uni_pipeline)
        instance = proc(instance)
        return instance

class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0,
                 skipgram_size=0, block_mask=False, mask_whole_word=False,
                 new_segment_ids=False, truncate_config={},
                 mask_source_words=False, mode="s2s", has_oracle=False, num_qkv=0,
                 s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False, fine_tune=False):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self._triu_matrix = torch.triu(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.task_idx = 3   # relax projection layer for different tasks
        self.mask_source_words = mask_source_words
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.has_oracle = has_oracle
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift
        self.fine_tune=fine_tune

    def __create_task_idx__(self):
        """
        :return: 预训练任务编号，特殊token的个数
        0：双向语言模型， 3表示双向语言模型有3个特殊token, 例如: [CLS]A[SEP]B[SEP]
        1：L2R语言模型,  2表示单向语言模型有2个特殊token,例如[CLS]AB[SEP]
        2：R2L语言模型
        3：Seq2Seq语言模型
        """
        random_number = rand()
        if random_number <= 1.0/3:
            return 0, 3
        elif random_number <= 3.0/6:
            return 1, 2
        elif random_number <= 2.0/3:
            return 2, 2
        else:
            return 3, 3

    def __call__(self, instance):
        tokens_a, tokens_b = instance[:2]
        if self.pos_shift:
            tokens_b = ['[S2S_SOS]'] + tokens_b
        self.task_idx, special_token_num = self.__create_task_idx__()
        # 微调 阶段
        if self.fine_tune:
            self.task_idx = 3
            special_token_num = 3

        # -3  for special tokens [CLS], [SEP], [SEP]
        # num_truncated_a, _ = truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3, max_len_a=self.max_len_a,
        #                                           max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

        # 修改
        num_truncated_a, _ = truncate_tokens_pair(tokens_a, tokens_b, self.max_len - special_token_num, max_len_a=self.max_len_a,
                                                  max_len_b=self.max_len_b, trunc_seg=self.trunc_seg,
                                                  always_truncate_tail=self.always_truncate_tail)

        num_tokens_a = len(tokens_a) + special_token_num - 1
        num_tokens_b = len(tokens_b) + 1

        # 增加
        # 双向语言模型的sop任务
        sop_label = -1
        #如果随机数小于0.5，构造sop任务的负例
        if self.task_idx == 0:
            if rand() < 0.5:
                sop_label = 0
                tokens_a, tokens_b = tokens_b, tokens_a
            else:
                sop_label = 1
        # Add Special Tokens
        if self.s2s_special_token:
            tokens = ['[S2S_CLS]'] + tokens_a + \
                ['[S2S_SEP]'] + tokens_b + ['[SEP]']
        else:
            # tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
            # 增加
            if self.task_idx == 0:
                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
            elif self.task_idx == 1:
                tokens = ['[CLS]'] + tokens_a + tokens_b + ['[SEP]']
            elif self.task_idx == 2:
                tokens = ['[CLS]'] + tokens_a + tokens_b + ['[SEP]']
            else:
                tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']


        # if self.new_segment_ids:
        #     if self.mode == "s2s":
        #         if self.s2s_add_segment:
        #             if self.s2s_share_segment:
        #                 segment_ids = [0] + [1] * \
        #                     (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
        #             else:
        #                 segment_ids = [4] + [6] * \
        #                     (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
        #         else:
        #             segment_ids = [4] * (len(tokens_a)+2) + \
        #                 [5]*(len(tokens_b)+1)
        #     else:
        #         segment_ids = [2] * (len(tokens))
        # else:
        #     segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        # 修改
        #这里全部注释掉，默认使用 双向:0,1  单向:2,3  seq2seq:4,5
        if self.new_segment_ids:
            if self.task_idx == 0:
                segment_ids = [0]*(len(tokens_a)+special_token_num-1) + [1]*(len(tokens_b)+1)
            elif self.task_idx == 1:
                segment_ids = [2] * (len(tokens_a) + special_token_num - 1) + [2] * (len(tokens_b) + 1)
            elif self.task_idx == 2:
                segment_ids = [3] * (len(tokens_a) + special_token_num - 1) + [3] * (len(tokens_b) + 1)
            else:
                segment_ids = [4] * (len(tokens_a) + special_token_num - 1) + [5] * (len(tokens_b) + 1)
        else:
            segment_ids = [0]*(len(tokens_a)+special_token_num-1) + [1]*(len(tokens_b)+1)


        if self.pos_shift:
            n_pred = min(self.max_pred, len(tokens_b))
            masked_pos = [len(tokens_a)+2+i for i in range(len(tokens_b))]
            masked_weights = [1]*n_pred
            masked_ids = self.indexer(tokens_b[1:]+['[SEP]'])
        else:
            # For masked Language Models
            # the number of prediction is sometimes less than max_pred when sequence is short
            effective_length = len(tokens_b)

            # 增加
            if self.task_idx in [0, 1, 2]:
                self.mask_source_words = True
            else:
                self.mask_source_words = False

            if self.mask_source_words:
                effective_length += len(tokens_a)
            n_pred = min(self.max_pred, max(1, int(round(effective_length*self.mask_prob))))
            # candidate positions of masked tokens
            cand_pos = []
            special_pos = set()

            # for i, tk in enumerate(tokens):
            #     # only mask tokens_b (target sequence)
            #     # we will mask [SEP] as an ending symbol
            #     if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
            #         cand_pos.append(i)
            #     elif self.mask_source_words and (i < len(tokens_a)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
            #         cand_pos.append(i)
            #     else:
            #         special_pos.add(i)
            # shuffle(cand_pos)

            # 修改
            for i, tk in enumerate(tokens):
                # only mask tokens_b (target sequence)
                # we will mask [SEP] as an ending symbol
                if (i >= len(tokens_a)+special_token_num-1) and (tk != '[CLS]'):
                    cand_pos.append(i)
                elif self.mask_source_words and (i < len(tokens_a)+special_token_num-1) and (tk != '[CLS]') and ('SEP]' not in tk):
                    cand_pos.append(i)
                else:
                    special_pos.add(i)
            shuffle(cand_pos)

            masked_pos = set()
            max_cand_pos = max(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos in masked_pos:
                    continue

                def _expand_whole_word(st, end):
                    new_st, new_end = st, end
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1
                    return new_st, new_end

                if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                    # ngram
                    cur_skipgram_size = randint(2, self.skipgram_size)
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(
                            pos, pos + cur_skipgram_size)
                    else:
                        st_pos, end_pos = pos, pos + cur_skipgram_size
                else:
                    # directly mask
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                    else:
                        st_pos, end_pos = pos, pos + 1

                for mp in range(st_pos, end_pos):
                    if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                        masked_pos.add(mp)
                    else:
                        break

            masked_pos = list(masked_pos)
            if len(masked_pos) > n_pred:
                shuffle(masked_pos)
                masked_pos = masked_pos[:n_pred]

            masked_tokens = [tokens[pos] for pos in masked_pos]
            for pos in masked_pos:
                if rand() < 0.8:  # 80%
                    tokens[pos] = '[MASK]'
                elif rand() < 0.5:  # 10%
                    tokens[pos] = get_random_word(self.vocab_words)
            # when n_pred < max_pred, we only calculate loss within n_pred
            masked_weights = [1]*len(masked_tokens)

            # Token Indexing
            masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)

        segment_ids.extend([0]*n_pad)

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(tokens_a)+2) + [1] * (len(tokens_b)+1)
            mask_qkv.extend([0]*n_pad)
        else:
            mask_qkv = None




        # if self.mode == "s2s":
        #     input_mask[:, :len(tokens_a)+2].fill_(1)
        #     second_st, second_end = len(
        #         tokens_a)+2, len(tokens_a)+len(tokens_b)+3
        #     input_mask[second_st:second_end, second_st:second_end].copy_(
        #         self._tril_matrix[:second_end-second_st, :second_end-second_st])
        # else:
        #     st, end = 0, len(tokens_a) + len(tokens_b) + 3
        #     input_mask[st:end, st:end].copy_(self._tril_matrix[:end, :end])


        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        start, end = 0, len(tokens_a) + len(tokens_b) + special_token_num
        if self.task_idx == 0:
            input_mask[start:end, start:end].fill_(1)
        elif self.task_idx == 1:
            input_mask[start:end, start:end].copy_(self._tril_matrix[start:end, start:end])
        elif self.task_idx == 2:
            input_mask[start:end, start:end].copy_(self._triu_matrix[start:end, start:end])
        else:
            input_mask[:, :len(tokens_a)+special_token_num-1].fill_(1)
            second_st, second_end = len(tokens_a)+special_token_num-1, len(tokens_a)+len(tokens_b)+special_token_num
            input_mask[second_st:second_end, second_st:second_end].copy_(self._tril_matrix[:second_end-second_st, :second_end-second_st])


        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        oracle_pos = None
        oracle_weights = None
        oracle_labels = None
        if self.has_oracle:
            s_st, labls = instance[2:]
            oracle_pos = []
            oracle_labels = []
            for st, lb in zip(s_st, labls):
                st = st - num_truncated_a[0]
                if st > 0 and st < len(tokens_a):
                    oracle_pos.append(st)
                    oracle_labels.append(lb)
            oracle_pos = oracle_pos[:20]
            oracle_labels = oracle_labels[:20]
            oracle_weights = [1] * len(oracle_pos)
            if len(oracle_pos) < 20:
                x_pad = 20 - len(oracle_pos)
                oracle_pos.extend([0] * x_pad)
                oracle_labels.extend([0] * x_pad)
                oracle_weights.extend([0] * x_pad)

            return (input_ids, num_tokens_a, num_tokens_b, mask_qkv, masked_ids,
                    masked_pos, masked_weights, self.task_idx,sop_label,
                    oracle_pos, oracle_weights, oracle_labels)

        return (input_ids, segment_ids, input_mask , mask_qkv, masked_ids, masked_pos, masked_weights, self.task_idx, sop_label)


class TrainDataLoader(object):
    def __init__(self, bi_uni_pipline, world_size, train_batch_size, num_workers,examples_size_once, data_dir, tokenizer, max_len):
        # 虚假的数据集
        self.bi_uni_pipeline = bi_uni_pipline
        self.world_size = world_size
        self.examples_size_once = examples_size_once
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        # 虚假的训练集
        self.train_dataset = Seq2SeqDataset(bi_uni_pipeline=bi_uni_pipline, ex_list=[0] * self.examples_size_once)
        if self.world_size < 1:
            self.train_sampler = RandomSampler(self.train_dataset, replacement=False)
        else:
            self.train_sampler = DistributedSampler(self.train_dataset)
            # _batch_size = args.train_batch_size // dist.get_world_size()
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size,sampler=self.train_sampler,
                                                       num_workers=self.num_workers,collate_fn=batch_list_to_batch_tensors,pin_memory=False)
    def __iter__(self):
        for file_idx, ex_list in enumerate(self.read_sentence_pairs()):
            setattr(self.train_dataset, 'ex_list', ex_list)
            # if self.cur_step % self.checkpoint_step == 0:
            #     self.train_sampler.set_epoch(self.epoch)
            for batch in self.train_dataloader:
                yield batch


    def read_sentence_pairs(self):
        """
        从self.data_dir中读取self.file_size个样本
        :return: list
        """
        sentence_pairs = []
        file_names = [os.path.join(self.data_dir, file_name) for file_name in os.listdir(self.data_dir)]
        file_index = 0
        while file_index < len(file_names):
            file_name = file_names[file_index]
            with open(file_name, 'r', encoding='utf-8') as f:
                for document in f:
                    for tokens_a, tokens_b in self.create_sentence_pairs(document.strip()):


                        sentence_pairs.append((tokens_a, tokens_b))
                        if len(sentence_pairs) == self.examples_size_once:
                            yield sentence_pairs
                            sentence_pairs = []
            if file_index == len(file_names) - 1:
                file_index = 0
    def create_sentence_pairs(self, document):
        sentences = []
        sen = []
        for p, w in enumerate(document):
            sen.append(w)
            if w in BIG_SPLIT_TOKENS or p == len(document)-1:
                if len(sen) > 1:
                    tokens = self.tokenizer.tokenize(''.join(sen))
                    if tokens:
                        sentences.append(tokens)
                sen = []
        i = 0
        current_chunk = []
        current_length = 0


        while i < len(sentences):
            segment = sentences[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(sentences) - 1 or current_length >= self.max_len:
                if len(current_chunk) == 1: continue
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    tokens_a = []
                    tokens_b = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1
                    yield (tokens_a, tokens_b)
                    current_chunk = []
                    current_length = 0
            i += 1









class Preprocess4Seq2seqDecoder(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512,
                 max_tgt_length=128, new_segment_ids=False,
                 mode="s2s", num_qkv=0, s2s_special_token=False,
                 s2s_add_segment=False, s2s_share_segment=False,
                 pos_shift=False, from_source=False):
        super().__init__()
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.max_tgt_length = max_tgt_length
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift
        self.from_source = from_source

    def __call__(self, instance):

        tokens_a, max_a_len = instance

        # Add Special Tokens
        if self.s2s_special_token:
            padded_tokens_a = ['[S2S_CLS]'] + tokens_a + ['[S2S_SEP]']
        else:
            padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']

        decode_mask = torch.zeros(len(self.vocab_words))
        # 如果只从原文生成
        #self.indexer 就是 bert.convert_word_to_ids()方法.
        if self.from_source:
            decode_mask[self.indexer(padded_tokens_a)] = 1
            # torch.where将其反过来， 如果一个词来自原文，那么decode_mask[bert.convert_word_to
            decode_mask = torch.where(decode_mask != 0, torch.full_like(decode_mask, 0), torch.full_like(decode_mask, 1))
        decode_mask[self.indexer(['[CLS]', '[UNK]']) ] = 1
        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(self.max_tgt_length +
                               max_a_len + 2, self.max_len)
        tokens = padded_tokens_a
        if self.new_segment_ids:
            if self.mode == "s2s":
                _enc_seg1 = 0 if self.s2s_share_segment else 4
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [
                            0] + [1]*(len(padded_tokens_a)-1) + [5]*(max_len_in_batch - len(padded_tokens_a))
                    else:
                        segment_ids = [
                            4] + [6]*(len(padded_tokens_a)-1) + [5]*(max_len_in_batch - len(padded_tokens_a))
                else:
                    segment_ids = [4]*(len(padded_tokens_a)) + \
                        [5]*(max_len_in_batch - len(padded_tokens_a))
            else:
                segment_ids = [2]*max_len_in_batch
        else:
            segment_ids = [0]*(len(padded_tokens_a)) \
                + [1]*(max_len_in_batch - len(padded_tokens_a))

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(padded_tokens_a)) + [1] * \
                (max_len_in_batch - len(padded_tokens_a))
        else:
            mask_qkv = None

        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        input_mask = torch.zeros(
            max_len_in_batch, max_len_in_batch, dtype=torch.long)
        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
        else:
            st, end = 0, len(tokens_a) + 2
            input_mask[st:end, st:end].copy_(
                self._tril_matrix[:end, :end])
            input_mask[end:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        return (input_ids, segment_ids, position_ids, input_mask, mask_qkv, self.task_idx, decode_mask)
