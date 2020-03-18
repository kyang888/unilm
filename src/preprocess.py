import json
import argparse
import os
import multiprocessing
import sys
from random import randint, shuffle, choice
import shutil
import pickle
sys.path.insert(0, "/data/nfs/yangkang227/summary/unilm/src")

print(sys.path)
from pytorch_pretrained_bert.tokenization import BertTokenizer
import biunilm.seq2seq_loader as seq2seq_loader
from preprocessors import BaiDuPreprocessor,News2016Preprocessor,WikiPreprocessor
def tokenize(file_name, args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    info = '进程:{},tokenize src_tgt:{},保存到:{}'.format(os.getpid(), file_name, file_name)
    print(info)
    count = 0
    with open(os.path.join(args.tokenized_dir, file_name), 'w', encoding='utf-8') as fout:
        with open(os.path.join(args.src_tgt_dir, file_name), 'r', encoding='utf-8') as fin:
            for line in fin:
                src, tgt = line.strip().split('***\t***')
                src_tk = tokenizer.tokenize(src)
                tgt_tk = tokenizer.tokenize(tgt)
                assert len(src_tk) > 0
                assert len(tgt_tk) > 0
                count += 1
                if count % args.log_per_size == 0:
                    print(info + "， 已经tokenize了{}个句子对".format(count))
                line = json.dumps(src_tk, ensure_ascii=False) + "***\t***" + json.dumps(tgt_tk, ensure_ascii=False) + '\n'
                if count == 1:
                    print(line)
                fout.write(line)

def processed(file_name, args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    if args.max_position_embeddings:
        tokenizer.max_len = args.max_position_embeddings
    bi_uni_pipeline = [
        seq2seq_loader.Preprocess4Seq2seq(args.max_pred, args.mask_prob, list(tokenizer.vocab.keys()),
                                          tokenizer.convert_tokens_to_ids, args.max_seq_length,
                                          new_segment_ids=args.new_segment_ids,
                                          truncate_config={'max_len_a': args.max_len_a, 'max_len_b': args.max_len_b,
                                                           'trunc_seg': args.trunc_seg,
                                                           'always_truncate_tail': args.always_truncate_tail},
                                          mask_source_words=args.mask_source_words, skipgram_prb=args.skipgram_prb,
                                          skipgram_size=args.skipgram_size,
                                          mask_whole_word=args.mask_whole_word, mode="s2s",
                                          has_oracle=args.has_sentence_oracle, num_qkv=args.num_qkv,
                                          s2s_special_token=args.s2s_special_token,
                                          s2s_add_segment=args.s2s_add_segment,
                                          s2s_share_segment=args.s2s_share_segment, pos_shift=args.pos_shift)]

    info = '进程:{},处理 tokenized_dir:{},保存到processed:{}'.format(os.getpid(), file_name, file_name)
    print(info)
    count = 0
    examples = []
    with open(os.path.join(args.processed_dir, file_name), 'w', encoding='utf-8') as fout:
        with open(os.path.join(args.tokenized_dir, file_name), 'r', encoding='utf-8') as fin:
            for line in fin:
                src, tgt = line.strip().split('***\t***')
                src_tk = json.loads(src)
                tgt_tk = json.loads(tgt)
                assert len(src_tk) > 0
                assert len(tgt_tk) > 0
                proc = choice(bi_uni_pipeline)
                example = proc((src_tk, tgt_tk))

                example = json.dumps(example)

                count += 1
                if count % args.log_per_size == 0:
                    print(info + "， 已经 processed 了{}个句子对".format(count))
                if count == 1:
                    print(example)
                fout.write(example + '\n')

def tokenization_to_processed(args):
    print(args)
    for key in args.__dict__:
        print(key+":"+ str(args.__dict__[key]))
    if os.path.exists(args.processed_dir):
        shutil.rmtree(args.processed_dir)
    os.makedirs(args.processed_dir)
    tokenized_files = os.listdir(args.tokenized_dir)[0:1]
    print(tokenized_files)
    p_list = []
    for file_name in tokenized_files:
        p = multiprocessing.Process(target=processed, args=(file_name, args))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
def src_tgt_to_tokenization(args):
    if os.path.exists(args.tokenized_dir):
        shutil.rmtree(args.tokenized_dir)
    os.makedirs(args.tokenized_dir)
    src_tgt_files = os.listdir(args.src_tgt_dir)
    p_list = []
    for file_name in src_tgt_files:
        p = multiprocessing.Process(target=tokenize, args=(file_name, args))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()


def raw_to_src_tgt(args):
    # 原始的语料文件名
    raw_files = args.raw_file.split('+')
    # 原始语料的文件路径
    raw_paths = [os.path.join(args.raw_dir, file) for file in raw_files]
    preprocessors = [eval(p + "('" + f + "')") for p, f in zip(args.preprocessors.split('+'), raw_paths)]
    assert len(raw_files) == len(raw_paths) == len(preprocessors)
    # 创建src_tgt_dir文件夹
    if os.path.exists(args.src_tgt_dir):
        shutil.rmtree(args.src_tgt_dir)
    os.makedirs(args.src_tgt_dir)
    for idx, file in enumerate(raw_files):
        corpus_name = file[:file.index('.')] if '.' in file else file
        preprocessor = preprocessors[idx]
        print(preprocessor.name)
        suffixe = 0
        count = 0
        path = os.path.join(args.src_tgt_dir, corpus_name + str(suffixe))
        f = open(path, 'w', encoding='utf-8')
        for src, tgt in preprocessor.input_fn(max_len=args.max_seq_length,
                                              min_len=args.min_seq_length,threshold=args.threshold):
            f.write(src + '***\t***' + tgt + '\n')
            count += 1
            if count % args.log_per_size == 0:
                print('{} 已经保存 {} 个句子对'.format(path, count))
            if args.file_size == count:
                f.close()
                print('{} 中保存{} 个句子对'.format(path, count))
                count = 0
                suffixe += 1
                f = open(os.path.join(args.src_tgt_dir, corpus_name + str(suffixe)), 'w', encoding='utf-8')
        f.close()
def tokenize_finetune_data(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    for file_name in os.listdir(args.finetune_data_dir):
        info = 'tokenize raw_finetune:{},保存到 tokenized_finetune:{}'.format(os.getpid(), file_name, file_name)
        print(info)
        count = 0
        with open(os.path.join(args.tokenized_finetune_data_dir, file_name), 'w', encoding='utf-8') as fout:
            with open(os.path.join(args.finetune_data_dir, file_name), 'r', encoding='utf-8') as fin:
                for line in fin:
                    example = json.loads(line.strip())
                    src = example["summarization"].strip()
                    tgt = example["article"].strip()
                    src_tk = tokenizer.tokenize(src)
                    tgt_tk = tokenizer.tokenize(tgt)
                    assert len(src_tk) > 0
                    assert len(tgt_tk) > 0
                    count += 1
                    if count % args.log_per_size == 0:
                        print(info + "， 已经tokenize了{}个句子对".format(count))
                    line = json.dumps(src_tk, ensure_ascii=False) + "***\t***" + json.dumps(tgt_tk, ensure_ascii=False) + '\n'
                    if count == 1:
                        print(line)
                    fout.write(line)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=None, type=str, required=True,
                        help="raw baiki corpus path.")
    # 原始的文件名与处理器名字一一对应
    parser.add_argument("--raw_file", default=None, type=str,required=True,
                        help="")
    parser.add_argument("--preprocessors", default=None, type=str,required=True,
                        help="")
    parser.add_argument('--raw_finetune_data_dir', default=None, type=str,
                        help="fine tune data dir.")
    parser.add_argument('--tokenized_finetune_data_dir', default=None, type=str,
                        help="save tokenized fine tune data dir.")
    parser.add_argument("--src_tgt_dir", default=None, type=str, help="sentence pair min len")
    parser.add_argument("--tokenized_dir", default=None, type=str, help="sentence pair min len")
    parser.add_argument("--processed_dir", default=None, type=str, help="sentence pair min len")
    parser.add_argument("--bert_model", default=None, type=str, help="sentence pair min len")
    parser.add_argument("--max_seq_length",default=128,type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--min_seq_length",default=10,type=int,help="sentence pair min len")
    parser.add_argument("--threshold", default=100, type=int, help="")
    parser.add_argument("--file_size", default=50000, type=int, help="sentence pair min len")
    parser.add_argument("--log_per_size", default=5000, type=int, help="sentence pair min len")
    parser.add_argument("--do_lower_case",action='store_true',
                        help="Set this flag if you are using an uncased model.")
    # parser.add_argument("--no_cuda", action='store_true',
    #                     help="Whether not to use CUDA when available")
    parser.add_argument('--seed',type=int,default=42,
                        help="random seed for initialization")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
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
    parser.add_argument('--has_sentence_oracle', action='store_true',
                        help="Whether to have sentence level oracle for training. "
                             "Only useful for summary generation")
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                        help="max position embeddings")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
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

    args = parser.parse_args()
    #raw_to_src_tgt(args)
    #src_tgt_to_tokenization(args)
    #tokenization_to_processed(args)
    tokenize_finetune_data(args)

if __name__ == "__main__":
    main()



