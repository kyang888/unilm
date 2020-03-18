ROOT_DIR=/home/unilm
CODE_DIR=$ROOT_DIR/src
DATA_DIR=$ROOT_DIR/data/pretrain
ENTRY=$CODE_DIR/train.py
OUTPUT_DIR=$ROOT_DIR/output

python -m torch.distributed.launch --nproc_per_node=1 \
--nnodes 1 \
--node_rank 0 \
--master_addr 10.11.6.11 \
--master_port 12345 \
train.py  \
--do_train  \
--data_dir $DATA_DIR/pretrain \
--vocab_path   $ROOT_DIR/vocab.txt \
--config_path $ROOT_DIR/bert_config.json \
--model_output_dir $OUTPUT_DIR/model\pretrain \
--log_dir  $OUTPUT_DIR/log \
--model_recover_path $ROOT_DIR/bert-base-chinese/pytorch_model.bin \
--max_seq_length 512 \
--max_position_embeddings 512 \
--do_lower_case  \
--new_segment_ids  \
--new_pos_ids  \
--num_workers 4 \
--max_pred 48 \
--mask_prob 0.15 \
--train_batch_size 2 \
--relax_projection \
--checkpoint_steps 20 \
--total_steps 200 \
--max_checkpoint 4 \
--examples_size_once  160 \
--gradient_accumulation_steps  1 \
--learning_rate 0.00002 \
--warmup_proportion 0.1 \
--label_smoothing 0.1
