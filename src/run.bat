python train.py  --do_train  --local_debug ^
--data_dir E:\py-workspace\summary\data\raw ^
--vocab_path   E:\py-workspace\unilm\bert-base-chinese ^
--config_path E:\py-workspace\unilm\bert_config.json ^
--model_output_dir E:\py-workspace\unilm\output\model_output\pretrain ^
--log_dir  E:\py-workspace\unilm\output\log ^
--model_recover_path E:\py-workspace\unilm\bert-base-chinese\pytorch_model.bin ^
--max_seq_length 512 ^
--max_position_embeddings 512 ^
--do_lower_case  ^
--new_segment_ids  ^
--new_pos_ids  ^
--num_workers 4 ^
--max_pred 48 ^
--mask_prob 0.15 ^
--train_batch_size 2 ^
--relax_projection ^
--checkpoint_steps 20 ^
--total_steps 200 ^
--max_checkpoint 4 ^
--examples_size_once  160 ^
--gradient_accumulation_steps  1 ^
--learning_rate 0.00002 ^
--warmup_proportion 0.1 ^
--label_smoothing 0.1


