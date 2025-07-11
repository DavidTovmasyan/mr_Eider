nohup python3 train.py \
--data_dir ./dataset/docred \
--transformer_type roberta \
--model_name_or_path xlm-roberta-base \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--learning_rate 3e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 30.0 \
--seed 66 \
--num_class 96 \
--rel_mode _inc1_comb \
--num_incr_head 2 \
--evaluation_steps -1 \
--save_path chkpt/EIDER_bert_eider__exp_combined_comb__test.pt \
--ablation eider \
--name _exp_combined_comb_ \
--feature_path saved_features \
--coref_method hoi \
--eval_mode dev_only \
--evi_eval_mode none \
--ensemble_mode 2 \
--ensemble_ablation evi_rule \
--evi_pred_file evi_results_eider_xlm-roberta-base.pkl \
--load_path chkpt/EIDER_bert_eider__exp_combined_incremental__best.pt \
--use_combined_inference true \
> output__exp_combined_comb__test.log 2>&1