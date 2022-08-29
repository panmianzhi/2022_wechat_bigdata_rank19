python src/pretrain.py \
--bert_seq_length 256 \
--batch_size 128 \
--val_batch_size 512 \
--max_epochs 40 \
--mlm_weight 0.6 --itm_weight 0.1 --mrfr_weight 0.3 \
--bert_dir src/data/nezha-chinese-base \
--num_gpus 4

python src/pretrain.py \
--bert_seq_length 256 \
--batch_size 128 \
--val_batch_size 512 \
--max_epochs 40 \
--mlm_weight 0.5 --itm_weight 0.3 --mrfr_weight 0.2 \
--bert_dir src/data/macbert-base \
--num_gpus 4

python src/pretrain.py \
--bert_seq_length 256 \
--batch_size 128 \
--val_batch_size 512 \
--max_epochs 40 \
--mlm_weight 0.5 --itm_weight 0.3 --mrfr_weight 0.2 \
--bert_dir src/data/roberta-wwm-ext \
--num_gpus 4

python src/kfold_finetune.py \
--bert_seq_length 128 \
--max_epochs 5 \
--batch_size 32 \
--val_batch_size 128 \
--bert_dir src/data/nezha-chinese-base \
--num_gpus 1 \
--pretrain_model_path src/save/nezha_Epoch39.pth \
--savedmodel_path src/save/5fold \
--best_score 0.67 

python src/kfold_finetune.py \
--bert_seq_length 128 \
--max_epochs 5 \
--batch_size 32 \
--val_batch_size 128 \
--bert_dir src/data/macbert-base \
--num_gpus 1 \
--pretrain_model_path src/save/macbert_Epoch39.pth \
--savedmodel_path src/save/5fold \
--best_score 0.67 

python src/kfold_finetune.py \
--bert_seq_length 128 \
--max_epochs 5 \
--batch_size 32 \
--val_batch_size 128 \
--bert_dir src/data/roberta-wwm-ext \
--num_gpus 1 \
--pretrain_model_path src/save/roberta_Epoch39.pth \
--savedmodel_path src/save/5fold \
--best_score 0.67 