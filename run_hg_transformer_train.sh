export CUDA_VISIBLE_DEVICES="0"
#pretrained_model_name="microsoft/MiniLM-L12-H384-uncased"
pretrained_model_name="bert-base-uncased"
data_dir=/mnt/d/MLData/data/summarization/bert_data/bert_data_cnndm_ext
python run_bert_extractive_summarizer_hg_train.py \
--data_dir /mnt/d/MLData/data/summarization/bert_data/bert_data_cnndm_final \
--dataset_name cnn_daily_mail \
--model_name Bert \
--pretrained_model_name ${pretrained_model_name} \
--gpus 1 \
--max_epochs 5 \
--lr 2e-3 \
--lr_warm_up_steps 10000 \
--batch_size 32 \
--num_workers 16 \
--project_name extractive_text_summarization \
--default_root_dir ./experiments/logs \
--exp_tag train_on_valid