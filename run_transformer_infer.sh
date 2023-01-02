export CUDA_VISIBLE_DEVICES="0"
#pretrained_model_name="microsoft/MiniLM-L12-H384-uncased"
pretrained_model_name="distilbert-base-uncased"
ckpt_path="/mnt/d/MLData/Repos/PyTorch-ExtractiveTextSummarization/ckpts/model-2vsovto4/model.ckpt"
python run_bert_extractive_summarizer_infer.py \
--data_dir /mnt/d/MLData/data/summarization/bert_data/bert_data_cnndm_final \
--dataset_name cnn_daily_mail \
--model_name Bert \
--pretrained_model_name ${pretrained_model_name} \
--gpus 1 \
--ckpt_path ${ckpt_path} \
--batch_size 32 \
--num_workers 16 \
--project_name extractive_text_summarization \
--default_root_dir ./experiments/logs