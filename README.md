# Russian GPT-2 

1. Download librusec library 
(http://trec.to/viewtopic.php?p=60)

1. Install dependencies

```sudo xargs -a apt.txt apt install
conda env create -f environment.yml
```

git clone https://github.com/google/sentencepiece.git
# Build and Install SentencePiece

# install fp16 support
# fp16 with opt_level O2 gives the exact same precision but much faster and with less memory

# make sure to install proper bare metal cuda 
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
wget http://developer.download.nvidia.com/compute/cuda/10.0/Prod/patches/1/cuda_10.0.130.1_linux.run

export CUDA_HOME=/usr/local/cuda-10.0

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# run corpus.ipynb

dd if=./data/russian_corpus.txt count=10 bs=1G > ./data/russian_corpus_for_vocab.txt
spm_train --input=./data/russian_corpus_for_vocab.txt --model_prefix=bpe/m50 --vocab_size=50257 --user_defined_symbols='<|endoftext|>','<|конец|>','<|n|>'

spm_train --input=./tmp/russian_corpus_for_vocab.txt --model_prefix=bpe/m50 --vocab_size=50257 --user_defined_symbols='<|n|>'

export TRAIN_FILE=./data/russian.txt
export CUDA_VISIBLE_DEVICES=3

python run_lm_finetuning.py \
    --output_dir=output3 \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size=8 \
    --save_steps=10000 \
    --logging_steps=1 \
    --fp16 \
    --fp16_opt_level O2 \
    --warmup_steps 100 \
    --learning_rate 2e-3 \
    --overwrite_output_dir

python run_lm_finetuning.py \
    --output_dir=output3 \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size=8 \
    --save_steps=10000 \
    --logging_steps=1 \
    --fp16 \
    --fp16_opt_level O2 \
    --warmup_steps 100 \
    --learning_rate 2e-3 \
    --overwrite_output_dir \
    --tokenizer_class SPEncoder \
    --tokenizer_name bpe/m50.model

python run_lm_finetuning.py \
    --output_dir=output3 \
    --model_type=gpt2 \
    --model_name_or_path=output3 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size=8 \
    --save_steps=10000 \
    --logging_steps=1 \
    --fp16 \
    --fp16_opt_level O2 \
    --warmup_steps 100 \
    --learning_rate 2e-3 \
    --overwrite_output_dir \
    --tokenizer_class SPEncoder \
    --tokenizer_name bpe/m50.model

##############################

python run_lm_finetuning.py \
    --output_dir=output3 \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size=3 \
    --save_steps=10000 \
    --logging_steps=1 \
    --fp16 \
    --fp16_opt_level O2 \
    --warmup_steps 100 \
    --learning_rate 5e-4 \
    --overwrite_output_dir

python run_lm_finetuning.py \
    --output_dir=output3 \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-large \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size=1 \
    --save_steps=10000 \
    --logging_steps=1 \
    --fp16 \
    --fp16_opt_level O2 \
    --warmup_steps 100 \
    --learning_rate 2e-4 \
    --overwrite_output_dir

