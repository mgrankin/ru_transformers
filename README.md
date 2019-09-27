# Russian GPT-2 

1. Download librusec library 

http://trec.to/viewtopic.php?p=60

1. Install dependencies
```bash
sudo xargs -a apt.txt apt install
conda env create -f environment.yml
```
1. Build and Install SentencePiece

Use instructions here https://github.com/google/sentencepiece
1. Install fp16 support 

Mixed precision training with opt_level O2 gives the exact same loss but much faster and with less memory.

1.1 Make sure to install proper bare metal cuda. 
```bash
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux -O nvidia.run
chmod +x nvidia.run
sudo ./nvidia.run
```
1.1 Apex

```bash
export CUDA_HOME=/usr/local/cuda-10.0
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

1. Prepare the dataset files - run corpus/corpus.ipynb

1. Create dictionary for the SentencePiece tokenizer
```bash
spm_train --input=./corpus/tmp/russian_corpus_for_vocab.txt --model_prefix=bpe/m50 --vocab_size=50257 --user_defined_symbols='<|n|>'
```

1. Train your model!

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

```
