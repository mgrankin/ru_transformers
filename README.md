# Russian GPT-2 

### 1. Download a fb2 library 

Main
https://booktracker.org/viewtopic.php?t=1198

For finetuning
https://booktracker.org/viewtopic.php?t=43884

https://booktracker.org/viewtopic.php?t=73891

### 2. Install dependencies
```bash
sudo xargs -a apt.txt apt install
conda env create -f environment.yml
```
### 3. Build and Install SentencePiece

Follow instructions here https://github.com/google/sentencepiece

### 4. Install fp16 support 

Mixed precision training with opt_level O2 gives the exact same loss but much faster and with less memory.

#### 4.1 Make sure to install proper bare metal cuda. 
```bash
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux -O nvidia.run
chmod +x nvidia.run
sudo ./nvidia.run
```
#### 4.2 Apex

```bash
export CUDA_HOME=/usr/local/cuda-10.0
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### 5. Prepare the dataset files 
Use `corpus/corpus.ipynb` on your dataset.

### 6. Create vocabulary for the SentencePiece tokenizer

You can skip this step if you want only to finetune the model with the existing vocab.

```bash
spm_train --input=./corpus/tmp/russian_corpus_for_vocab.txt --model_prefix=bpe/m50 --vocab_size=50257 --user_defined_symbols='<|n|>'
```

### 7. Train your model!
``` bash
cd ru_transformers
conda activate gpt
export TRAIN_FILE=./dataset
export CUDA_VISIBLE_DEVICES=1

##############################
# small

python run_lm_finetuning.py \
    --output_dir=output_s \
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
    --learning_rate 2e-4 \
    --overwrite_output_dir \
    --tokenizer_class SPEncoder \
    --tokenizer_name bpe/m50.model

while true
do
    python run_lm_finetuning.py \
        --output_dir=output_s \
        --model_type=gpt2 \
        --model_name_or_path=output_s \
        --do_train \
        --train_data_file=$TRAIN_FILE \
        --per_gpu_train_batch_size=8 \
        --save_steps=10000 \
        --logging_steps=1 \
        --fp16 \
        --fp16_opt_level O2 \
        --warmup_steps 100 \
        --learning_rate 2e-4 \
        --overwrite_output_dir \
        --tokenizer_class SPEncoder \
        --tokenizer_name bpe/m50.model
    sleep 1
done


##############################
# medium

python run_lm_finetuning.py \
    --output_dir=output_m \
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
    --learning_rate 5e-5 \
    --overwrite_output_dir \
    --tokenizer_class SPEncoder \
    --tokenizer_name bpe/m50.model

while true
do
    python run_lm_finetuning.py \
        --output_dir=output_m \
        --model_type=gpt2 \
        --model_name_or_path=output_m \
        --do_train \
        --train_data_file=$TRAIN_FILE \
        --per_gpu_train_batch_size=3 \
        --save_steps=10000 \
        --logging_steps=1 \
        --fp16 \
        --fp16_opt_level O2 \
        --warmup_steps 100 \
        --learning_rate 5e-5 \
        --overwrite_output_dir \
        --tokenizer_class SPEncoder \
        --tokenizer_name bpe/m50.model
    sleep 1
done

##############################
# large

python run_lm_finetuning.py \
    --output_dir=output_l \
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
    --learning_rate 2e-5 \
    --overwrite_output_dir \
    --tokenizer_class SPEncoder \
    --tokenizer_name bpe/m50.model

while true
do
    python run_lm_finetuning.py \
        --output_dir=output_l \
        --model_type=gpt2 \
        --model_name_or_path=output_l \
        --do_train \
        --train_data_file=$TRAIN_FILE \
        --per_gpu_train_batch_size=1 \
        --save_steps=10000 \
        --logging_steps=1 \
        --fp16 \
        --fp16_opt_level O2 \
        --warmup_steps 100 \
        --learning_rate 2e-5 \
        --overwrite_output_dir
        --overwrite_output_dir \
        --tokenizer_class SPEncoder \
        --tokenizer_name bpe/m50.model
    sleep 1
done

```
