# Russian GPT-2 

The training is not finished yet.

### 1. Download a fb2 library 

Main [link](https://booktracker.org/viewtopic.php?t=1198)

For finetuning [first](https://booktracker.org/viewtopic.php?t=43884) [second](https://booktracker.org/viewtopic.php?t=73891) [Dostoyevskiy](https://booktracker.org/viewtopic.php?t=7594) [Tolstoy](https://booktracker.org/viewtopic.php?t=8109) [Pushkin](https://booktracker.org/viewtopic.php?t=13615) [Bulgakov](https://booktracker.org/viewtopic.php?t=4397) [Gogol](https://booktracker.org/viewtopic.php?t=17643) [Pelevin](https://booktracker.org/viewtopic.php?t=48699)


### 2. Install dependencies
```bash
sudo xargs -a apt.txt apt install
conda env create -f environment.yml
```
### 3. Build and Install SentencePiece

Follow instructions here https://github.com/google/sentencepiece

### 4. Install fp16 support 

Mixed precision training with opt_level O2 gives the exact same loss but much faster and with less memory. The downside - APEX with O2 doesnt work with `DataParallel` yet, see https://github.com/NVIDIA/apex/issues/227

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
export TRAIN_FILE=./data/full

# GPT-2 124M, final perplexity ?

export CUDA_VISIBLE_DEVICES=1
export MODEL_SIZE=gpt2
export OUTPUT=output_s
export BS=8
export LR=5e-5

# GPT-2 355M, final perplexity 18.99?

export CUDA_VISIBLE_DEVICES=2
export MODEL_SIZE=gpt2-medium
export OUTPUT=output_m
export BS=3
export LR=3e-5

# GPT-2 774M, final perplexity 21.09?

export CUDA_VISIBLE_DEVICES=3
export MODEL_SIZE=gpt2-large
export OUTPUT=output_l
export BS=1
export LR=1e-5

# training script

python run_lm_finetuning.py \
    --output_dir=$OUTPUT \
    --model_type=gpt2 \
    --model_name_or_path=$MODEL_SIZE \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size $BS \
    --save_steps=10000 \
    --logging_steps=1 \
    --fp16 \
    --fp16_opt_level O2 \
    --warmup_steps 1000 \
    --learning_rate $LR \
    --tokenizer_class SPEncoder \
    --tokenizer_name bpe/m50.model \
    --do_eval \
    --evaluate_during_training \
    --eval_steps 1000 \
    --eval_data_file=./data/classic/valid

# My dataset is 230Gb and it doesn't fit in RAM, so each epoch is a random sample from it. That is why the loop.
while true
do
    python run_lm_finetuning.py \
        --output_dir=$OUTPUT \
        --model_type=gpt2 \
        --model_name_or_path=$OUTPUT \
        --do_train \
        --train_data_file=$TRAIN_FILE \
        --per_gpu_train_batch_size $BS \
        --save_steps=10000 \
        --logging_steps=1 \
        --fp16 \
        --fp16_opt_level O2 \
        --warmup_steps 100 \
        --learning_rate $LR \
        --overwrite_output_dir \
        --tokenizer_class SPEncoder \
        --tokenizer_name bpe/m50.model \
        --do_eval \
        --evaluate_during_training \
        --eval_steps 1000 \
        --eval_data_file=./data/classic/valid

    sleep 1
done

```

### 8. Deploy your model

