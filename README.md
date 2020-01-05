# Russian GPT-2 

# 1. I just want to play with your models

You can try poetry with Telegram chat bot ```@NeuroPoetBot```

You can try writing with the model here https://porfirevich.ru

# 2. What are results?

Your perplexity will be different, depending on the tokenizer, the vocab and the dataset. The better your tokenizer the worse your perplexity, actually.

Values in the table are perplexity on the validation set.

Huge dataset

GPT-2                           | Small, 124M. BS 64 | Medium, 355M. BS 32   | 
---                                  | -- | ---                          | 
Unfreeze 0, LR 24e-4         | 80 epoch, 85-90 | 80 epoch,  81-85                         |   
Unfreeze 0, LR 3e-4          | 80 epoch, 75-76 | 100 epoch,  64-65                         |   
Unfreeze 0, LR 6e-5          | 80 epoch, 73-73.5 | 40 epoch,  63-63.5                         |   
Unfreeze 1, LR 3e-4          | 118 epoch, 51-52 | 142 epoch, 42.3-43.7                    |   
Unfreeze 1, LR 6e-5         | 80 epoch, 49-49.5 | 40 epoch, 41.-41.6                     |   
Unfreeze 2, LR 3e-4          | 70 epoch, 45.5 |  68 epoch, 37.2-38.6                        |   
Unfreeze 2, LR 6e-5         | 200 epoch, 41.18-42.19 | 87 epoch, 35.4-35.9                          |   
Unfreeze 7, LR 3e-4          | 90 epoch, 35.3 - 35.9 | 163 epoch, 28.6-29.6                          |   
Unfreeze 7, LR 6e-5         | 88 epoch, 32.6-33. | 90 epoch, 27.2-27.5                          |   
Unfreeze -1 (all), LR 6e-5         | 160 epoch, 30.5-30.9 | 163 epoch, 23.8-24.15                          |   

Classics dataset. 
It's only 500Mb and GPT-2 overfits it pretty fast. 

GPT-2                           | Small, 124M  | Medium, 355M   | 
---                                  | -- | ---                          | 
Unfreeze -1 (all)         | 28 epoch, 26.22 | 7 epoch, **20.9722**                          |     

Poetry dataset

GPT-2                           | Small, 124M  | Medium, 355M   | 
---                                  | -- | ---                          | 
Unfreeze -1 (all)         | 25 epoch, 26.22 | 7 epoch, 48.36                         |     

Pelevin dataset

GPT-2                           | Small, 124M  | Medium, 355M   | 
---                                  | -- | ---                          | 
Unfreeze -1 (all)         | 5 epoch, 44.55 | 3 epoch, 33.38                          |    

# 3. I'd like to download your models

```bash
pip install awscli
aws s3 sync --no-sign-request s3://models.dobro.ai/gpt2/ru/unfreeze_all gpt2
```

Folders with ```s_``` prefix contain Small (124M) model, ```m_``` - for Medium (355M) model. 

# 4. I've got a small Russian dataset and I want to finetune your model on it

Download the models (intructions above), choose the model and put it in your output folder. Use validation set and be careful with overfitting. On small dataset it will overfit very fast - 3-7 epoch. Follow instructions below, except you don't need to train you tokenization dictionary, because you already have one.

# 5. I've got a big dataset on my lang and I want to train GPT-2 on it

I'd suggest that if you don't have a bunch of GPU's you should consider renting a Google TPU. On my Nvidia Titan RTX an epoch takes 70 minutes and the same epoch takes 12.5 minutes on TPU v3-8. I've used fp16 on GPU, but I can't use bfloat16 on TPU, because it's training poorly on bfloat16 at the moment (it could have been 8 minutes if implemented properly).

You can ask for access to Google's TensorFlow Research Cloud and use TPUs for free for one month.

In the process, I've switched tokenization library from SentencePiece to YTTM. YTTM is better (10% smaller files) and much faster. If you for some reason want to use SentencePiece then the code is here, just change the tokenizer in the command line.

First, the GPT-2 model will learn Russian on a huge dataset (230 GB), and then it will learn good Russian on the Russian classical literature (500 MB). I use progressive layer unfreezing to use transfer training. Validation set is the correspondence between Leo Tolstoy with young Mahatma Gandhi.

### 5.1. Download a fb2 library 

Main [link](https://booktracker.org/viewtopic.php?t=1198)

For finetuning [first](https://booktracker.org/viewtopic.php?t=43884) [second](https://booktracker.org/viewtopic.php?t=73891) [Dostoyevskiy](https://booktracker.org/viewtopic.php?t=7594) [Tolstoy](https://booktracker.org/viewtopic.php?t=8109) [Pushkin](https://booktracker.org/viewtopic.php?t=13615) [Bulgakov](https://booktracker.org/viewtopic.php?t=4397) [Gogol](https://booktracker.org/viewtopic.php?t=17643) [Pelevin](https://booktracker.org/viewtopic.php?t=48699)


### 5.2. Install dependencies
```bash
sudo xargs -a apt.txt apt install
conda env create -f environment.yml
```
### 5.3. Build and Install SentencePiece (skip if use YTTM)

Follow instructions here https://github.com/google/sentencepiece

### 5.4. Prepare the dataset files 
Use `corpus/corpus.ipynb` on your dataset.

### 5.5. Create vocabulary for the YTTM (and SentencePiece) tokenizer

You can skip this step if you want only to finetune the model with the existing vocab.

```bash
yttm bpe --data ./corpus/tmp/russian_corpus_for_vocab.txt --model bpe/yt.model --vocab_size 50257 --coverage 0.9999

# SentencePiece
spm_train --input=./corpus/tmp/russian_corpus_for_vocab.txt --model_prefix=bpe/m50 --vocab_size=50257 --user_defined_symbols='<|n|>'
```

### 5.6. If you want to use Google TPU, go here https://github.com/mgrankin/ru_transformers/tree/master/tpu

### 5.7. Install fp16 support 

Mixed precision training with opt_level O2 gives the exact same loss but much faster and with less memory. The downside - APEX with O2 doesnt work with `DataParallel` yet, see https://github.com/NVIDIA/apex/issues/227

#### 5.7.1 Make sure to install proper bare metal cuda. 
```bash
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux -O nvidia.run
chmod +x nvidia.run
sudo ./nvidia.run
```
#### 5.7.2 Apex

```bash
export CUDA_HOME=/usr/local/cuda-10.0
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### 5.8. Train your model!
``` bash
cd ru_transformers
conda activate gpt
export TRAIN_FILE=./data/classic

# GPT-2 124M, final perplexity ?

export CUDA_VISIBLE_DEVICES=1
export MODEL_SIZE=gpt2
export OUTPUT=output_yt/s
export BS=8
export LR=5e-5

# GPT-2 355M, final perplexity 18.99?

export CUDA_VISIBLE_DEVICES=2
export MODEL_SIZE=gpt2-medium
export OUTPUT=output_yt/m
export BS=3
export LR=3e-5

# GPT-2 774M, final perplexity 21.09?

export CUDA_VISIBLE_DEVICES=3
export MODEL_SIZE=gpt2-large
export OUTPUT=output_yt/l
export BS=1
export LR=1e-5

# training script

# You shouldn't use --model_name_or_path=$MODEL_SIZE if you want to start with pre-trained Russian GPT-2. If you set --model_name_or_path=gpt2 you'll start with English GPT-2.
# This step will download an English GPT-2 to the $OUTPUT and start training it.
# If you want to start from Russian GPT-2 then skip this step. Instead download the Russian GPT-2, put it to $OUTPUT manually. 
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
    --warmup_samples 16000 \
    --learning_rate $LR \
    --tokenizer_class YTEncoder \
    --tokenizer_name bpe/yt.model \
    --do_eval \
    --evaluate_during_training \
    --eval_steps 1000 \
    --eval_data_file=./data/classic/valid \
    --unfreeze_level 0

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
        --logging_steps=10 \
        --fp16 \
        --fp16_opt_level O2 \
        --warmup_samples 16000 \
        --learning_rate $LR \
        --overwrite_output_dir \
        --tokenizer_class YTEncoder \
        --tokenizer_name bpe/yt.model \
        --do_eval \
        --evaluate_during_training \
        --eval_steps 1000 \
        --eval_data_file=./data/classic/valid \
        --save_total_limit 30 \
        --num_train_epochs 10.0 \
        --unfreeze_level 0

    sleep 1
done


# with decay
python run_lm_finetuning.py \
    --output_dir=$OUTPUT \
    --model_type=gpt2 \
    --model_name_or_path=$OUTPUT \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size $BS \
    --save_steps=10000 \
    --logging_steps=10 \
    --fp16 \
    --fp16_opt_level O2 \
    --warmup_samples 16000 \
    --learning_rate $LR \
    --overwrite_output_dir \
    --tokenizer_class YTEncoder \
    --tokenizer_name bpe/yt.model \
    --do_eval \
    --evaluate_during_training \
    --eval_steps 1000 \
    --eval_data_file=./data/classic/valid \
    --save_total_limit 30 \
    --num_train_epochs 3.0 \
    --unfreeze_level 0 \
    --lr_decay

# and then repeat with unfreeze_level 1,2,3...
```

### 5.9. Save trained model

``` bash
aws s3 cp output_s/config.json s3://models.dobro.ai/gpt2/ru/small/
aws s3 cp output_s/encoder.model s3://models.dobro.ai/gpt2/ru/small/
aws s3 cp output_s/pytorch_model.bin s3://models.dobro.ai/gpt2/ru/small/
```

### 5.10. Deploy the model

``` bash
git clone https://github.com/mgrankin/ru_transformers.git
cd ru_transformers
aws s3 sync --no-sign-request s3://models.dobro.ai/gpt2/ru gpt2
conda env create -f environment.yml
conda activate gpt
uvicorn rest:app --reload --host 0.0.0.0
# crontab  DEVICE="cuda:1"
# @reboot /bin/bash -c "cd ru_transformers; git pull; source ~/.bashrc; conda activate gpt; DEVICE="cuda:1" uvicorn rest:app --reload --host 0.0.0.0"

```