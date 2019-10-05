
cd ru_transformers
conda activate gpt
export TRAIN_FILE=./data/classic
export CUDA_VISIBLE_DEVICES=0

python run_lm_finetuning.py \
    --output_dir=output_cs \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size=3 \
    --save_steps=10000 \
    --logging_steps=1 \
    --fp16 \
    --fp16_opt_level O2 \
    --warmup_steps 100 \
    --learning_rate 3e-5 \
    --overwrite_output_dir \
    --tokenizer_class SPEncoder \
    --tokenizer_name bpe/m50.model \
    --do_eval \
    --evaluate_during_training \
    --eval_steps 1000 \
    --eval_data_file=./data/classic/valid

while true
do
    python run_lm_finetuning.py \
        --output_dir=output_cs \
        --model_type=gpt2 \
        --model_name_or_path=output_cs \
        --do_train \
        --train_data_file=$TRAIN_FILE \
        --per_gpu_train_batch_size=3 \
        --save_steps=10000 \
        --logging_steps=1 \
        --fp16 \
        --fp16_opt_level O2 \
        --warmup_steps 100 \
        --learning_rate 3e-5 \
        --overwrite_output_dir \
        --tokenizer_class SPEncoder \
        --tokenizer_name bpe/m50.model \
        --do_eval \
        --evaluate_during_training \
        --eval_steps 1000 \
        --eval_data_file=./data/classic/valid

    sleep 1
done

