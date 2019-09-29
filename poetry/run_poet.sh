#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate gpt
cd /home/u/ru_transformers/poetry
python scheduler.py
