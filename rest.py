from tendo import singleton
me = singleton.SingleInstance()

from run_generation import sample_sequence
from yt_encoder import YTEncoder
from transformers import GPT2LMHeadModel
import threading
import regex as re

import logging

logging.basicConfig(filename="rest.log", level=logging.INFO)
logger = logging.getLogger(__name__)

import yaml
cfg = yaml.safe_load(open('rest_config.yaml'))
device = cfg['device']
model_path = cfg['model_path']

tokenizer = YTEncoder.from_pretrained(model_path)

model = GPT2LMHeadModel.from_pretrained(model_path)
model.to(device)
model.eval()

poetry_model = GPT2LMHeadModel.from_pretrained(cfg['poetry_path'])
poetry_model.to(device)
poetry_model.eval()

def get_sample(model, prompt, length:int, num_samples:int, allow_linebreak:bool):
    logger.info("*" * 200)
    logger.info(prompt)
   
    filter_n = tokenizer.encode('\n')[-1:]
    filter_single = [tokenizer.sp.unk_id()] 
    filter_single += [] if allow_linebreak else filter_n

    context_tokens = tokenizer.encode(prompt)
    out = sample_sequence(
        model=model,
        context=context_tokens,
        length=length,
        temperature=1,
        top_k=0,
        top_p=0.9,
        device=device,
        filter_single=filter_single,
        filter_double=filter_n,
        num_samples=num_samples,
    ).to('cpu')

    prompt = tokenizer.decode(context_tokens)
    len_prompt = len(prompt)
   
    replies = [out[item, :].tolist() for item in range(len(out))]
    text = [tokenizer.decode(item)[len_prompt:] for item in replies]
    reg_text = [re.match(r'[\w\W]*[\.!?]\n', item) for item in text]
    result = [reg_item[0] if reg_item else item  for reg_item, item in zip(reg_text,text)]
    logger.info("=" * 200)
    logger.info(result)
    return result

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Schema

app = FastAPI(title="Russian GPT-2", version="0.1",)
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

lock = threading.RLock()

class Prompt(BaseModel):
    prompt:str = Schema(..., max_length=3000, title='Model prompt')
    length:int = Schema(15, ge=1, le=500, title='Number of tokens generated in each sample')
    num_samples:int = Schema(3, ge=1, le=10, title='Number of samples generated')
    allow_linebreak:bool = Schema(False, title='Allow linebreak in a sample')

@app.post("/" + model_path + "/")
def gen_sample(prompt: Prompt):
    with lock:
        return {"replies": get_sample(model, prompt.prompt, prompt.length, prompt.num_samples, prompt.allow_linebreak)}

class PromptPoetry(BaseModel):
    prompt:str = Schema(..., max_length=3000, title='Model prompt')
    length:int = Schema(15, ge=1, le=500, title='Number of tokens generated in each sample')

@app.post("/gpt2_poetry/")
def gen_sample(prompt: PromptPoetry):
    with lock:
        return {"replies": get_sample(poetry_model, prompt.prompt, prompt.length, 1, True)}

@app.get("/health")
def healthcheck():
    return True
