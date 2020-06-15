import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from run_generation import sample_sequence
from yt_encoder import YTEncoder
from transformers import GPT2LMHeadModel
import threading
import regex as re

from os import environ
device = environ.get('DEVICE', 'cuda:0')

flavor_id = device + environ.get('INSTANCE', ':0')
from tendo import singleton
me = singleton.SingleInstance(flavor_id=flavor_id)

import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename=f"logs/{hash(flavor_id)}.log", level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = 'gpt2/medium'

tokenizer = YTEncoder.from_pretrained(model_path)

model = GPT2LMHeadModel.from_pretrained(model_path)
model.to(device)
model.eval()

poetry_model = GPT2LMHeadModel.from_pretrained(model_path + '/poetry')
poetry_model.to(device)
poetry_model.eval()

from apex import amp
[model, poetry_model] = amp.initialize([model, poetry_model], opt_level='O2')

def get_sample(model, prompt, length:int, num_samples:int, allow_linebreak:bool):
    logger.info(prompt)
   
    filter_n = tokenizer.encode('\n')[-1:]
    filter_single = [1] + tokenizer.encode('[')[-1:] + tokenizer.encode('(')[-1:]
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
    reg_text2 = [re.match(r'[\w\W]*[\.!?]', item) for item in text]
    result = [reg_item[0] if reg_item else reg_item2[0] if reg_item2 else item for reg_item, reg_item2, item in zip(reg_text, reg_text2, text)]
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
    length:int = Schema(15, ge=1, le=60, title='Number of tokens generated in each sample')
    num_samples:int = Schema(3, ge=1, le=5, title='Number of samples generated')
    allow_linebreak:bool = Schema(False, title='Allow linebreak in a sample')

@app.post("/" + model_path + "/")
def gen_sample(prompt: Prompt):
    with lock:
        return {"replies": get_sample(model, prompt.prompt, prompt.length, prompt.num_samples, prompt.allow_linebreak)}

class PromptPoetry(BaseModel):
    prompt:str = Schema(..., max_length=3000, title='Model prompt')
    length:int = Schema(15, ge=1, le=150, title='Number of tokens generated in each sample')

@app.post("/gpt2_poetry/")
def gen_sample(prompt: PromptPoetry):
    with lock:
        return {"replies": get_sample(poetry_model, prompt.prompt, prompt.length, 1, True)}

@app.get("/health")
def healthcheck():
    return True
