from tendo import singleton
me = singleton.SingleInstance()

from run_generation import sample_sequence
from sp_encoder import SPEncoder
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

tokenizer = SPEncoder.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.to(device)
model.eval()

def get_sample(prompt, length:int, num_samples:int, allow_linebreak:bool):
    logger.info("*" * 200)
    logger.info(prompt)

    model.to(device)
    model.eval()
    
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
    replies = [out[item, len(context_tokens):].tolist() for item in range(len(out))]
    text = [tokenizer.decode(item) for item in replies]
    reg_text = [re.match(r'[\w\W]*[\.!?]\n', item) for item in text]
    result = [reg_item[0] if reg_item else item  for reg_item, item in zip(reg_text,text)]
    logger.info("=" * 200)
    logger.info(result)
    return result

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    prompt:str
    length:int=15
    num_samples:int=3
    allow_linebreak:bool=False

@app.post("/" + model_path + "/")
def gen_sample(prompt: Prompt):
    prompt.num_samples = min(prompt.num_samples, 10)
    prompt.length = min(prompt.length, 500)
    with lock:
        return {"replies": get_sample(prompt.prompt, prompt.length, prompt.num_samples, prompt.allow_linebreak)}

