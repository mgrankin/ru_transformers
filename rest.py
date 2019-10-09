from run_generation import sample_sequence
from sp_encoder import SPEncoder
from transformers import GPT2LMHeadModel
import threading
import regex as re

import logging

logging.basicConfig(filename="stihbot.log", level=logging.INFO)
logger = logging.getLogger(__name__)

device="cuda"
path = 'gpt2/medium'

lock = threading.RLock()

def get_sample(prompt, model, tokenizer, device, length:int, num_samples:int, allow_linebreak:bool):
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

tokenizer = SPEncoder.from_pretrained(path)
model = GPT2LMHeadModel.from_pretrained(path)
model.to(device)
model.eval()

from fastapi import FastAPI

app = FastAPI(title="Russian GPT-2", version="0.1",)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/gpt2-large/{prompt}")
def gen_(prompt:str, length:int=5, num_samples:int=3, allow_linebreak:bool=False):
    with lock:
        return {"replies": get_sample(prompt, model, tokenizer, device, length, num_samples, allow_linebreak)}

