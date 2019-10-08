from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/gpt2-large/{prompt}")
def gen_(prompt:str):
    return {"replies": [prompt.upper(), prompt.lower()]}

