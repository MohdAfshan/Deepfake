from fastapi import FastAPI
from inference import reset, step

app = FastAPI()


@app.post("/reset")
def reset_env(config: dict = {}):
    return reset(config)


@app.post("/step")
def step_env(action: dict):
    return step(action)


@app.get("/")
def health():
    return {"status": "running"}
