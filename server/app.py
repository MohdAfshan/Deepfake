from fastapi import FastAPI
from inference import reset, step
import uvicorn

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

# 1️⃣ Add the main function
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)

# 2️⃣ Make it executable
if __name__ == "__main__":
    main()
