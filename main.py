from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Carregando modelo...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.eval()

print("Modelo carregado!")

class Message(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/chat")
def chat(msg: Message):
    prompt = f"""
Você é um assistente útil.

Usuário: {msg.text}
Assistente:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response}
