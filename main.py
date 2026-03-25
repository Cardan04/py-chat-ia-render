from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

MODEL_NAME = "distilgpt2"

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is None:
        print("🔄 Carregando modelo...")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        model.eval()

        print("✅ Modelo carregado!")

class Message(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/chat")
def chat(msg: Message):
    load_model()

    # Prompt otimizado para DistilGPT2
    prompt = f"""
Answer the question clearly and helpfully.

Question: {msg.text}
Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # limpa resposta
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    return {"response": response}
