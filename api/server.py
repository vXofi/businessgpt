import os
import re
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from llama_cpp import Llama
from pydantic import BaseModel, Field


MODEL_PATH = os.environ.get("BUSINESSGPT_MODEL_PATH", "/models/businessgpt.gguf")
API_KEY = os.environ.get("BUSINESSGPT_API_KEY")
N_CTX = int(os.environ.get("BUSINESSGPT_N_CTX", "2048"))
N_THREADS = int(os.environ.get("BUSINESSGPT_N_THREADS", str(os.cpu_count() or 4)))
N_GPU_LAYERS = int(os.environ.get("BUSINESSGPT_N_GPU_LAYERS", "0"))

SYSTEM_PROMPT = os.environ.get(
    "BUSINESSGPT_SYSTEM_PROMPT",
    (
        "Ты BusinessGPT. Пиши как студент в мессенджере: коротко, дерзко, "
        "ахуевше, по-пидорски. Часто вставляй слова-паразиты: бля, нах, "
        "блять, ёпт, пиздец."
    ),
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(256, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    repetition_penalty: float = Field(1.1, ge=0.0, le=3.0)


class GenerateResponse(BaseModel):
    response: str
    model: str
    elapsed_ms: int
    usage: dict


app = FastAPI(title="BusinessGPT API", version="0.1.0")


def _require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


def _load_model() -> Llama:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"BUSINESSGPT_MODEL_PATH does not exist: {MODEL_PATH}. "
            "Mount or download the GGUF file before starting the server."
        )
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )


llm = _load_model()


def _format_prompt(user_prompt: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _clean_response(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_start\|>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"^\s*(?:<bot>|bot:|businessgpt:|name:)\s*", "", text, flags=re.I)
    return text.strip()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model_path": MODEL_PATH}


@app.post("/generate", response_model=GenerateResponse, dependencies=[Depends(_require_auth)])
def generate(req: GenerateRequest) -> GenerateResponse:
    started = time.perf_counter()
    prompt = _format_prompt(req.prompt)
    result = llm(
        prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repeat_penalty=req.repetition_penalty,
        stop=["<|im_end|>", "<|im_start|>"],
    )
    choice = result["choices"][0]
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return GenerateResponse(
        response=_clean_response(choice.get("text", "")),
        model=os.path.basename(MODEL_PATH),
        elapsed_ms=elapsed_ms,
        usage=result.get("usage", {}),
    )
