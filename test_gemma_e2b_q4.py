"""Memory + speed sanity-check for Gemma 4 E2B Q4_K_M GGUF on CPU.

Goal: verify the model fits prod constraints (CPU only, low RAM) BEFORE
committing to migrating the whole pipeline (training, eval, dpo) from
Qwen3.5-2B to Gemma E2B.

Compared to Qwen3.5-2B (~1.3 GB Q4_K_M):
  - Gemma E2B Q4_K_M = ~3.46 GB on disk (per-layer embeddings inflate file)
  - Effective compute ≈ 2.3B params (similar latency on CPU)
  - Better Russian / multilingual priors (built that way by Google)

Defaults to bartowski's non-abliterated GGUF — abliterated GGUF isn't published
yet, but architecture is identical so memory/speed numbers transfer 1:1.

Usage:
    python3 test_gemma_e2b_q4.py
    GGUF_REPO=foo/bar GGUF_FILE=model.gguf python3 test_gemma_e2b_q4.py
    N_CTX=4096 N_THREADS=4 python3 test_gemma_e2b_q4.py

Requirements: pip install llama-cpp-python huggingface_hub psutil
"""

import os
import sys
import time
import threading
from pathlib import Path

import psutil
from huggingface_hub import hf_hub_download

# ── Config ──────────────────────────────────────────────────
GGUF_REPO = os.environ.get("GGUF_REPO", "bartowski/google_gemma-4-E2B-it-GGUF")
GGUF_FILE = os.environ.get("GGUF_FILE", "google_gemma-4-E2B-it-Q4_K_M.gguf")
N_CTX     = int(os.environ.get("N_CTX", "2048"))      # llama.cpp context window
N_THREADS = int(os.environ.get("N_THREADS", "0")) or None  # None = auto
N_RUNS    = int(os.environ.get("N_RUNS", "5"))        # how many test prompts

# Prod target: model + KV cache + buffers must fit under this
RAM_TARGET_MB = int(os.environ.get("RAM_TARGET_MB", "4096"))

SYSTEM_PROMPT = (
    "Ты BusinessGPT. Пиши как студент в мессенджере: коротко, дерзко, ахуевше, "
    "по-пидорски. Часто вставляй слова-паразиты: бля, нах, блять, ёпт, пиздец."
)

# 10-message contexts — matches prod chat window size
TEST_CONTEXTS = [
    [
        "кто сделал 15 практику?", "я не делал", "я тоже нет",
        "может кто-нибудь уже начнёт", "ну давайте", "мне лень",
        "аналогично", "я могу попробовать", "давай мельник", "наш герой",
    ],
    [
        "го в кс", "я не могу", "почему", "мне завтра сдавать",
        "чё сдавать", "матан", "ахахах", "не смешно", "ладно удачи", "спасибо",
    ],
    [
        "когда дедлайн по курсовой?", "вроде завтра", "ЧТОООО", "да ладно тебе",
        "я даже не начинал", "аналогично бля", "ну пиздец", "может продлят",
        "не продлят", "нам конец",
    ],
    [
        "кто тут самый умный?", "я", "нет я", "вы оба тупые", "а ты кто такой",
        "я тот кто матан сдал на 5", "пиздишь", "зуб даю", "ахахахахах", "сосал?",
    ],
    [
        "го в пятницу бухать", "я за", "у кого хата?", "можно ко мне",
        "скинемся на алко?", "по 500?", "норм", "кто ещё идёт?",
        "я позову пацанов", "топ будет вечер",
    ],
]


def rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


class PeakMonitor:
    """Background sampler — captures max RSS during a code block."""
    def __init__(self, hz=20):
        self._stop = threading.Event()
        self._peak = rss_mb()
        self._hz = hz
        self._thread = None

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                self._peak = max(self._peak, rss_mb())
                time.sleep(1.0 / self._hz)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()

    @property
    def peak(self):
        return self._peak


def format_gemma_prompt(system, ctx_lines):
    """Gemma format: system embedded into first user turn, no separate system role.

    <start_of_turn>user
    {system}\n\n{user content}<end_of_turn>
    <start_of_turn>model
    """
    user_text = system + "\n\n" + "\n".join(ctx_lines)
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"


def main():
    print("=" * 64)
    print(f"Gemma 4 E2B Q4_K_M memory + speed test")
    print(f"  GGUF: {GGUF_REPO}/{GGUF_FILE}")
    print(f"  n_ctx={N_CTX}  n_threads={N_THREADS or 'auto'}  n_runs={N_RUNS}")
    print(f"  RAM target: ≤ {RAM_TARGET_MB} MB")
    print("=" * 64)

    rss_initial = rss_mb()
    print(f"\nRSS at start: {rss_initial:.0f} MB")

    # ── Download ────────────────────────────────────────────
    print(f"\n[1/3] Downloading {GGUF_FILE} (~3.5 GB, may take a few min)...")
    t0 = time.time()
    try:
        gguf_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"  Set GGUF_REPO and GGUF_FILE env vars to override.")
        sys.exit(1)
    file_mb = os.path.getsize(gguf_path) / 1024 / 1024
    print(f"  Done in {time.time()-t0:.0f}s. Size on disk: {file_mb:.0f} MB")

    # ── Load ────────────────────────────────────────────────
    print(f"\n[2/3] Loading via llama.cpp (CPU only)...")
    rss_before = rss_mb()
    t0 = time.time()
    from llama_cpp import Llama
    llm = Llama(
        model_path=gguf_path,
        n_ctx=N_CTX,
        n_gpu_layers=0,        # match prod: CPU only
        n_threads=N_THREADS,
        verbose=False,
    )
    t_load = time.time() - t0
    rss_loaded = rss_mb()
    print(f"  Load time: {t_load:.1f}s")
    print(f"  RSS before: {rss_before:.0f} MB → after: {rss_loaded:.0f} MB")
    print(f"  Δ from load: +{rss_loaded - rss_before:.0f} MB")

    # ── Inference benchmark ─────────────────────────────────
    print(f"\n[3/3] Generation: {N_RUNS} runs, 10-msg contexts, max_tokens=128")
    print()

    total_tokens = 0
    total_gen_time = 0
    sample_outputs = []

    with PeakMonitor() as mon:
        for i in range(min(N_RUNS, len(TEST_CONTEXTS))):
            ctx = TEST_CONTEXTS[i]
            prompt = format_gemma_prompt(SYSTEM_PROMPT, ctx)

            t0 = time.time()
            out = llm(
                prompt,
                max_tokens=128,
                temperature=0.95,
                top_p=0.9,
                top_k=50,
                repeat_penalty=1.1,
                stop=["<end_of_turn>", "<start_of_turn>"],
            )
            t_gen = time.time() - t0

            text = out["choices"][0]["text"].strip()
            usage = out.get("usage") or {}
            n_out_tok = usage.get("completion_tokens") or len(text.split())

            tok_per_s = n_out_tok / t_gen if t_gen > 0 else 0
            print(f"  [{i+1}/{N_RUNS}] {t_gen:5.1f}s  {n_out_tok:3d} tok  {tok_per_s:5.1f} tok/s  "
                  f"RSS={rss_mb():.0f} MB")
            print(f"        ctx tail: ...{ctx[-2]} | {ctx[-1]}")
            print(f"        response: {text[:140]!r}")
            print()

            total_tokens += n_out_tok
            total_gen_time += t_gen
            sample_outputs.append(text)

    peak_rss = mon.peak
    avg_tok_s = total_tokens / total_gen_time if total_gen_time > 0 else 0

    # ── Summary ─────────────────────────────────────────────
    print("=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"  Disk:          {file_mb:.0f} MB")
    print(f"  RSS loaded:    {rss_loaded:.0f} MB")
    print(f"  Peak RSS:      {peak_rss:.0f} MB  (during {N_RUNS} generations)")
    print(f"  Headroom:      +{peak_rss - rss_loaded:.0f} MB above load (KV cache, buffers)")
    print(f"  Avg speed:     {avg_tok_s:.1f} tok/s")
    print(f"  Total gen time:{total_gen_time:.1f}s for {total_tokens} tokens")
    print()

    fits = peak_rss <= RAM_TARGET_MB
    status = "PASS ✓" if fits else "FAIL ✗"
    print(f"  Prod RAM target ({RAM_TARGET_MB} MB): {status}  ({peak_rss:.0f} MB)")
    if not fits:
        print(f"  Over by {peak_rss - RAM_TARGET_MB:.0f} MB — try Q4_K_S "
              f"(~3.4 GB on disk) or migrate is risky.")

    return 0 if fits else 1


if __name__ == "__main__":
    sys.exit(main())
