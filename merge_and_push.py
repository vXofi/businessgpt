"""
Download model from HF, convert to GGUF via llama.cpp, and push to HuggingFace Hub.

Pipeline:
  1. Download from HF (LoRA or full model)
  2. Merge LoRA into base if needed
  3. Build llama.cpp (clone + cmake)
  4. Convert merged model to F16 GGUF
  5. (Optional) Generate imatrix from local train.jsonl / val.jsonl
  6. Quantize to target formats (using --imatrix when available)
  7. Push GGUFs to HF

imatrix calibration uses ~200 chat examples to weight quantization toward
the activation channels that matter most for THIS dataset. Biggest effect on
low-bit quants (Q4 and below); modest but real gain on Q8_0.
Set USE_IMATRIX=False to disable.

Requirements: pip install torch transformers peft huggingface_hub
"""

import os
import sys
import subprocess
import json

# ── Config ──────────────────────────────────────────────────
# v15+ are 9B; 2B-class versions stayed up to v14.
SOURCE_REPO = os.environ.get("SOURCE_REPO", "vXofi/businessgpt-v16-qwen3.5-9b")
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "huihui-ai/Huihui-Qwen3.5-9B-abliterated")  # only needed for LoRA
HF_GGUF_REPO = os.environ.get("HF_GGUF_REPO", "vXofi/businessgpt-v16-qwen3.5-9b-gguf")
MERGED_DIR = os.environ.get("MERGED_DIR", "merged_model_v16_9b")
LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR", "llama.cpp")
# 9B GGUF sizes: Q8_0=~9.5 GB, Q5_K_M=~6.4 GB, Q4_K_M=~5.4 GB.
# Q5_K_M is the recommended prod quant for 9B on 12 GB CPU RAM (loss <2% vs fp16).
GGUF_QUANTS = [q.strip() for q in os.environ.get("GGUF_QUANTS", "Q5_K_M,Q4_K_M").split(",") if q.strip()]

# When SOURCE_REPO is a preference adapter, set SFT_REPO to the SFT base it sits on.
# The script will apply+merge SFT first, then apply+merge the preference LoRA on top.
# Set to None for plain SFT-only repos (single LoRA over base).
SFT_REPO = os.environ.get("SFT_REPO") or None  # set to v16 SFT repo when converting a preference adapter

# F16 is an intermediate. Uploading it costs a lot of time/storage and the
# production VM should serve a quantized GGUF, so keep this off by default.
UPLOAD_F16 = os.environ.get("UPLOAD_F16", "0").lower() in {"1", "true", "yes"}
PUSH_TO_HF = os.environ.get("PUSH_TO_HF", "1").lower() in {"1", "true", "yes"}
SKIP_EXISTING_HF = os.environ.get("SKIP_EXISTING_HF", "1").lower() in {"1", "true", "yes"}
REUSE_LOCAL_GGUF = os.environ.get("REUSE_LOCAL_GGUF", "1").lower() in {"1", "true", "yes"}

# imatrix calibration
USE_IMATRIX = os.environ.get("USE_IMATRIX", "0").lower() in {"1", "true", "yes"}
# Try in order — val_examples.json (Kaggle output, v12 prompt) preferred over old jsonl.
# .json files must be a list of {"messages": [...]} objects.
# .jsonl files must have one such object per line.
CALIBRATION_SOURCES = ["val_examples.json", "train.jsonl", "val.jsonl"]
CALIBRATION_SAMPLES = 200    # number of chat examples to format
IMATRIX_CHUNKS = 100         # number of context windows llama-imatrix processes
# ─────────────────────────────────────────────────────────────


def run(cmd, **kwargs):
    print(f"  $ {cmd}")
    subprocess.check_call(cmd, shell=True, **kwargs)


def is_lora_repo(path):
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def setup_llama_cpp():
    """Clone (if missing) and build llama.cpp. Pulls latest if already cloned.

    Builds llama-quantize always; llama-imatrix only when USE_IMATRIX=True.
    """
    quantize_bin = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-quantize")
    imatrix_bin = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-imatrix")
    convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
    build_dir = os.path.join(LLAMA_CPP_DIR, "build")

    if not os.path.isdir(LLAMA_CPP_DIR):
        print("Cloning llama.cpp...")
        run(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA_CPP_DIR}")
    else:
        print("Updating llama.cpp...")
        run(f"git -C {LLAMA_CPP_DIR} pull --ff-only")

    # Install only the gguf package — do NOT use llama.cpp's requirements.txt,
    # it can downgrade transformers to a version that doesn't know Qwen3.5
    run(f"{sys.executable} -m pip install --upgrade gguf")

    targets = [(quantize_bin, "llama-quantize")]
    if USE_IMATRIX:
        targets.append((imatrix_bin, "llama-imatrix"))
    missing = [(path, t) for path, t in targets if not os.path.isfile(path)]

    if missing:
        print(f"Building llama.cpp targets: {[t for _, t in missing]}...")
        os.makedirs(build_dir, exist_ok=True)
        run(f"cmake -B {build_dir} -S {LLAMA_CPP_DIR} -DCMAKE_BUILD_TYPE=Release")
        for _, target in missing:
            run(f"cmake --build {build_dir} --target {target} -j")

    assert os.path.isfile(quantize_bin), f"llama-quantize not found at {quantize_bin}"
    assert os.path.isfile(convert_script), f"convert script not found at {convert_script}"
    if USE_IMATRIX:
        assert os.path.isfile(imatrix_bin), f"llama-imatrix not found at {imatrix_bin}"
    return convert_script, quantize_bin, imatrix_bin


def _iter_examples(src):
    """Yield {messages: [...]} objects from a .json (array) or .jsonl file."""
    if src.endswith(".jsonl"):
        with open(src, encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    else:
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
        for ex in data:
            yield ex


def build_calibration_file():
    """Build calibration text file from local chat data.

    Each chat example is formatted with the same Qwen3-style chat template the
    model was trained on, so imatrix sees the same activation patterns as
    inference. Returns the file path, or None if no source was found.
    """
    src = next((s for s in CALIBRATION_SOURCES if os.path.isfile(s)), None)
    if src is None:
        print(f"  No calibration source found (tried {CALIBRATION_SOURCES}) — skipping imatrix")
        return None

    examples = []
    for ex in _iter_examples(src):
        msgs = ex.get("messages", [])
        if not msgs:
            continue
        parts = [
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
            for m in msgs
        ]
        examples.append("\n".join(parts))
        if len(examples) >= CALIBRATION_SAMPLES:
            break

    if not examples:
        print(f"  No examples extracted from {src} — skipping imatrix")
        return None

    os.makedirs(MERGED_DIR, exist_ok=True)
    out_path = os.path.join(MERGED_DIR, "calibration.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(examples))
    print(f"  Wrote {len(examples)} calibration examples from {src} to {out_path}")
    return out_path


def generate_imatrix(f16_gguf_path, calibration_file, imatrix_bin):
    """Run llama-imatrix to produce activation-importance weights."""
    imatrix_path = os.path.join(MERGED_DIR, "imatrix.dat")
    run(
        f"{imatrix_bin} -m {f16_gguf_path} -f {calibration_file} "
        f"-o {imatrix_path} --chunks {IMATRIX_CHUNKS}"
    )
    return imatrix_path


# ── Step 1: Download model ──────────────────────────────────
print("=" * 60)
print(f"Step 1: Downloading {SOURCE_REPO}...")
print("=" * 60)

from huggingface_hub import snapshot_download

source_path = snapshot_download(SOURCE_REPO)
print(f"Downloaded to: {source_path}")


# ── Step 2: Merge LoRA(s) or use directly ───────────────────
if is_lora_repo(source_path):
    print("\n" + "=" * 60)
    if SFT_REPO is not None:
        print(f"Step 2: Preference stack — merging SFT ({SFT_REPO}) then adapter ({SOURCE_REPO})...")
    else:
        print("Step 2: LoRA adapter detected — merging with base model...")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    print(f"Base model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # If this is a preference-tuned repo, apply SFT first before stacking the adapter.
    if SFT_REPO is not None:
        sft_path = snapshot_download(SFT_REPO)
        sft = PeftModel.from_pretrained(model, sft_path)
        model = sft.merge_and_unload()
        del sft
        print(f"SFT merged from {SFT_REPO}")

    # Apply the SOURCE_REPO LoRA (preference adapter if SFT_REPO is set, else SFT alone).
    model = PeftModel.from_pretrained(model, source_path)
    model = model.merge_and_unload()
    print(f"Final merged: {sum(p.numel() for p in model.parameters()):,} params")

    os.makedirs(MERGED_DIR, exist_ok=True)
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(source_path, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Saved to {MERGED_DIR}/")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hf_model_dir = MERGED_DIR
else:
    print("\n" + "=" * 60)
    print("Step 2: Full model detected — skipping merge")
    print("=" * 60)
    hf_model_dir = source_path


# ── Step 3: Setup llama.cpp ─────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Setting up llama.cpp...")
print("=" * 60)

convert_script, quantize_bin, imatrix_bin = setup_llama_cpp()
print("llama.cpp ready.")


# ── Step 4: Convert to F16 GGUF ─────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Converting to F16 GGUF...")
print("=" * 60)

os.makedirs(MERGED_DIR, exist_ok=True)
f16_gguf = os.path.join(MERGED_DIR, "model-f16.gguf")

if REUSE_LOCAL_GGUF and os.path.isfile(f16_gguf) and os.path.getsize(f16_gguf) > 0:
    print(f"Reusing existing F16 GGUF: {f16_gguf}")
else:
    run(f"{sys.executable} {convert_script} {hf_model_dir} --outtype f16 --outfile {f16_gguf}")

size_mb = os.path.getsize(f16_gguf) / 1024 / 1024
print(f"F16 GGUF: {f16_gguf} ({size_mb:.0f} MB)")


# ── Step 4.5: imatrix calibration (optional) ────────────────
imatrix_path = None
if USE_IMATRIX:
    print("\n" + "=" * 60)
    print("Step 4.5: Generating imatrix from calibration data...")
    print("=" * 60)
    calibration_file = build_calibration_file()
    if calibration_file is not None:
        imatrix_path = generate_imatrix(f16_gguf, calibration_file, imatrix_bin)
        size_mb = os.path.getsize(imatrix_path) / 1024 / 1024
        print(f"imatrix: {imatrix_path} ({size_mb:.1f} MB)")


# ── Step 5: Quantize ────────────────────────────────────────
print("\n" + "=" * 60)
mode_note = "with imatrix calibration" if imatrix_path else "WITHOUT imatrix (calibration data missing)"
print(f"Step 5: Quantizing to {', '.join(GGUF_QUANTS)} ({mode_note})...")
print("=" * 60)

gguf_files = [f16_gguf]
imatrix_arg = f"--imatrix {imatrix_path} " if imatrix_path else ""

for quant in GGUF_QUANTS:
    out_path = os.path.join(MERGED_DIR, f"model-{quant}.gguf")
    if REUSE_LOCAL_GGUF and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        print(f"  Reusing existing {quant}: {out_path}")
    else:
        run(f"{quantize_bin} {imatrix_arg}{f16_gguf} {out_path} {quant}")
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  {quant}: {out_path} ({size_mb:.0f} MB)")
    gguf_files.append(out_path)


# ── Step 6: Push GGUF files to HF ───────────────────────────
print("\n" + "=" * 60)
if UPLOAD_F16:
    upload_files = gguf_files
else:
    upload_files = [p for p in gguf_files if p != f16_gguf]

if not PUSH_TO_HF:
    print("\n" + "=" * 60)
    print("Step 6: PUSH_TO_HF=0, skipping HuggingFace upload.")
    print("=" * 60)
    print(f"Files: {', '.join(os.path.basename(f) for f in gguf_files)}")
    raise SystemExit(0)

print("Step 6: Pushing GGUF files to HuggingFace Hub...")
print("=" * 60)

from huggingface_hub import HfApi

api = HfApi()
api.create_repo(HF_GGUF_REPO, exist_ok=True)
remote_files = set()
if SKIP_EXISTING_HF:
    try:
        remote_files = set(api.list_repo_files(repo_id=HF_GGUF_REPO, repo_type="model"))
        print(f"Remote repo has {len(remote_files)} file(s); existing GGUF uploads will be skipped.")
    except Exception as e:
        print(f"Could not list remote files, will upload normally: {e}")

for gguf_path in upload_files:
    fname = os.path.basename(gguf_path)
    if SKIP_EXISTING_HF and fname in remote_files:
        print(f"  Skipping {fname}: already present on HF")
        continue
    print(f"  Uploading {fname}...")
    api.upload_file(
        path_or_fileobj=gguf_path,
        path_in_repo=fname,
        repo_id=HF_GGUF_REPO,
        commit_message=f"Upload {fname}",
    )

print(f"\nDone! GGUF models pushed to https://huggingface.co/{HF_GGUF_REPO}")
print(f"Files: {', '.join(os.path.basename(f) for f in upload_files)}")
