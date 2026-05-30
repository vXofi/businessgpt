# BusinessGPT API

Small HTTP wrapper around the BusinessGPT GGUF model.

## What to Host

Use a regular Yandex Compute Cloud Linux VM first, not Serverless Containers.

Recommended starting VM:

- Ubuntu 22.04 or 24.04 LTS
- 4 vCPU minimum, 8 vCPU if you want tolerable CPU latency
- 16 GB RAM
- 40-60 GB disk
- static public IP

Why VM: the current production target is a 9B GGUF via llama.cpp. The model has to stay warm in RAM; cold-loading a multi-GB model per request is the wrong shape for serverless. Serverless can be useful later for a thin gateway, but the inference worker should be long-lived.

## Endpoint

`POST /generate`

Request body:

```json
{
  "prompt": "string",
  "max_tokens": 256,
  "temperature": 0.7
}
```

`top_p`, `top_k`, and `repetition_penalty` are optional.

Response:

```json
{
  "response": "model answer",
  "model": "businessgpt-q5_k_m.gguf",
  "elapsed_ms": 1234,
  "usage": {}
}
```

## Local Smoke Test

Put a GGUF model at `models/businessgpt.gguf`, then:

```bash
docker build -f api/Dockerfile -t businessgpt-api .
docker run --rm -p 8000:8000 \
  -e BUSINESSGPT_MODEL_PATH=/models/businessgpt.gguf \
  -e BUSINESSGPT_API_KEY=change-me \
  -v "$PWD/models:/models:ro" \
  businessgpt-api
```

Test:

```bash
curl -s http://localhost:8000/generate \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"сосал?","max_tokens":64,"temperature":0.7}'
```

## Yandex VM Deployment

1. Create a Yandex Compute Cloud Linux VM.
2. Give it a public IP.
3. Security group:
   - allow SSH `22` only from your IP
   - allow HTTP `80` / HTTPS `443` for production
   - optionally allow `8000` only for a short smoke test
4. Install Docker on the VM.
5. Copy or download the GGUF to `/opt/businessgpt/models/businessgpt.gguf`.
6. Clone this repo to `/opt/businessgpt/app`.
7. Build and run:

```bash
cd /opt/businessgpt/app
docker build -f api/Dockerfile -t businessgpt-api .
docker run -d --name businessgpt-api --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -e BUSINESSGPT_MODEL_PATH=/models/businessgpt.gguf \
  -e BUSINESSGPT_API_KEY='replace-with-long-token' \
  -e BUSINESSGPT_N_CTX=2048 \
  -e BUSINESSGPT_N_THREADS=8 \
  -v /opt/businessgpt/models:/models:ro \
  businessgpt-api
```

For production, put Nginx/Caddy in front of `127.0.0.1:8000` and terminate TLS there. Keep the model API bound to localhost.

## Notes

- Q5_K_M is the practical first quant for quality. Q4_K_M is cheaper in RAM and usually worse.
- `BUSINESSGPT_N_CTX=2048` is enough for short Telegram-style prompts. Raise it only if you need longer chat context.
- Use one uvicorn worker. Multiple workers would load multiple copies of the GGUF into RAM.
- Reward-model best-of-N can be added later as a separate reranking layer; ship the single-generation endpoint first.
