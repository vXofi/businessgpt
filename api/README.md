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

## Domain + HTTPS

Point an `A` record for your API domain to the VM public IP:

```text
api.example.com  A  <vm-public-ip>
```

In the Yandex security group:

- allow TCP `80` from `0.0.0.0/0`
- allow TCP `443` from `0.0.0.0/0`
- remove public TCP `8000` after HTTPS works
- keep SSH `22` restricted to your own IP

Run the model API on localhost only:

```bash
docker rm -f businessgpt-api
docker run -d --name businessgpt-api --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -e BUSINESSGPT_MODEL_PATH=/models/model-Q5_K_M.gguf \
  -e BUSINESSGPT_API_KEY='your-token' \
  -e BUSINESSGPT_N_CTX=2048 \
  -e BUSINESSGPT_N_THREADS=8 \
  -v /opt/businessgpt/models:/models:ro \
  businessgpt-api:llama-server
```

Use Caddy as the public HTTPS reverse proxy:

```bash
mkdir -p /opt/businessgpt/caddy
cat > /opt/businessgpt/caddy/Caddyfile <<'EOF'
api.example.com {
    reverse_proxy 127.0.0.1:8000
}
EOF

docker rm -f businessgpt-caddy
docker run -d --name businessgpt-caddy --restart unless-stopped \
  --network host \
  -v /opt/businessgpt/caddy/Caddyfile:/etc/caddy/Caddyfile:ro \
  -v businessgpt_caddy_data:/data \
  -v businessgpt_caddy_config:/config \
  caddy:2
```

Caddy will request and renew TLS certificates automatically. If certificate
issuance fails, first check that DNS resolves to the VM and ports `80`/`443` are
open publicly.

Test:

```bash
curl -s https://api.example.com/health
curl -s https://api.example.com/generate \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"сосал?","max_tokens":64,"temperature":0.7}'
```

Telegram bot endpoint:

```text
https://api.example.com/generate
```

## Updating the Container

`api/server.py` is copied into the Docker image:

```dockerfile
COPY api/server.py ./server.py
```

So API code changes require rebuilding and restarting the container. If
`api/Dockerfile` did not change, do not use `--no-cache`: Docker will reuse the
slow llama.cpp build layer and only refresh the app layer.

On the VM:

```bash
cd /opt/businessgpt/app
git pull
docker rm -f businessgpt-api
docker build -f api/Dockerfile -t businessgpt-api:llama-server .
docker run -d --name businessgpt-api --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -e BUSINESSGPT_MODEL_PATH=/models/model-Q5_K_M.gguf \
  -e BUSINESSGPT_API_KEY='your-token' \
  -e BUSINESSGPT_N_CTX=2048 \
  -e BUSINESSGPT_N_THREADS=8 \
  -v /opt/businessgpt/models:/models:ro \
  businessgpt-api:llama-server
```

Use `--no-cache` only after changing the llama.cpp build layer or when a cached
Docker layer is suspected to be broken.

## Notes

- Q5_K_M is the practical first quant for quality. Q4_K_M is cheaper in RAM and usually worse.
- `BUSINESSGPT_N_CTX=2048` is enough for short Telegram-style prompts. Raise it only if you need longer chat context.
- Use one uvicorn worker. Multiple workers would load multiple copies of the GGUF into RAM.
- Reward-model best-of-N can be added later as a separate reranking layer; ship the single-generation endpoint first.
