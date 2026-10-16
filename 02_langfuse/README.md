# Langfuse Setup Guide

Run a self-hosted Langfuse server locally with Docker and connect your Python code to it.

The account, project, and keys are created automatically on first start. To change them, copy [`.env.example`](.env.example) to `.env` in this folder before the first start.
---

# Prerequisites

- Docker Desktop running (WSL2 backend on Windows)
- Python 3.11+

```powershell
docker version
docker compose version
py -3.11 --version
```

Langfuse runs six containers and needs roughly 3-5 GB of RAM. If WSL2 uses too much memory, cap it in `%UserProfile%\.wslconfig` (then run `wsl --shutdown`). Do not go below about 8 GB.

```ini
[wsl2]
memory=16GB
```

---

# Step 1: Start Langfuse

```powershell
cd 02_langfuse
docker compose up -d
docker compose ps
```

Wait until `langfuse-web` is `running` and the other containers are `healthy`. The first start takes 1-3 minutes.

---

# Step 2: Verify

```powershell
Invoke-RestMethod http://127.0.0.1:3000/api/public/health
```

Open http://localhost:3000 and sign in:

| Setting | Value |
|---|---|
| UI login | `student@example.com` / `change-me-please` |
| Public key | `pk-lf-local-dev-key` |
| Secret key | `sk-lf-local-dev-secret` |
| Base URL | `http://127.0.0.1:3000` |



> The default secrets are placeholders, safe only because everything is bound to `127.0.0.1`. Replace them before exposing Langfuse to a network.

---

# Step 3: Stop / Reset

```powershell
docker compose stop        # stop, keep data
docker compose start       # start again
docker compose down        # remove containers, keep data
docker compose down -v     # remove containers AND delete all traces
```
