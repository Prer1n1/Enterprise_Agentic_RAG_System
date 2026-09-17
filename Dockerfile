# Slim, not full — this image only needs Python + pip installs, no
# compilers/dev headers that the full python:3.11 image carries.
FROM python:3.11-slim

WORKDIR /app

# Dependencies copied and installed BEFORE the rest of the source code —
# this becomes its own cached layer. Changing agent/nodes.py later
# invalidates layers after this point, but NOT this one, so `docker build`
# doesn't reinstall ~20 heavy packages (langchain, chromadb, torch-free
# but still substantial) on every code change, only on a requirements.txt
# change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now the actual source code.
COPY . .

EXPOSE 8000

# Lets `docker ps` / docker-compose / an orchestrator know whether the
# container is actually SERVING traffic, not just that the process is
# alive — hits our own real /health endpoint (the one that checks both
# stores), not a fake liveness ping.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
