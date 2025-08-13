FROM python:3.11.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# runtime, qui ne contient que l’app et les paquets installés
FROM python:3.11.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copier les paquets
COPY --from=builder /install /usr/local
ENV PATH="/usr/local/bin:$PATH"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]