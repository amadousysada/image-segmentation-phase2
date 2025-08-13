FROM python:3.11.13-slim
LABEL authors="amadousy"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. Copier le script Streamlit
COPY main.py .

ENTRYPOINT ["sh", "-c", "\
    : \"Assurez-vous que la variable d’environnement STREAMLIT_API_URL pointe vers l’API FastAPI\" \
    && streamlit run main.py --server.address=0.0.0.0 --server.port=8501 \
"]