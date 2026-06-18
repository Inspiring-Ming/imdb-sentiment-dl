# Hugging Face Spaces (Docker SDK) — serves the Streamlit sentiment demo.
# HF routes traffic to port 7860, so Streamlit must listen there.
FROM python:3.11-slim

WORKDIR /app

# System deps occasionally needed by TensorFlow wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces convention: the app is served on 7860.
EXPOSE 7860
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
