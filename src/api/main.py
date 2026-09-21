"""
API delgada que llama al Model Serving endpoint de Databricks.

Simula "producción real" fuera del workspace: un cliente cualquiera le pega
a esta API por HTTP y no sabe (ni le importa) que detrás hay un agente
LangChain corriendo en Databricks Model Serving.

Pensada para desplegarse como Databricks App (ver resources/app.yml), pero
funciona igual con `uvicorn src.api.main:app` en cualquier lado.
"""

import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Binance SQL Agent API", version="0.1.0")

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
SERVING_ENDPOINT_NAME = os.environ.get("SERVING_ENDPOINT_NAME", "dev-sql-agent-binance")
# En una Databricks App este token se inyecta automáticamente (contexto del
# service principal de la app); localmente usa un PAT tuyo para probar.
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok", "endpoint": SERVING_ENDPOINT_NAME}


@app.post("/ask", response_model=Answer)
def ask(payload: Question):
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        raise HTTPException(500, "DATABRICKS_HOST / DATABRICKS_TOKEN no configurados")

    url = f"{DATABRICKS_HOST}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
        json={"inputs": [{"question": payload.question}]},
        timeout=60,
    )
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"Error del serving endpoint: {resp.text}")

    body = resp.json()
    # El formato exacto de "predictions" depende de cómo MLflow serializó el
    # output del chain (StrOutputParser -> normalmente lista de strings).
    predictions = body.get("predictions", body)
    text = predictions[0] if isinstance(predictions, list) else predictions
    return Answer(answer=text)
