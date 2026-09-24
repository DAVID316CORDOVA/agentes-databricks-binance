"""
Agente RAG: pregunta -> busca chunks relevantes en Vector Search -> LLM
responde usando esos chunks como contexto.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import ChatDatabricks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

CATALOG = os.environ.get("UC_CATALOG", "dbw_fintech_fdcg01")
SCHEMA = os.environ.get("UC_SCHEMA", "agents_demo")
INDEX_NAME = f"{CATALOG}.{SCHEMA}.trading_docs_index"
LLM_ENDPOINT = os.environ.get("DATABRICKS_LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente experto en trading. Responde la pregunta del "
            "usuario basandote UNICAMENTE en el siguiente contexto recuperado. "
            "Si el contexto no alcanza para responder, dilo explicitamente, "
            "no inventes.\n\nContexto:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def _retrieve(inputs: dict) -> dict:
    """Operacion de RETRIEVAL: busca los chunks mas relevantes por similitud."""
    vsc = VectorSearchClient(
        workspace_url=os.environ["DATABRICKS_HOST"],
        personal_access_token=os.environ["DATABRICKS_TOKEN"],
    )
    index = vsc.get_index(index_name=INDEX_NAME)
    results = index.similarity_search(
        query_text=inputs["question"],
        columns=["content", "source"],
        num_results=3,
    )
    rows = results.get("result", {}).get("data_array", [])
    context = "\n\n".join(r[0] for r in rows) if rows else "(sin resultados)"
    return {"question": inputs["question"], "context": context}


def build_rag_chain():
    ws_client = WorkspaceClient(
        host=os.environ["DATABRICKS_HOST"],
        token=os.environ["DATABRICKS_TOKEN"],
    )
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.0, workspace_client=ws_client)

    chain = RunnableLambda(_retrieve) | RAG_PROMPT | llm | StrOutputParser()
    return chain


if __name__ == "__main__":
    chain = build_rag_chain()
    result = chain.invoke({"question": "¿Qué significa que el RSI esté sobre 70?"})
    print(result)
