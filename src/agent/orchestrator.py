"""
Orquestador multiagente (LangGraph): decide si la pregunta es sobre datos
de mercado (-> agente SQL via MCP) o sobre conceptos de trading (-> agente RAG).
"""
import asyncio
import os
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

import httpx2
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider
from langgraph.graph import StateGraph, END
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from src.agent.rag_agent import build_rag_chain

CATALOG = os.environ.get("UC_CATALOG", "dbw_fintech_fdcg01")
SCHEMA = os.environ.get("UC_SCHEMA", "agents_demo")
LLM_ENDPOINT = os.environ.get("DATABRICKS_LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
MCP_SERVER_URL = f"{os.environ['DATABRICKS_HOST']}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}"

ws_client = WorkspaceClient(host=os.environ["DATABRICKS_HOST"], token=os.environ["DATABRICKS_TOKEN"])
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.0, workspace_client=ws_client)
rag_chain = build_rag_chain()


class AgentState(TypedDict):
    question: str
    route: str
    answer: str


def router_node(state: AgentState) -> AgentState:
    """El LLM decide: esto es de DATOS (precio real) o de CONCEPTOS (teoria)?"""
    decision = llm.invoke(
        "Responde SOLO con la palabra 'datos' o 'conceptos'.\n"
        "'datos' si la pregunta pide un precio, valor o numero concreto de mercado.\n"
        "'conceptos' si la pregunta pide explicar que es un indicador o tecnica de trading.\n"
        f"Pregunta: {state['question']}"
    ).content.strip().lower()
    route = "datos" if "datos" in decision else "conceptos"
    print(f"  [orquestador] ruta elegida: {route}")
    return {**state, "route": route}


def route_condition(state: AgentState) -> Literal["sql_agent", "rag_agent"]:
    return "sql_agent" if state["route"] == "datos" else "rag_agent"


async def _call_mcp_tool(question: str) -> str:
    async with httpx2.AsyncClient(
        auth=DatabricksOAuthClientProvider(ws_client),
        follow_redirects=True,
        timeout=120.0,
    ) as http_client:
        async with Client(
            streamable_http_client(MCP_SERVER_URL, http_client=http_client)
        ) as session:
            result = await session.call_tool(
                f"{CATALOG}__{SCHEMA}__get_max_price",
                arguments={"p_symbol": "BTCUSDT", "p_hours": 1},
            )
            return str(result.content)


def sql_agent_node(state: AgentState) -> AgentState:
    """Agente de DATOS: llama la tool de mercado via MCP."""
    raw = asyncio.run(_call_mcp_tool(state["question"]))
    print(f"  [debug] raw devuelto por MCP: {raw}")

    question = state["question"]
    instruction = (
        "Ya se ejecuto una consulta real contra la base de datos via una tool "
        "de Databricks. El resultado que ves abajo ES el dato real y actual, "
        "no una simulacion. NUNCA digas que no tienes acceso a datos en tiempo "
        "real -- ya los tienes, en 'Resultado crudo'.\n\n"
        f"Pregunta: {question}\n"
        f"Resultado crudo de la tool MCP: {raw}\n\n"
        "Extrae el numero del resultado y responde en espanol, en una frase clara."
    )
    answer = llm.invoke(instruction).content
    return {**state, "answer": answer}


def rag_agent_node(state: AgentState) -> AgentState:
    """Agente de CONCEPTOS: usa el RAG que armamos."""
    answer = rag_chain.invoke({"question": state["question"]})
    return {**state, "answer": answer}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("rag_agent", rag_agent_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_condition, {
        "sql_agent": "sql_agent",
        "rag_agent": "rag_agent",
    })
    graph.add_edge("sql_agent", END)
    graph.add_edge("rag_agent", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    for q in [
        "¿Cuál fue el precio máximo de BTCUSDT en la última hora?",
        "¿Qué significa que el RSI esté sobre 70?",
    ]:
        print(f"\n=== Pregunta: {q} ===")
        result = app.invoke({"question": q, "route": "", "answer": ""})
        print("Respuesta:", result["answer"])