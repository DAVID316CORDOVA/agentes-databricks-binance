"""
Prueba minima del protocolo MCP contra el servidor gestionado de Databricks
para Unity Catalog Functions.

Dos operaciones estandar del protocolo:
  - tools/list  -> "que herramientas hay disponibles en este servidor?"
  - tools/call  -> "ejecuta esta herramienta con estos argumentos"
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from databricks.sdk import WorkspaceClient
from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider

CATALOG = os.environ.get("UC_CATALOG", "dbw_fintech_fdcg01")
SCHEMA = os.environ.get("UC_SCHEMA", "agents_demo")

MCP_SERVER_URL = (
    f"{os.environ['DATABRICKS_HOST']}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}"
)


async def main():
    ws_client = WorkspaceClient(
        host=os.environ["DATABRICKS_HOST"],
        token=os.environ["DATABRICKS_TOKEN"],
    )

    print(f"Conectando al servidor MCP: {MCP_SERVER_URL}\n")

    async with httpx2.AsyncClient(
        auth=DatabricksOAuthClientProvider(ws_client),
        follow_redirects=True,
        timeout=120.0,  # el warehouse puede tardar en "despertar" (cold start)
    ) as http_client:
        async with Client(
            streamable_http_client(MCP_SERVER_URL, http_client=http_client)
        ) as session:
            # --- Operacion 1 del protocolo: tools/list ---
            tools_response = await session.list_tools()
            print("Herramientas que este servidor MCP expone:")
            for tool in tools_response.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()

            # --- Operacion 2 del protocolo: tools/call ---
            tool_name = f"{CATALOG}__{SCHEMA}__get_max_price"
            print(f"Llamando a la tool '{tool_name}' via MCP...")
            result = await session.call_tool(
                tool_name,
                arguments={"p_symbol": "BTCUSDT", "p_hours": 1},
            )
            print("Resultado:", result.content)


if __name__ == "__main__":
    asyncio.run(main())
