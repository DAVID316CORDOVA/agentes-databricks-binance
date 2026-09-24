"""
Crea el endpoint de Vector Search (motor de computo) y el indice
(Delta Sync: se mantiene sincronizado solo con la tabla trading_docs).

OJO: crear el endpoint tarda varios minutos la primera vez. Este script
lanza la creacion y sale -- no se queda esperando.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from databricks.vector_search.client import VectorSearchClient

CATALOG = os.environ.get("UC_CATALOG", "dbw_fintech_fdcg01")
SCHEMA = os.environ.get("UC_SCHEMA", "agents_demo")
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.trading_docs"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.trading_docs_index"
ENDPOINT_NAME = "agents-demo-vs-endpoint"

vsc = VectorSearchClient(
    workspace_url=os.environ["DATABRICKS_HOST"],
    personal_access_token=os.environ["DATABRICKS_TOKEN"],
)

existing_endpoints = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if ENDPOINT_NAME not in existing_endpoints:
    print(f"Creando endpoint '{ENDPOINT_NAME}' (tarda varios minutos)...")
    vsc.create_endpoint(name=ENDPOINT_NAME, endpoint_type="STANDARD")
else:
    print(f"Endpoint '{ENDPOINT_NAME}' ya existe, sigo con el indice.")

try:
    print(f"Creando indice '{INDEX_NAME}' (managed embeddings, modelo databricks-gte-large-en)...")
    vsc.create_delta_sync_index(
        endpoint_name=ENDPOINT_NAME,
        source_table_name=SOURCE_TABLE,
        index_name=INDEX_NAME,
        pipeline_type="TRIGGERED",  # sincroniza manual/on-demand, no continuo (mas barato)
        primary_key="id",
        embedding_source_column="content",
        embedding_model_endpoint_name="databricks-gte-large-en",
    )
    print("Indice creado (o ya existia). Puede tardar en quedar ONLINE.")
except Exception as e:
    print(f"Nota: {e}")
    print("Si dice que ya existe, esta bien, seguimos.")
