"""
Chunking (recursive, el default razonable) + carga a una tabla Delta que
luego Vector Search va a indexar.
"""
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from databricks import sql as dbsql
from langchain_text_splitters import RecursiveCharacterTextSplitter

CATALOG = os.environ.get("UC_CATALOG", "dbw_fintech_fdcg01")
SCHEMA = os.environ.get("UC_SCHEMA", "agents_demo")
TABLE = f"{CATALOG}.{SCHEMA}.trading_docs"

DOC_PATH = Path(__file__).resolve().parents[1] / "data" / "trading_concepts.txt"

# Recursive: corta por parrafo (\n\n) primero; si un parrafo es muy largo,
# cae a cortar por linea, luego por espacio. chunk_size en caracteres.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=40,
    separators=["\n\n", "\n", ". ", " "],
)

text = DOC_PATH.read_text(encoding="utf-8")
chunks = splitter.split_text(text)
print(f"Documento cortado en {len(chunks)} chunks.\n")
for i, c in enumerate(chunks):
    print(f"--- chunk {i} ({len(c)} chars) ---")
    print(c.strip()[:120].replace("\n", " "), "...")
print()

with dbsql.connect(
    server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", ""),
    http_path=os.environ["DATABRICKS_HTTP_PATH"],
    access_token=os.environ["DATABRICKS_TOKEN"],
) as conn:
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
              id STRING NOT NULL,
              content STRING,
              source STRING
            ) USING DELTA
            TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        # CDC (change data feed) es requisito de Vector Search para poder
        # sincronizar incrementalmente la tabla con el indice.

        cur.execute(f"DELETE FROM {TABLE}")  # limpio para poder re-correr este script

        values_sql = ", ".join(
            "('{}', '{}', 'trading_concepts.txt')".format(
                uuid.uuid4().hex, c.replace("'", "''").replace("\n", " ")
            )
            for c in chunks
        )
        cur.execute(f"INSERT INTO {TABLE} (id, content, source) VALUES {values_sql}")

print(f"{len(chunks)} chunks insertados en {TABLE}")
