"""
Agente text-to-SQL simple sobre trades_ohlc_1min (Binance BTCUSDT/ETHUSDT).

Diseño deliberadamente simple para el plazo ajustado:
  1. El LLM (Foundation Model de Databricks vía ChatDatabricks) recibe la
     pregunta del usuario + el esquema de UNA tabla y genera SQL.
  2. Ejecutamos ese SQL contra el SQL Warehouse con databricks-sql-connector.
  3. El LLM resume el resultado en lenguaje natural.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Ruta absoluta al .env en la raiz del proyecto (dos carpetas arriba de este
# archivo: src/agent/sql_agent.py -> src/agent -> src -> raiz).
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from databricks import sql as dbsql
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

CATALOG = os.environ.get("UC_CATALOG", "dbw_fintech_fdcg01")
SCHEMA = os.environ.get("UC_SCHEMA", "gold")
TABLE = os.environ.get("UC_TABLE", "trades_ohlc_1min")
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"

# Ajusta este esquema al real de tu tabla (columnas exactas de tus velas OHLC).
TABLE_SCHEMA = f"""
Tabla: {FULL_TABLE}
Columnas:
  - symbol            STRING      (ej. 'BTCUSDT', 'ETHUSDT')
  - window_start       TIMESTAMP  (inicio de la vela de 1 minuto)
  - open, high, low, close   DOUBLE
  - volume             DOUBLE
"""

LLM_ENDPOINT = os.environ.get("DATABRICKS_LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")

SQL_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un generador de SQL para Databricks SQL (dialecto Spark SQL). "
            "SOLO puedes usar la siguiente tabla, no inventes otras:\n"
            f"{TABLE_SCHEMA}\n"
            "Reglas:\n"
            "- Devuelve UNICAMENTE la sentencia SQL, sin explicacion, sin markdown, sin ```.\n"
            "- Usa siempre el nombre completo de la tabla: " + FULL_TABLE + "\n"
            "- Limita los resultados a 200 filas como maximo si no se especifica.\n"
            "- Solo SELECT. Nunca generes INSERT/UPDATE/DELETE/DROP.",
        ),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente de trading cuantitativo. Se te da la pregunta original "
            "del usuario, el SQL que se ejecuto y el resultado en formato de filas. "
            "Responde en lenguaje natural, claro y conciso, en espanol, citando "
            "cifras concretas del resultado.",
        ),
        (
            "human",
            "Pregunta: {question}\n\nSQL ejecutado:\n{sql}\n\nResultado:\n{rows}",
        ),
    ]
)

_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|truncate|merge)\b", re.IGNORECASE)


def _clean_sql(raw_sql: str) -> str:
    sql = raw_sql.strip()
    sql = re.sub(r"^```sql\s*|^```\s*|```$", "", sql, flags=re.MULTILINE).strip()
    if _FORBIDDEN.search(sql):
        raise ValueError(f"SQL no permitido (solo lectura): {sql}")
    if FULL_TABLE not in sql:
        raise ValueError(f"El SQL generado no referencia la tabla permitida {FULL_TABLE}: {sql}")
    return sql


def _run_sql(sql: str) -> list[dict]:
    with dbsql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", ""),
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
            return [dict(zip(cols, r)) for r in rows]


def _execute_step(inputs: dict) -> dict:
    """Limpia el SQL generado por el LLM y lo ejecuta contra el Warehouse."""
    sql = _clean_sql(inputs["raw_sql"])
    rows = _run_sql(sql)
    return {"question": inputs["question"], "sql": sql, "rows": rows}


def build_chain():
    """Construye el LCEL Runnable que MLflow va a loguear y servir."""
    # Credenciales EXPLICITAS: evitamos que la libreria busque sola en
    # ~/.databrickscfg (que puede tener otro workspace como default).
    ws_client = WorkspaceClient(
        host=os.environ["DATABRICKS_HOST"],
        token=os.environ["DATABRICKS_TOKEN"],
    )
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.0, workspace_client=ws_client)

    sql_gen = SQL_GEN_PROMPT | llm | StrOutputParser()

    generate_and_execute = (
        
        RunnablePassthrough.assign(raw_sql=sql_gen) | RunnableLambda(_execute_step)
    )

    chain = generate_and_execute | ANSWER_PROMPT | llm | StrOutputParser()
    return chain


if __name__ == "__main__":
    chain = build_chain()
    result = chain.invoke({"question": "¿Cuál fue el precio máximo de BTCUSDT en la última hora?"})
    print(result)