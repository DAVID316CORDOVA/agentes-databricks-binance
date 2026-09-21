# Databricks notebook source
# MAGIC %md
# MAGIC # Log + registro del agente SQL en MLflow / Unity Catalog
# MAGIC
# MAGIC Corre esto en un cluster/serverless con Databricks Runtime ML (o instala los
# MAGIC requirements). Este notebook es el que el job de CI/CD ejecuta para producir
# MAGIC una nueva versión del modelo antes de actualizar el serving endpoint.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.append("../src")

import mlflow
from agent.sql_agent import build_chain, FULL_TABLE

CATALOG = dbutils.widgets.get("catalog") if "catalog" in [w.name for w in dbutils.widgets.getAll()] else "dbw_fintech_fdcg01"
SCHEMA = "gold"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.sql_agent_binance"

mlflow.set_registry_uri("databricks-uc")
mlflow.langchain.autolog()  # tracing automático de cada invocación (LLMOps con MLflow nativo)

# COMMAND ----------

os.environ.setdefault("DATABRICKS_HOST", spark.conf.get("spark.databricks.workspaceUrl", ""))
os.environ.setdefault("UC_CATALOG", CATALOG)
os.environ.setdefault("UC_SCHEMA", SCHEMA)
# DATABRICKS_HTTP_PATH y DATABRICKS_TOKEN: setéalos como secrets/widgets antes de correr.

chain = build_chain()

input_example = {"question": "¿Cuál fue el precio de cierre más alto de ETHUSDT hoy?"}

with mlflow.start_run(run_name="sql_agent_binance") as run:
    model_info = mlflow.langchain.log_model(
        lc_model=chain,
        name="sql_agent_binance",
        input_example=input_example,
        registered_model_name=REGISTERED_MODEL_NAME,
        pip_requirements="../requirements.txt",
        metadata={"table": FULL_TABLE},
    )
    print(f"Modelo registrado: {REGISTERED_MODEL_NAME}, run_id={run.info.run_id}")
    print(f"model_uri: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC Tras correr esto, anota la versión impresa (o consúltala en el UC Model
# MAGIC Registry) y actualiza `entity_version` en `resources/model_serving.yml`
# MAGIC antes de `databricks bundle deploy`. Para no hacerlo a mano cada vez, el
# MAGIC job de CI/CD incluye un paso que lee la última versión automáticamente
# MAGIC (ver `.github/workflows/cicd.yml`).
