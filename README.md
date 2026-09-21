# agentes-databricks-binance

Agente text-to-SQL (LangChain) sobre `dbw_fintech_fdcg01.gold.trades_ohlc_1min`
(velas OHLC de BTCUSDT/ETHUSDT ingeridas por `kafka-databricks-streaming-pipeline`),
desplegado como Databricks Model Serving endpoint, con tracing nativo de MLflow
y una API FastAPI delgada (Databricks App) como capa "de producción" externa.

MVP de 1-2 días. Evoluciona hacia el sistema multiagente (SQL-Agents, tesis)
cuando haya más tiempo.

## Estado (marca lo que ya corriste)

- [ ] `01` — Chain de LangChain corre localmente contra el SQL Warehouse
- [ ] `02` — Modelo logueado y registrado en UC Model Registry vía MLflow
- [ ] `03` — `databricks bundle deploy -t dev` crea el serving endpoint
- [ ] `04` — Endpoint responde a una invocación de prueba
- [ ] `05` — FastAPI (Databricks App) responde `/ask` end-to-end
- [ ] `06` — CI/CD en GitHub Actions corre en verde (dev on PR, prod on merge)

## Quickstart local (paso 1)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export DATABRICKS_HOST=https://adb-xxxx.azuredatabricks.net
export DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/xxxxxxxx
export DATABRICKS_TOKEN=dapiXXXX...
export UC_CATALOG=dbw_fintech_fdcg01
export UC_SCHEMA=gold

python -m src.agent.sql_agent
```

Si esto imprime una respuesta coherente sobre BTCUSDT/ETHUSDT, el chain
funciona y podemos pasar a loguearlo con MLflow.

## Registrar el modelo (paso 2)

Sube `notebooks/01_log_and_register_model.py` a tu workspace (o corre
`databricks bundle deploy` y ejecútalo como job) — configura las mismas env
vars como secrets/widgets del notebook.

## Deploy del bundle (paso 3)

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run register_agent_model -t dev
```

## TODOs antes de la entrevista

- [ ] Reemplazar los `host:` placeholder en `databricks.yml` por los reales
      de tu workspace (mismos que usas en el proyecto Kafka).
- [ ] Confirmar `DATABRICKS_LLM_ENDPOINT` — lista los Foundation Models
      pay-per-token habilitados en tu workspace y usa el nombre exacto.
- [ ] Ajustar `TABLE_SCHEMA` en `src/agent/sql_agent.py` a las columnas
      reales de tu tabla `trades_ohlc_1min` (nombres pueden diferir).
- [ ] `prod_service_principal` en `databricks.yml` — mismo SP del proyecto
      Kafka o uno nuevo con permisos de UC sobre `gold.trades_ohlc_1min`.
- [ ] Secrets en GitHub: `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`,
      `DATABRICKS_CLIENT_SECRET`.
