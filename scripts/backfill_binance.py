"""
Backfill historico (una sola vez, sin streaming) de velas OHLC 1-minuto
desde la API publica de Binance hacia dbw_fintech_fdcg01.agents_demo.trades_ohlc_1min

Uso:
    python scripts/backfill_binance.py --days 3
"""

import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv
import os

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from databricks import sql as dbsql

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1m"
CATALOG = os.environ.get("UC_CATALOG", "dbw_fintech_fdcg01")
SCHEMA = os.environ.get("UC_SCHEMA", "agents_demo")
TABLE = os.environ.get("UC_TABLE", "trades_ohlc_1min")
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """Trae velas de Binance en paginas de 1000 (limite de la API)."""
    all_rows = []
    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get(
            BINANCE_KLINES_URL,
            params={
                "symbol": symbol,
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + 60_000  # siguiente minuto despues de la ultima vela
        time.sleep(0.2)  # ser gentil con el rate limit publico
    return all_rows


def klines_to_rows(symbol: str, klines: list[list]) -> list[tuple]:
    rows = []
    for k in klines:
        open_time_ms = k[0]
        ts = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
        rows.append(
            (
                symbol,
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                float(k[1]),  # open
                float(k[2]),  # high
                float(k[3]),  # low
                float(k[4]),  # close
                float(k[5]),  # volume
            )
        )
    return rows


def insert_rows(rows: list[tuple]) -> None:
    if not rows:
        print("Nada que insertar.")
        return

    with dbsql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", ""),
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    ) as conn:
        with conn.cursor() as cur:
            chunk_size = 500
            for i in range(0, len(rows), chunk_size):
                chunk = rows[i : i + chunk_size]
                values_sql = ", ".join(
                    "('{}', TIMESTAMP'{}', {}, {}, {}, {}, {})".format(*row)
                    for row in chunk
                )
                cur.execute(f"INSERT INTO {FULL_TABLE} VALUES {values_sql}")
                print(f"  insertadas {i + len(chunk)}/{len(rows)} filas...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3, help="dias hacia atras a traer")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    for symbol in SYMBOLS:
        print(f"Trayendo {symbol} desde Binance ({args.days} dias)...")
        klines = fetch_klines(symbol, start_ms, end_ms)
        print(f"  {len(klines)} velas obtenidas para {symbol}")
        rows = klines_to_rows(symbol, klines)
        insert_rows(rows)

    print(f"Listo. Tabla destino: {FULL_TABLE}")


if __name__ == "__main__":
    main()
