"""
user_broker_configs 테이블에 auto_trade_enabled 컬럼을 추가하는 마이그레이션 스크립트.

사용법 (서버 중지 상태에서 한 번만 실행):

    python -m backend.migrate_user_broker_auto_trade
"""

from __future__ import annotations

import os

import psycopg2
from dotenv import load_dotenv


def main():
    load_dotenv()

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME", "stock_ai")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")

    conn = None
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        conn.autocommit = True
        cur = conn.cursor()

        print("🔧 ALTER TABLE user_broker_configs ...")
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'user_broker_configs'
                      AND column_name = 'auto_trade_enabled'
                ) THEN
                    ALTER TABLE user_broker_configs
                    ADD COLUMN auto_trade_enabled BOOLEAN NOT NULL DEFAULT FALSE;
                END IF;
            END
            $$;
            """
        )
        print("✅ auto_trade_enabled 컬럼 추가(또는 이미 존재) 완료")

    except Exception as e:
        print(f"❌ 마이그레이션 실패: {e}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()





