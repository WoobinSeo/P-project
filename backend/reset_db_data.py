"""
기존 PostgreSQL 주가 데이터를 정리(삭제)하는 스크립트.

- stock_prices, stock_prices_processed 테이블에서
  국내 3개 종목(삼성전자, 네이버, 현대차) 데이터 삭제.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def main():
    load_dotenv()

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")

    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(conn_str, echo=False, future=True)

    target_codes = ["005930", "035420", "005380"]

    print("DB 연결:", conn_str)
    print("삭제 대상 종목 코드:", target_codes)

    with engine.begin() as conn:
        # processed 먼저 삭제 (FK가 있을 수 있으므로)
        conn.execute(
            text(
                """
                DELETE FROM stock_prices_processed
                WHERE stock_code = ANY(:codes)
                """
            ),
            {"codes": target_codes},
        )
        conn.execute(
            text(
                """
                DELETE FROM stock_prices
                WHERE stock_code = ANY(:codes)
                """
            ),
            {"codes": target_codes},
        )

    print("✅ 기존 데이터 삭제 완료")


if __name__ == "__main__":
    main()














