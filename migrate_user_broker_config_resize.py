"""
user_broker_configs.kis_app_key / kis_app_secret 컬럼 길이를
보다 크게 늘리기 위한 마이그레이션 스크립트.

기존 값은 그대로 두고, 타입만 VARCHAR(256/512) 로 확장한다.
"""

from sqlalchemy import text

from database import DatabaseManager


def main() -> None:
  db = DatabaseManager()
  if not db.connect():
      print("❌ DB 연결 실패")
      return

  engine = db.engine
  stmts = [
      "ALTER TABLE user_broker_configs ALTER COLUMN kis_app_key TYPE VARCHAR(256);",
      "ALTER TABLE user_broker_configs ALTER COLUMN kis_app_secret TYPE VARCHAR(512);",
  ]

  with engine.begin() as conn:
      for s in stmts:
          print(f"실행 중: {s}")
          conn.execute(text(s))

  print("✅ user_broker_configs 앱키/시크릿 컬럼 길이 확장 완료")


if __name__ == "__main__":
    main()


