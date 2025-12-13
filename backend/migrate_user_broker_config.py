"""
user_broker_configs 테이블에 새 컬럼(kis_app_key, kis_app_secret, real_mode)을
안전하게 추가하기 위한 간단한 마이그레이션 스크립트.

기존 데이터는 그대로 두고, 부족한 컬럼만 IF NOT EXISTS 로 추가합니다.
"""

from sqlalchemy import text

from backend.database import DatabaseManager


def main() -> None:
    db = DatabaseManager()
    if not db.connect():
        print("❌ DB 연결 실패")
        return

    engine = db.engine
    stmts = [
        "ALTER TABLE user_broker_configs ADD COLUMN IF NOT EXISTS kis_app_key VARCHAR(128);",
        "ALTER TABLE user_broker_configs ADD COLUMN IF NOT EXISTS kis_app_secret VARCHAR(128);",
        "ALTER TABLE user_broker_configs ADD COLUMN IF NOT EXISTS real_mode BOOLEAN NOT NULL DEFAULT FALSE;",
    ]

    with engine.begin() as conn:
        for s in stmts:
            print(f"실행 중: {s}")
            conn.execute(text(s))

    print("✅ user_broker_configs 컬럼 마이그레이션 완료")


if __name__ == "__main__":
    main()



