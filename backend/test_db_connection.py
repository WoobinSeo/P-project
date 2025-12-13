"""
PostgreSQL 연결 테스트
"""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

print("="*60)
print("PostgreSQL 연결 테스트")
print("="*60)

# 환경 변수 확인
host = os.getenv('DB_HOST', 'localhost')
port = os.getenv('DB_PORT', '5432')
database = os.getenv('DB_NAME', 'stock_ai')
user = os.getenv('DB_USER', 'postgres')
password = os.getenv('DB_PASSWORD', '')

print(f"\n연결 정보:")
print(f"  Host: {host}")
print(f"  Port: {port}")
print(f"  Database: {database}")
print(f"  User: {user}")
print(f"  Password: {'*' * len(password) if password else '(없음)'}")

print(f"\n연결 시도 중...")

try:
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    
    print("✅ 연결 성공!")
    
    # 버전 확인
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"\n PostgreSQL 버전:")
    print(f"  {version[0]}")
    
    # 기존 테이블 확인
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    tables = cursor.fetchall()
    
    print(f"\n기존 테이블: {len(tables)}개")
    for table in tables:
        print(f"  - {table[0]}")
    
    cursor.close()
    conn.close()
    
    print("\n✅ 테스트 완료")
    
except psycopg2.OperationalError as e:
    print(f"\n❌ 연결 실패: {e}")
    print("\n확인 사항:")
    print("1. PostgreSQL 서버가 실행 중인가요?")
    print("   - Windows: 서비스에서 'postgresql-x64-xx' 확인")
    print("   - 또는 pgAdmin을 실행해보세요")
    print("2. 데이터베이스 'stock_ai'가 생성되어 있나요?")
    print("3. 비밀번호가 정확한가요?")
    print("4. 포트가 5432가 맞나요?")
    
except Exception as e:
    print(f"\n❌ 오류: {e}")



