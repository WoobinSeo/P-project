"""
시간 단위 자동 매매 스크립트 (SAC + KIS 모의투자).

설계:
  - 학습된 SAC 정책을 이용해 1시간마다 종목별 매수/매도 여부를 결정
  - 의사결정 로직은 단순 데모용:
      action > 0.3  →  1주 시장가 매수
      action < -0.3 →  1주 시장가 매도
      그 외         →  관망
  - 실제 운용에서는
      - 포지션 한도
      - 종목별 자산 비중
      - 주문 시간(장 개시 후 N분 등)
    등을 추가하는 것이 좋다.

사용 예시 (Windows 스케줄러/cron 에 등록):
  python auto_trader.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from kis_broker import KISBroker
from trading_api import check_risk_limit
from database import DatabaseManager, StockPriceProcessed


@dataclass
class StockConfig:
    name: str          # "삼성전자"
    code: str          # "005930"
    model_path: str    # SAC 모델 경로
    window_size: int = 60


STOCKS: List[StockConfig] = [
    StockConfig(name="삼성전자", code="005930", model_path="rl_models/sac_삼성전자_best/best_model"),
    StockConfig(name="네이버", code="035420", model_path="rl_models/sac_네이버_best/best_model"),
    # 현대차는 final 모델이 더 성능이 좋아서 final 사용
    StockConfig(name="현대차", code="005380", model_path="rl_models/sac_현대차_final"),
]


_db_manager: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """
    자동매매 스크립트 전용 DB 매니저.

    - trading_api.py 의 get_db 와 동일한 패턴
    - 실시간에 가까운 피처를 위해 StockPriceProcessed 를 조회할 때 사용
    """
    global _db_manager
    if _db_manager is None:
        mgr = DatabaseManager()
        mgr.connect()
        mgr.create_tables()
        _db_manager = mgr
    return _db_manager


def _find_latest_preprocessed_csv(stock_name: str) -> str:
    """
    (레거시) data/preprocessed 아래에서 해당 종목의 CSV를 우선순위(test > val > train)로 찾는다.

    - 과거 버전에서는 CSV 기반 관측값을 사용했지만,
      이제는 StockPriceProcessed 테이블 기반 관측값을 기본으로 사용한다.
    - CSV 는 백테스트/디버깅 용도로만 남겨둔다.
    """
    base_dir = os.path.join("data", "preprocessed")
    candidates = [
        os.path.join(base_dir, f"{stock_name}_test.csv"),
        os.path.join(base_dir, f"{stock_name}_val.csv"),
        os.path.join(base_dir, f"{stock_name}_train.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{stock_name} 에 대한 전처리 CSV 를 찾을 수 없습니다: {candidates}")


def build_latest_observation_from_db(stock: StockConfig) -> np.ndarray:
    """
    SAC 정책에 넣을 **실시간에 가까운** 최신 시점 관측값을 DB에서 구성한다.

    - StockPriceProcessed 테이블에서 해당 종목의 최신 window_size개 행을 조회
    - 학습 시 사용한 전처리 CSV와 동일한 컬럼 구성을 가정
    - 포지션/누적수익률(col 2개)은 현재 0으로 두고, 정책이 방향성만 보도록 유지
    """
    db = get_db()
    session = db.get_session()
    try:
        rows = (
            session.query(StockPriceProcessed)
            .filter(StockPriceProcessed.stock_code == stock.code)
            .order_by(StockPriceProcessed.datetime.desc())
            .limit(stock.window_size)
            .all()
        )
    finally:
        session.close()

    if not rows or len(rows) < stock.window_size:
        raise ValueError(
            f"{stock.name} ({stock.code}) 에 대한 StockPriceProcessed 데이터가 "
            f"{stock.window_size}개 미만입니다. (현재 {len(rows) if rows else 0}개)"
        )

    # 시간 오름차순으로 정렬 (과거 → 현재)
    rows = list(reversed(rows))

    # SQLAlchemy 객체를 DataFrame 으로 변환
    records: List[Dict] = []
    for r in rows:
        record = {
            "id": r.id,
            "stock_code": r.stock_code,
            "stock_name": r.stock_name,
            "datetime": r.datetime,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
            "ma_5": r.ma_5,
            "ma_10": r.ma_10,
            "ma_20": r.ma_20,
            "ma_60": r.ma_60,
            "ema_12": r.ema_12,
            "ema_26": r.ema_26,
            "macd": r.macd,
            "macd_signal": r.macd_signal,
            "macd_hist": r.macd_hist,
            "bb_upper": r.bb_upper,
            "bb_middle": r.bb_middle,
            "bb_lower": r.bb_lower,
            "bb_width": r.bb_width,
            "bb_pctb": r.bb_pctb,
            "rsi": r.rsi,
            "stoch_k": r.stoch_k,
            "stoch_d": r.stoch_d,
            "atr": r.atr,
            "volume_ma_5": r.volume_ma_5,
            "volume_ma_20": r.volume_ma_20,
            "volume_ratio": r.volume_ratio,
            "obv": r.obv,
            "return_1d": r.return_1d,
            "log_return": r.log_return,
            "return_5d": r.return_5d,
            "return_10d": r.return_10d,
            "return_20d": r.return_20d,
            "hl_ratio": r.hl_ratio,
            "co_ratio": r.co_ratio,
        }
        records.append(record)

    df = pd.DataFrame.from_records(records)

    # 학습 시 CSV 에서 제외했던 메타컬럼과 id 를 제외하고 피처만 사용
    exclude_cols = ["id", "datetime", "stock_code", "stock_name"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feats = df[feature_cols].values.astype(np.float32)

    if feats.shape[0] < stock.window_size:
        raise ValueError(
            f"{stock.name} 데이터 길이({feats.shape[0]})가 window_size({stock.window_size}) 보다 짧습니다."
        )

    # 현재 포지션/누적수익률은 0 으로 두고, 정책이 순수 방향성만 보게 함
    position_col = np.zeros((stock.window_size, 1), dtype=np.float32)
    equity_col = np.zeros((stock.window_size, 1), dtype=np.float32)

    obs = np.concatenate([feats, position_col, equity_col], axis=1)  # (window, n_features+2)
    # SAC.predict 에 넣기 위해 배치 차원 추가 → (1, window, dim)
    return obs[np.newaxis, :, :]


def build_latest_observation(stock: StockConfig) -> np.ndarray:
    """
    SAC 정책에 넣을 최신 시점 관측값을 구성한다.

    우선 DB(StockPriceProcessed) 기반 관측값을 시도하고,
    실패 시 레거시 CSV 기반 관측값으로 폴백한다.
    """
    try:
        return build_latest_observation_from_db(stock)
    except Exception as e:
        print(f"[경고] DB 기반 관측값 생성 실패, CSV 로 폴백 합니다: {e}")

    csv_path = _find_latest_preprocessed_csv(stock.name)
    df = pd.read_csv(csv_path)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

    exclude_cols = ["datetime", "stock_code", "stock_name", "target"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feats = df[feature_cols].values.astype(np.float32)

    if len(feats) < stock.window_size:
        raise ValueError(f"{stock.name} 데이터 길이({len(feats)})가 window_size({stock.window_size}) 보다 짧습니다.")

    feats_win = feats[-stock.window_size :]

    position_col = np.zeros((stock.window_size, 1), dtype=np.float32)
    equity_col = np.zeros((stock.window_size, 1), dtype=np.float32)

    obs = np.concatenate([feats_win, position_col, equity_col], axis=1)  # (window, n_features+2)
    # SAC.predict 에 넣기 위해 배치 차원 추가 → (1, window, dim)
    return obs[np.newaxis, :, :]


def decide_order_from_action(action: float) -> str:
    """
    SAC 연속 행동값 [-1, 1] 을 단순 매매 의사로 변환.

    반환:
      "BUY" / "SELL" / "HOLD"
    """
    if action > 0.3:
        return "BUY"
    elif action < -0.3:
        return "SELL"
    return "HOLD"


def run_hourly_trading():
    """
    1시간마다 실행하는 자동 매매 루틴.

    - 각 종목별로 SAC 정책을 로드
    - 최신 관측값으로 액션을 계산
    - 간단한 규칙에 따라 1주 시장가 매수/매도 실행
    """
    print("\n==============================")
    print("  StuckAI 시간 단위 자동 매매 시작 (1시간 주기 가정)")
    print("==============================\n")

    broker = KISBroker()

    for stock in STOCKS:
        print(f"\n[{stock.name}] 모델 로드 및 액션 계산 중...")
        if not os.path.exists(stock.model_path + ".zip") and not os.path.exists(stock.model_path):
            print(f"  경고: 모델 파일을 찾을 수 없습니다: {stock.model_path}")
            continue

        # env 없이도 predict 는 가능하므로 env=None 으로 로드
        # (학습 시와 동일한 설정으로 모델을 로드한다)
        model = SAC.load(stock.model_path)

        try:
            obs = build_latest_observation(stock)
        except Exception as e:
            print(f"  관측값 생성 실패: {e}")
            continue

        action_arr, _ = model.predict(obs, deterministic=True)
        # action_arr 는 shape (1, 1) 로 가정
        action = float(action_arr[0])
        decision = decide_order_from_action(action)

        print(f"  SAC 행동값: {action:.3f} → 의사결정: {decision}")

        if decision == "HOLD":
            print("  → 오늘은 관망 (주문 없음)")
            continue

        # 데모용: 항상 1주 기준으로만 매수/매도
        quantity = 1

        # 리스크 한도 체크 (트레이딩 API와 동일한 규칙 사용)
        try:
            check_risk_limit(broker, stock_code=stock.code, side=decision, quantity=quantity)
        except Exception as e:
            print(f"  리스크 한도 초과로 주문 스킵: {e}")
            continue

        try:
            if decision == "BUY":
                res = broker.buy_market(stock_code=stock.code, quantity=quantity)
            else:
                res = broker.sell_market(stock_code=stock.code, quantity=quantity)
            print(f"  주문 성공: {decision} {stock.code} x {quantity}주")
            print(f"  KIS 응답 요약: rt_cd={res.get('rt_cd')}, msg1={res.get('msg1')}")
        except Exception as e:
            print(f"  주문 실패: {e}")

    print("\n==============================")
    print("  시간 단위 자동 매매 루틴 종료")
    print("==============================\n")


if __name__ == "__main__":
    # OS 스케줄러(Windows 작업 스케줄러, cron 등)나
    # 리눅스에서는 cron 에서 이 스크립트를 1시간마다 호출하면
    # 시간 단위 자동 매매가 된다.
    run_hourly_trading()


