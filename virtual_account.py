"""
가상 계좌 및 주문/포지션 관리를 담당하는 모듈.

단순 롱 전용 전략을 가정:
- MARKET_BUY: 현재가로 매수
- MARKET_SELL: 현재가로 매도 (보유 수량 한도)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Position:
    """단일 종목 포지션 정보"""

    stock_code: str
    quantity: int = 0
    avg_price: float = 0.0


@dataclass
class Trade:
    """체결 기록"""

    datetime: datetime
    stock_code: str
    side: str  # "BUY" or "SELL"
    price: float
    quantity: int
    fee: float
    pnl: float  # 실현 손익


@dataclass
class EquityPoint:
    """시간별 계좌 가치"""

    datetime: datetime
    equity: float


@dataclass
class VirtualAccount:
    """
    간단한 가상 계좌 클래스 (롱 전용).

    - 수수료는 체결 금액 * fee_rate 로 계산.
    - 슬리피지는 현재 단계에서 미적용(필요 시 확장).
    """

    initial_cash: float
    fee_rate: float = 0.00015  # 0.015% 정도 예시
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[EquityPoint] = field(default_factory=list)

    def __post_init__(self):
        self.cash = float(self.initial_cash)

    # ------------------------------------------------------------------
    # 포지션 및 주문 관련
    # ------------------------------------------------------------------
    def get_position(self, stock_code: str) -> Position:
        if stock_code not in self.positions:
            self.positions[stock_code] = Position(stock_code=stock_code)
        return self.positions[stock_code]

    def buy_market(self, dt: datetime, stock_code: str, price: float, quantity: int):
        """
        시장가 매수 (롱 진입/증액).
        """
        if quantity <= 0:
            return

        cost = price * quantity
        fee = cost * self.fee_rate
        total_cost = cost + fee

        if total_cost > self.cash:
            # 현금 부족 시 가능한 최대 수량으로 조정하거나 무시
            return

        pos = self.get_position(stock_code)

        # 새로운 평균단가 계산
        new_qty = pos.quantity + quantity
        if new_qty > 0:
            pos.avg_price = (pos.avg_price * pos.quantity + cost) / new_qty
        pos.quantity = new_qty

        self.cash -= total_cost

        trade = Trade(
            datetime=dt,
            stock_code=stock_code,
            side="BUY",
            price=price,
            quantity=quantity,
            fee=fee,
            pnl=0.0,
        )
        self.trades.append(trade)

    def sell_market(self, dt: datetime, stock_code: str, price: float, quantity: Optional[int] = None):
        """
        시장가 매도 (롱 청산/감축).
        quantity=None 이면 전량 매도.
        """
        pos = self.get_position(stock_code)
        if pos.quantity <= 0:
            return

        sell_qty = pos.quantity if quantity is None else min(quantity, pos.quantity)
        if sell_qty <= 0:
            return

        proceeds = price * sell_qty
        fee = proceeds * self.fee_rate

        # 실현 손익
        pnl = (price - pos.avg_price) * sell_qty - fee

        pos.quantity -= sell_qty
        if pos.quantity == 0:
            pos.avg_price = 0.0

        self.cash += proceeds - fee

        trade = Trade(
            datetime=dt,
            stock_code=stock_code,
            side="SELL",
            price=price,
            quantity=sell_qty,
            fee=fee,
            pnl=pnl,
        )
        self.trades.append(trade)

    # ------------------------------------------------------------------
    # 평가 손익 및 에쿼티
    # ------------------------------------------------------------------
    def compute_equity(self, prices: Dict[str, float]) -> float:
        """
        현재 시점의 계좌 총 가치를 계산.
        prices: {stock_code: last_price}
        """
        equity = self.cash
        for code, pos in self.positions.items():
            if pos.quantity > 0 and code in prices:
                equity += pos.quantity * prices[code]
        return equity

    def record_equity(self, dt: datetime, prices: Dict[str, float]):
        equity = self.compute_equity(prices)
        self.equity_curve.append(EquityPoint(datetime=dt, equity=equity))

    # ------------------------------------------------------------------
    # 요약 통계
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, float]:
        """
        기본 요약 통계 반환.
        - 최종 에쿼티
        - 총 수익률
        - 거래 횟수 / 승률
        """
        if not self.equity_curve:
            final_equity = self.cash
        else:
            final_equity = self.equity_curve[-1].equity

        total_return = (final_equity - self.initial_cash) / self.initial_cash

        wins = sum(1 for t in self.trades if t.pnl > 0)
        losses = sum(1 for t in self.trades if t.pnl < 0)
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        return {
            "initial_cash": self.initial_cash,
            "final_equity": final_equity,
            "total_return": total_return,
            "num_trades": total_trades,
            "win_rate": win_rate,
        }














