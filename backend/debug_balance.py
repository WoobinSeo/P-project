from __future__ import annotations

import json

from backend.kis_broker import KISBroker


def main():
    b = KISBroker()
    res = b.get_balance()
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




