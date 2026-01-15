from utils.yfinance_client import YFinanceClient
from strategies.vegas import vegas_tunnel


def run_examples(ticker: str = "AAPL"):
    """
    Fetch data and print ladder and Vegas strategy signals.
    Adjust the ticker or intervals to experiment with other markets.
    """
    client = YFinanceClient()
    data = client.get_past_three_years(ticker)

    vegas_view = vegas_tunnel(data)
    multipliers = (1.618, 2.618, 4.236)
    label = lambda x: str(x).replace(".", "p")
    vegas_cols = [
        "Close",
        "ema_fast",
        "ema_slow",
        "atr",
        "signal",
        "signal_change",
    ] + [f"upper_{label(m)}" for m in multipliers] + [f"lower_{label(m)}" for m in multipliers]
    print("\nVegas Tunnel channels and signals (tail):")
    print(vegas_view[vegas_cols].tail(50))


if __name__ == "__main__":
    run_examples("DXYZ")
