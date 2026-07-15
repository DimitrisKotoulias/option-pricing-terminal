import pandas as pd
from optpricing.data.market_data_fetcher import fetch_10y_historical_data
  
def test_fetch_10y_data():
    df = fetch_10y_historical_data("^SPX")
    assert not df.empty
    assert len(df) > 2000
    assert "Close" in df.columns
