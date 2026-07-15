from optpricing.data.market_data_fetcher import fetch_risk_free_rate, fetch_dividend_yield
  
def test_fetch_market_inputs():
    r = fetch_risk_free_rate()
    assert 0.0 < r < 0.15
    q = fetch_dividend_yield("^SPX")
    assert 0.0 <= q < 0.10
