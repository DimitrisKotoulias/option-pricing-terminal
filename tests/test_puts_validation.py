from optpricing import PutCallParity
  
def test_parity_calculation():
    res = PutCallParity.verify(
        call_price=10.05,
        put_price=8.02,
        S=100.0,
        K=100.0,
        r=0.05,
        T=1.0,
        q=0.02
    )
    assert "parity_holds" in res
    assert "difference" in res
    assert "lhs" in res
    assert "rhs" in res
