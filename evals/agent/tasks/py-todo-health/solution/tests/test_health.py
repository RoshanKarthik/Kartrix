def test_health(call):
    assert call("GET", "/health") == (200, {"status": "ok"})
