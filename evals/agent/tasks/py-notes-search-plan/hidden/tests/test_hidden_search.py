def test_search_case_insensitive(run):
    run("add", "Buy MILK", "--tag", "home")
    run("add", "Call Bob")
    run("add", "milkshake recipe")
    code, out = run("search", "milk")
    assert code == 0
    assert out.splitlines() == ["#1 [home] Buy MILK", "#3 milkshake recipe"]


def test_search_no_match(run):
    run("add", "something")
    code, out = run("search", "nothing")
    assert code == 1
    assert out.strip() == "no notes found"
