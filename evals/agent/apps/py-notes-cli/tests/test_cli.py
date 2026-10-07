def test_add_and_list(run):
    code, out = run("add", "Call the plumber", "--tag", "Home")
    assert code == 0
    assert out.strip() == "Added #1 [home] Call the plumber"
    run("add", "Buy stamps")
    code, out = run("list")
    assert out.splitlines() == ["#1 [home] Call the plumber", "#2 Buy stamps"]


def test_list_by_tag(run):
    run("add", "a", "--tag", "work")
    run("add", "b", "--tag", "home")
    _, out = run("list", "--tag", "WORK")
    assert out.splitlines() == ["#1 [work] a"]


def test_stats(run):
    run("add", "red apple")
    run("add", "green apple pie")
    _, out = run("stats")
    lines = out.splitlines()
    assert lines[0] == "notes: 2"
    assert lines[1] == "words: 5"
    assert lines[2] == "  apple: 2"
