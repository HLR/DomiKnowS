import sys
import pytest
import csp_demo

def test_ILP_exactL(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["csp_demo","--constraint","exactL"])
    csp_demo.main()
    out = capsys.readouterr().out
    assert not "ERROR" in out

