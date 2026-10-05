import pytest
from biojev.systemone.schemas import SystemOneValidationError, validate_request

def test_all_types():
    r = validate_request({"model":"biojev","state":{"text":"x"},"questions":{
        "c":{"type":"choice","instructions":"Pick","criteria":{"a":"Alpha","b":None}},
        "n":{"type":"noul","instructions":"True?"},
        "s":{"type":"score","instructions":"Rate","criteria":["Low","High"]},
    }})
    assert r["questions"]["c"]["options"] == [("a","Alpha"),("b","b")]
    assert r["questions"]["n"]["options"] == [("false","No"),("true","Yes")]
    assert r["questions"]["s"]["options"] == [("0","Low"),("1","High")]

def test_bad_choice():
    with pytest.raises(SystemOneValidationError):
        validate_request({"model":"biojev","state":"x","questions":{
            "q":{"type":"choice","instructions":"Pick","criteria":{"a":"only"}}
        }})
