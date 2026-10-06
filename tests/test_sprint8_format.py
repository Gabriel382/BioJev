
from biojev.systemone_native.format import normalize_record, render_messages

def test_normalize_answer_key_after_reordering():
    r=normalize_record({
        "state":"x","question":"q?",
        "options":[{"key":"b","description":"B"},{"key":"a","description":"A"}],
        "answer_key":"a",
    })
    assert [o["label"] for o in r["options"]]==["A","B"]
    assert r["answer"]=="B"
    assert render_messages(r)[-1]["content"]=="B"
