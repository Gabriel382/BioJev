from biojev.corpus.packing import PackedTokenStream, estimate_sequence_count


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids": [ord(c) % 50 for c in text]}


def test_sequence_count_rounds_up():
    assert estimate_sequence_count(100, 32) == 4


def test_packer_respects_exact_budget():
    docs = [{"text": "abcdefghij"}, {"text": "klmnopqrst"}, {"text": "uvwxyz"}]
    rows = list(PackedTokenStream(docs, FakeTokenizer(), token_budget=17, sequence_length=8))
    assert sum(len(row["input_ids"]) for row in rows) == 17
    assert [len(row["input_ids"]) for row in rows] == [8, 8, 1]
    assert all(row["labels"] == row["input_ids"] for row in rows)
