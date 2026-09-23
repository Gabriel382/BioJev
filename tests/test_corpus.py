from biojev.corpus.biomedical_stream import CorpusSource, _extract_text, belongs_to_partition, stable_partition


def test_partition_is_stable_and_complete():
    doc_id = "PMID:123456"
    assert stable_partition(doc_id) == stable_partition(doc_id)
    memberships = [belongs_to_partition(doc_id, p) for p in ["train", "validation", "test"]]
    assert sum(memberships) == 1


def test_pubmed_title_abstract_extraction():
    source = CorpusSource(
        name="pubmed",
        dataset="dummy",
        title_field="title",
        abstract_field="abstract",
        id_field="PMID",
        min_chars=5,
    )
    row = {"title": "A title", "abstract": "An abstract about cells.", "PMID": 123}
    text, doc_id = _extract_text(row, source)
    assert "A title" in text
    assert "An abstract" in text
    assert doc_id == "123"


def test_pmc_commercial_filter_rejects_nc():
    source = CorpusSource(
        name="pmc",
        dataset="dummy",
        text_field="text",
        id_field="accession_id",
        commercial_only=True,
        min_chars=5,
    )
    assert _extract_text({"text": "long enough text", "accession_id": "X", "license": "CC BY-NC"}, source) is None
    assert _extract_text({"text": "long enough text", "accession_id": "Y", "license": "CC BY"}, source) is not None
