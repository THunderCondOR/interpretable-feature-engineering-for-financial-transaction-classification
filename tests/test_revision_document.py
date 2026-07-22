from scripts.build_revision_document import SECTIONS, build_html, fallback_pdf


def test_revision_document_has_all_required_sections_and_inline_svg(tmp_path):
    document = build_html()
    titles = {title for title, _ in SECTIONS}
    for required in ("Train-only data boundary", "64 → cooldown → 10×10 → 64 state machine", "Publication cluster map", "Grounding", "Classifier fidelity", "Reviewer response checklist"):
        assert required in titles
        assert required in document
    assert "<svg" in document
    assert "http://" not in document and "https://" not in document
    output = tmp_path / "revision.pdf"
    fallback_pdf(output)
    assert output.read_bytes().startswith(b"%PDF")
