from main import _strip_markdown


def test_strips_headings():
    assert _strip_markdown("# About\n\nSome text.") == "About Some text."
    assert _strip_markdown("## Experience\nMore text.") == "Experience More text."
    assert _strip_markdown("### Sub Heading\ntext") == "Sub Heading text"


def test_strips_bullet_markers_but_keeps_content():
    text = "- First point\n- Second point\n* Third point"
    result = _strip_markdown(text)
    assert "-" not in result.split()
    assert "First point" in result
    assert "Second point" in result
    assert "Third point" in result


def test_strips_bold_and_italic_markers():
    assert _strip_markdown("I build **AI** systems.") == "I build AI systems."
    assert _strip_markdown("I build __AI__ systems.") == "I build AI systems."
    assert _strip_markdown("Some *italic* text.") == "Some italic text."


def test_strips_inline_code_backticks():
    assert _strip_markdown("I use `Python` daily.") == "I use Python daily."


def test_leaves_plain_prose_unchanged():
    text = "I have experience as an AI/ML Engineer at Chase, working on RAG and MLOps."
    assert _strip_markdown(text) == text


def test_empty_string_stays_empty():
    assert _strip_markdown("") == ""


def test_collapses_multiple_document_sections_into_flowing_text():
    messy = "# About\n\nI build things.\n\n## Skills\n\n- Python\n- SQL"
    result = _strip_markdown(messy)
    assert "#" not in result
    assert "I build things." in result
    assert "Python" in result
    assert "SQL" in result
