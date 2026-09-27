from civiclens.nlp.preprocessing import (
    clean_text,
    preprocess_to_tokens,
    remove_stopwords,
    stem_tokens,
    tokenize,
)


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("There is a POTHOLE!! On MG Road...") == "there is a pothole on mg road"


def test_clean_text_strips_urls():
    assert "http" not in clean_text("See photos at https://example.com/photo.jpg please")


def test_tokenize_splits_on_non_letters():
    assert tokenize("Water-leak, near house #42!") == ["water", "leak", "near", "house"]


def test_remove_stopwords_drops_common_words_and_single_letters():
    tokens = ["the", "pothole", "is", "on", "a", "road"]
    assert remove_stopwords(tokens) == ["pothole", "road"]


def test_stem_tokens_reduces_inflections():
    assert stem_tokens(["running", "complaints", "flooding"]) == ["run", "complaint", "flood"]


def test_preprocess_to_tokens_full_pipeline():
    tokens = preprocess_to_tokens("The streetlights on MG Road have been broken for weeks!")
    assert "streetlight" in tokens or "streetlights" in tokens
    assert "the" not in tokens
    assert "on" not in tokens


def test_preprocess_to_tokens_handles_empty_string():
    assert preprocess_to_tokens("") == []
