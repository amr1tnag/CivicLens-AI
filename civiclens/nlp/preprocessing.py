"""Text preprocessing: cleaning, tokenization, stopword removal, stemming.

Deliberately avoids NLTK's downloadable corpora (punkt, stopwords, wordnet)
so the pipeline works fully offline / air-gapped once dependencies are
`pip install`-ed. Instead it uses:
  - a regex tokenizer (letters-only word tokenization, adequate for this
    domain of short, mostly well-formed complaint text)
  - scikit-learn's bundled English stopword list (ships inside the package)
  - NLTK's PorterStemmer, which is a pure algorithm with no external data
"""

from __future__ import annotations

import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-zA-Z]+")
_STEMMER = PorterStemmer()
_STOPWORDS = frozenset(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Lowercase and strip anything that isn't a letter or whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split cleaned text into lowercase alphabetic tokens."""
    return _TOKEN_RE.findall(text.lower())


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def stem_tokens(tokens: list[str]) -> list[str]:
    return [_STEMMER.stem(t) for t in tokens]


def preprocess_to_tokens(text: str) -> list[str]:
    """Full pipeline: tokenize -> drop stopwords -> stem."""
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return tokens


def preprocess(text: str) -> str:
    """Full pipeline returning a space-joined string (handy for inspection/debugging)."""
    return " ".join(preprocess_to_tokens(text))


def vectorizer_tokenizer(text: str) -> list[str]:
    """Tokenizer callable handed to scikit-learn's Tfidf/CountVectorizer.

    Kept as a named module-level function (not a lambda) so fitted
    vectorizers remain picklable with joblib.
    """
    return preprocess_to_tokens(text)
