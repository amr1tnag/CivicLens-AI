from civiclens.nlp import lexicons


def test_critical_text_scores_higher_than_low_urgency_text():
    critical = lexicons.analyze(
        "This is a fire hazard, live wires are sparking and someone could be electrocuted, please treat as an emergency!"
    )
    low = lexicons.analyze(
        "Just a minor suggestion, no rush, whenever convenient it would be nice to repaint the fence."
    )
    assert critical.urgency_score_normalized > low.urgency_score_normalized
    assert critical.urgency_score_normalized > 0.8
    assert low.urgency_score_normalized < 0.4


def test_matched_urgent_terms_are_detected():
    result = lexicons.analyze("There was an explosion and the building is on fire, danger to life.")
    assert "explosion" in result.matched_urgent_terms
    assert "fire" in result.matched_urgent_terms


def test_dampener_terms_reduce_score():
    with_dampener = lexicons.analyze("This is a minor issue, not urgent, whenever convenient.")
    assert with_dampener.matched_dampener_terms  # at least one dampener matched
    assert with_dampener.urgency_score_normalized < 0.5


def test_frustration_words_detected_in_sentiment():
    result = lexicons.analyze("This is absolutely unacceptable and disgusting, I am furious.")
    assert result.sentiment_score > 0
    assert "unacceptable" in result.matched_frustration_terms


def test_repeat_mention_detected():
    result = lexicons.analyze("This is the second time I'm reporting this issue.")
    assert result.has_repeat_mention is True


def test_duration_mentions_extracted():
    result = lexicons.analyze("The water supply has been off for three days now.")
    assert any("day" in d for d in result.duration_mentions)


def test_explain_returns_expected_keys():
    explanation = lexicons.explain("There is a dangerous pothole on the road.")
    assert set(explanation.keys()) == {
        "urgency_score_normalized",
        "sentiment_score",
        "matched_urgent_terms",
        "matched_dampener_terms",
        "matched_frustration_terms",
        "has_repeat_mention",
        "duration_mentions",
    }
