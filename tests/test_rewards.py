from triage_grpo.rewards.outcome import extract_boxed_answer, score_outcome_exact
from triage_grpo.rewards.rule_process import RuleProcessScorer


def test_extract_boxed():
    text = r"reason...\n\\boxed{42}"
    assert extract_boxed_answer(text) == "42"


def test_score_outcome():
    resp = r"After calculation, \\boxed{7}"
    assert score_outcome_exact(resp, "7") == 1.0
    assert score_outcome_exact(resp, "8") == 0.0


def test_rule_process_in_unit_interval():
    scorer = RuleProcessScorer()
    s = scorer.score(
        "1+1=?",
        "Step 1: add\nStep 2: get 2\n\\boxed{2}",
    )
    assert 0.0 <= s <= 1.0
    assert s > 0.5
