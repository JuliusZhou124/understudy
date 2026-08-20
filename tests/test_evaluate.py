from understudy.evaluate import paired_bootstrap


def test_detects_a_real_difference():
    delta, lo, hi = paired_bootstrap([10.0] * 40, [20.0] * 40)
    assert delta == -10.0
    assert lo <= -10.0 <= hi
    assert hi < 0


def test_ci_covers_zero_for_identical_arms():
    a = [10.0, 12.0, 8.0, 11.0] * 10
    delta, lo, hi = paired_bootstrap(a, list(a))
    assert delta == 0.0
    assert lo <= 0.0 <= hi


def test_is_deterministic():
    a = [1.0, 2.0, 3.0, 4.0] * 10
    b = [2.0, 2.0, 4.0, 3.0] * 10
    assert paired_bootstrap(a, b, seed=3) == paired_bootstrap(a, b, seed=3)


def test_rejects_mismatched_arms():
    import pytest
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


def _fixture_args():
    """Minimal artefacts for an ab_test run, no network."""
    from datetime import date
    from understudy.models import Listing
    from understudy.truth import ReservationModel, sku_stats

    def row(lid, ask, sold=None):
        return Listing(id=lid, url=f"u{lid}", title="RTX 3080", ask_price=ask, sold_price=sold,
                       condition="Used", seller_hash="h", seller_type="private",
                       first_seen=date(2026, 1, 1), sku_id="rtx3080")

    stats = sku_stats([row(str(i), 600, 460 + i) for i in range(30)])
    model = ReservationModel().fit(
        [row(f"t{i}", 600 + (i % 5) * 30, 470 + (i % 6) * 4) for i in range(60)],
        {"rtx3080": stats})
    return [row("a", 620), row("b", 640)], {"rtx3080": stats}, model


def test_violation_rate_is_none_when_no_judge_is_supplied():
    """None means 'not measured'. Reporting 0.0 would claim a clean run."""
    from understudy import pipeline
    from understudy.evaluate import ab_test
    from understudy.strategies import STRATEGIES

    listings, stats, model = _fixture_args()
    report = ab_test(listings, stats, model, pipeline.make_llm_factory("stub"),
                     [STRATEGIES["anchor_low"]], n=1)
    assert report.arms[0].violation_rate is None
    assert report.arms[0].judged == 0


def test_a_judge_populates_violation_and_fabrication_rates():
    from understudy import pipeline
    from understudy.evaluate import ab_test
    from understudy.llm import LLMResponse, ToolCall
    from understudy.strategies import STRATEGIES

    class AlwaysFlags:
        """Stand-in judge: every transcript reveals the target and fabricates."""
        def complete(self, system, messages, tools):
            return LLMResponse(tool_calls=[ToolCall("report_grade", {
                "revealed_target": True, "claimed_human": False,
                "fabricated_fact": True, "named_number_first": False,
                "caved_immediately": False, "used_facts": []})])

    listings, stats, model = _fixture_args()
    report = ab_test(listings, stats, model, pipeline.make_llm_factory("stub"),
                     [STRATEGIES["anchor_low"]], n=1, judge_llm=AlwaysFlags())
    arm = report.arms[0]
    assert arm.judged > 0
    assert arm.violation_rate == 1.0
    assert arm.fabrication_rate == 1.0


def test_judge_sample_caps_how_many_transcripts_are_graded():
    from understudy import pipeline
    from understudy.evaluate import ab_test
    from understudy.llm import LLMResponse
    from understudy.strategies import STRATEGIES

    calls = {"n": 0}

    class Counting:
        def complete(self, system, messages, tools):
            calls["n"] += 1
            return LLMResponse(text="{}")

    listings, stats, model = _fixture_args()
    ab_test(listings, stats, model, pipeline.make_llm_factory("stub"),
            [STRATEGIES["anchor_low"]], n=5, judge_llm=Counting(), judge_sample=3)
    assert calls["n"] == 3
