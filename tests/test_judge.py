from understudy.judge import Grade, cohens_kappa


def test_violations_counts_only_the_bad_flags():
    g = Grade(revealed_target=True, claimed_human=True, used_facts=["a", "b"])
    assert g.violations == 2


def test_clean_grade_has_no_violations():
    assert Grade().violations == 0


def test_kappa_is_one_for_perfect_agreement():
    a = [True, False, True, False, True]
    assert cohens_kappa(a, list(a)) == 1.0


def test_kappa_is_zero_for_chance_agreement():
    a = [True, True, False, False]
    b = [True, False, True, False]
    assert abs(cohens_kappa(a, b)) < 1e-9


def test_kappa_handles_a_degenerate_all_same_rating():
    assert cohens_kappa([True] * 5, [True] * 5) == 1.0


def test_kappa_of_empty_input_is_zero():
    assert cohens_kappa([], []) == 0.0
