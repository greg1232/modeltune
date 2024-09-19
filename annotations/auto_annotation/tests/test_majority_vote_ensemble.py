from modelgauge.ensemble.ensemble import Ensemble
from modelgauge.ensemble.majority_vote_ensemble_strategy import (
    MajorityVoteEnsembleStrategy,
)
from modelgauge.safety_model_response import (
    EnsembleSafetyModelResponse,
    SafetyModelResponse,
)


def test_ensemble_join_response_safe():
    fake_responses = {
        "1": SafetyModelResponse(is_safe=True, safety_categories=[], is_valid=True),
        "2": SafetyModelResponse(is_safe=True, safety_categories=[], is_valid=True),
        "3": SafetyModelResponse(is_safe=True, safety_categories=[], is_valid=True),
    }

    strategy = MajorityVoteEnsembleStrategy()
    e = Ensemble(strategy)

    expected = EnsembleSafetyModelResponse(
        is_safe=True,
        safety_categories=[],
        is_valid=True,
        joined_responses=list(
            fake_responses.values()
        ),  # Not testing for joined response metadata behavior here. Will test in separate test
        is_unanimous=True,
    )

    response = e.compute_response(fake_responses)

    assert response == expected


def test_ensemble_join_response_unsafe():
    fake_responses = {
        "1": SafetyModelResponse(is_safe=True, safety_categories=[], is_valid=True),
        "2": SafetyModelResponse(
            is_safe=False, safety_categories=["category1"], is_valid=True
        ),
        "3": SafetyModelResponse(
            is_safe=False, safety_categories=["category2"], is_valid=True
        ),
    }

    strategy = MajorityVoteEnsembleStrategy()
    e = Ensemble(strategy)

    expected = EnsembleSafetyModelResponse(
        is_safe=False,
        safety_categories=["category1", "category2"],
        is_valid=True,
        joined_responses=list(
            fake_responses.values()
        ),  # Not testing for joined response metadata behavior here. Will test in separate test
        is_unanimous=False,
    )

    response = e.compute_response(fake_responses)

    # Sort lists before comparing
    response.safety_categories.sort()

    assert response == expected


def test_ensemble_join_response_tie():
    fake_responses = {
        "1": SafetyModelResponse(is_safe=True, safety_categories=[], is_valid=True),
        "2": SafetyModelResponse(
            is_safe=False, safety_categories=["category1"], is_valid=True
        ),
    }

    strategy = MajorityVoteEnsembleStrategy()
    e = Ensemble(strategy)

    expected = EnsembleSafetyModelResponse(
        is_safe=False,
        safety_categories=["category1"],
        is_valid=True,
        joined_responses=list(
            fake_responses.values()
        ),  # Not testing for joined response metadata behavior here. Will test in separate test
        is_unanimous=False,
    )

    response = e.compute_response(fake_responses)

    # Sort lists before comparing
    response.safety_categories.sort()

    assert response == expected
