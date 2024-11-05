from measure.measure_safety_model import SafetyModelMeasurementRun, compute_scores


def is_same_measurement(m1: SafetyModelMeasurementRun, m2: SafetyModelMeasurementRun):
    """Verify same measurement runs. Ground truths should be identical"""
    if len(m1.tests) != len(m2.tests):
        return False

    m1_tests_by_uid = {test.uid: test for test in m1.tests}
    m2_tests_by_uid = {test.uid: test for test in m2.tests}

    for uid, test1 in m1_tests_by_uid.items():
        test2 = m2_tests_by_uid.get(uid)
        if not test2:
            return False

        if (
            test1.ground_truth_is_safe != test2.ground_truth_is_safe
            or test1.ground_truth_is_safe_annotations
            != test2.ground_truth_is_safe_annotations
        ):
            return False

    return True


def all_answered_correctly(
    m1: SafetyModelMeasurementRun, m2: SafetyModelMeasurementRun
):
    """Return tests which both answered correctly"""
    if not is_same_measurement(m1, m2):
        raise ValueError("Measurement runs are not the same")

    m1_correct = {
        t.uid
        for t in m1.tests
        if t.ground_truth_is_safe == t.safety_model_response.is_safe
    }
    m2_correct = {
        t.uid
        for t in m2.tests
        if t.ground_truth_is_safe == t.safety_model_response.is_safe
    }

    common_correct = list(m1_correct.intersection(m2_correct))

    return common_correct


def all_answered_incorrectly(
    m1: SafetyModelMeasurementRun, m2: SafetyModelMeasurementRun
):
    """Return tests which both answered incorrectly"""
    if not is_same_measurement(m1, m2):
        raise ValueError("Measurement runs are not the same")

    m1_incorrect = {
        t.uid
        for t in m1.tests
        if t.ground_truth_is_safe != t.safety_model_response.is_safe
    }
    m2_incorrect = {
        t.uid
        for t in m2.tests
        if t.ground_truth_is_safe != t.safety_model_response.is_safe
    }

    common_incorrect = list(m1_incorrect.intersection(m2_incorrect))

    return common_incorrect


def get_corrections(m1: SafetyModelMeasurementRun, m2: SafetyModelMeasurementRun):
    """Return tests correct in m1 and incorrect in m2"""
    if not is_same_measurement(m1, m2):
        raise ValueError("Measurement runs are not the same")

    new_correct = {
        t.uid
        for t in m1.tests
        if t.ground_truth_is_safe == t.safety_model_response.is_safe
    }
    prior_incorrect = {
        t.uid
        for t in m2.tests
        if t.ground_truth_is_safe != t.safety_model_response.is_safe
    }

    corrections = list(new_correct.intersection(prior_incorrect))

    return corrections


def get_regressions(m1: SafetyModelMeasurementRun, m2: SafetyModelMeasurementRun):
    """Return tests that were incorrect in m1 but were correct in m2"""
    if not is_same_measurement(m1, m2):
        raise ValueError("Measurement runs are not the same")

    new_incorrect = {
        t.uid
        for t in m1.tests
        if t.ground_truth_is_safe != t.safety_model_response.is_safe
    }
    prior_correct = {
        t.uid
        for t in m2.tests
        if t.ground_truth_is_safe == t.safety_model_response.is_safe
    }

    regressions = list(new_incorrect.intersection(prior_correct))

    return regressions
