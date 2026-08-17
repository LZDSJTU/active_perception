"""Regression thresholds for visual and kinematic acceptance."""
MAX_PRECONTACT_DRIFT_M = .001
MAX_PREGRASP_DRIFT_M = .001
MAX_POST_RELEASE_DRIFT_M = .001
MIN_VERTICAL_ALIGNMENT = .97
MIN_CARRY_ATTACHMENT_CONTINUITY = .95

def test_acceptance_thresholds_are_strict():
    assert MAX_PRECONTACT_DRIFT_M <= .001
    assert MAX_PREGRASP_DRIFT_M <= .001
    assert MAX_POST_RELEASE_DRIFT_M <= .001
    assert MIN_VERTICAL_ALIGNMENT >= .97
    assert MIN_CARRY_ATTACHMENT_CONTINUITY >= .95

def test_contact_is_not_replaced_by_position_tolerance():
    # Push contact may terminate on physical bilateral contact even though the
    # controller EEF site is offset from the finger pads. This threshold must
    # never be used as a substitute for contact verification.
    assert MAX_PRECONTACT_DRIFT_M < .002
