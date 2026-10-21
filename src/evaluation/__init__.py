"""
Evaluation metrics, error analysis, and validation test suites.
"""
from src.evaluation.metrics import evaluate_classification_model, evaluate_threshold_sweep
from src.evaluation.error_analysis import (
    get_top_features,
    analyze_prediction_errors,
    analyze_false_positives_by_scenario,
)
from src.evaluation.test_cases import (
    SMS_UNSEEN_TEST_CASES,
    ICFD_BASELINE_TEST_CASES,
    SANITY_CHECK_TEST_CASES,
    CONTRAST_TEST_PAIRS,
)

__all__ = [
    "evaluate_classification_model",
    "evaluate_threshold_sweep",
    "get_top_features",
    "analyze_prediction_errors",
    "analyze_false_positives_by_scenario",
    "SMS_UNSEEN_TEST_CASES",
    "ICFD_BASELINE_TEST_CASES",
    "SANITY_CHECK_TEST_CASES",
    "CONTRAST_TEST_PAIRS",
]
