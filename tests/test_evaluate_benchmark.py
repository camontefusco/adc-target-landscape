import numpy as np
import pandas as pd
from scripts.evaluate_benchmark import evaluate

def test_perfect_ranking():
    rows = []
    for patient in ["p1", "p2"]:
        for target, value in zip(["A", "B", "C"], [3, 2, 1]):
            rows.append({"patient_id": patient, "target": target, "y_true": value, "y_pred": value})
    patient, summary = evaluate(pd.DataFrame(rows))
    assert np.isclose(patient["pairwise_accuracy"].mean(), 1.0)
    assert np.isclose(patient["top1_correct"].mean(), 1.0)
    assert np.isclose(patient["top2_recall"].mean(), 1.0)
