import numpy as np

from adc_landscape.analysis import (
    mean_pool_patch_embeddings,
    patient_ranking_metrics,
    patient_target_zscores,
)


def test_mean_pool():
    x = np.array([[1.0, 3.0], [3.0, 5.0]])
    assert np.allclose(mean_pool_patch_embeddings(x), [2.0, 4.0])


def test_target_zscore_uses_training_stats():
    train = np.array([[0.0, 10.0], [2.0, 14.0]])
    test = np.array([[4.0, 18.0]])
    ztr, zte = patient_target_zscores(train, test)
    assert np.allclose(ztr.mean(axis=0), 0.0)
    assert np.allclose(zte, [[3.0, 3.0]])


def test_perfect_ranking():
    y = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    out = patient_ranking_metrics(y, y.copy())
    assert np.allclose(out.spearman, 1.0)
    assert np.allclose(out.kendall_tau, 1.0)
    assert np.allclose(out.pairwise_accuracy, 1.0)
    assert out.top1_correct.eq(1).all()
    assert out.top2_correct.eq(1).all()
