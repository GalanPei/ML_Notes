from ..utils import ROC as roc
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from utils.diff import relative_error


def roc_test_case(case_num, case_dim, threshold):
    case_num = 100
    case_dim = 100
    for _ in range(case_num):
        predict = np.random.randn(case_dim)
        ground_truth = np.where(np.random.rand(case_dim) > .5, 1, 0)
        _roc = roc.ROC(label=ground_truth, predict=predict, pos_label=1)
        fpr, tpr, _ = roc_curve(
            ground_truth, predict, pos_label=1)
        ret = _roc.auc_score
        golden = auc(fpr, tpr)
        if relative_error(ret, golden) > threshold:
            return False
    return True


if __name__ == "__main__":
    assert(roc_test_case(100, 100, .04))
