import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def true_positive(predict, actual):
    if predict and actual:
        return True
    return False


def true_negative(predict, actual):
    if (not predict) and (not actual):
        return True
    return False


def false_positive(predict, actual):
    if (predict) and (not actual):
        return True
    return False


def false_negative(predict, actual):
    if (not predict) and actual:
        return True
    return False


class ROC(object):
    def __init__(self, label, value) -> None:
        """ Plot the ROC curve and calculate AUC value for given data.

        Args:
            label (list): ground true list of the data
            value (list): predict value
        """
        assert(len(label) == len(value))
        self.label = label
        self.value = value
        self.dataframe = pd.DataFrame(
            {"label": self.label, "value": self.value})
        self.pos_num = np.sum(label)
        self.neg_num = len(label) - self.pos_num
        self.pos_table = self.dataframe[self.dataframe.label == 1].reset_index(
            drop=True)
        self.neg_table = self.dataframe[self.dataframe.label == 0].reset_index(
            drop=True)

    def ROC_curve(self, is_plot=True):
        sorted_table = self.dataframe.sort_values(
            by="value", ascending=False).reset_index(drop=True)
        for col in ["TP", "TN", "FP", "FN", "FPR", "TPR"]:
            sorted_table[col] = 0
        for i in range(sorted_table.shape[0]):
            TP, TN, FP, FN = 0, 0, 0, 0
            for j in range(sorted_table.shape[0]):
                estimate = False
                if j <= i:
                    estimate = True
                actual = sorted_table.loc[j, "label"] == 1
                if true_positive(estimate, actual):
                    TP += 1
                if true_negative(estimate, actual):
                    TN += 1
                if false_negative(estimate, actual):
                    FN += 1
                if false_positive(estimate, actual):
                    FP += 1
            sorted_table.loc[i, "TP"] = TP
            sorted_table.loc[i, "TN"] = TN
            sorted_table.loc[i, "FP"] = FP
            sorted_table.loc[i, "FN"] = FN
            sorted_table.loc[i, "TPR"] = TP / (TP + FN)
            sorted_table.loc[i, "FPR"] = FP / (TN + FP)
        sorted_table.sort_values(by="FPR").reset_index(drop=True)
        if is_plot:
            plot.plot(sorted_table.FPR, sorted_table.TPR)
            plot.show()
        return sorted_table

    @property
    def auc_score(self):
        cnt = 0
        for i in range(self.pos_num):
            for j in range(self.neg_num):
                if self.pos_table.loc[i, "value"] > self.neg_table.loc[j, "value"]:
                    cnt += 1
        return cnt / self.pos_num / self.neg_num


if __name__ == "__main__":
    predict = [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51,
               0.505, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.3, 0.1]
    ground_truth = [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
    roc = ROC(label=ground_truth, value=predict)
    roc.ROC_curve()

    fpr, tpr, thresholds = roc_curve(
        np.array(ground_truth), np.array(predict), pos_label=1)
    print(f"Our method of AUC value: {roc.auc_score}")
    print(f"sklearn's AUC value: {auc(fpr, tpr)}")
