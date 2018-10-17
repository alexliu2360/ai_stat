# coding=utf-8
# ------------------------------------------------------------
#  版本：0.1
#  版权：****
#  模块：****
#  功能：****
#  语言：Python2.7
#  作者：******
#  日期：2018-08-25
# ------------------------------------------------------------
#  修改人：******
#  修改日期：2018-08-26
#  修改内容：创建
# ------------------------------------------------------------
import keras.backend as K


def f1(y_true, y_pred):
    def recall(y_true_, y_pred_):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true_ * y_pred_, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true_, 0, 1)))
        recall_ = true_positives / (possible_positives + K.epsilon())
        return recall_

    def precision(y_true_, y_pred_):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true_ * y_pred_, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_, 0, 1)))
        precision_ = true_positives / (predicted_positives + K.epsilon())
        return precision_

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
