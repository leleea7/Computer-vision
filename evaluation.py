import numpy as np

def accuracy(y_true, y_pred):
    assert(len(y_true) == len(y_pred))
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    return correct / len(y_true)

def confusion_matrix(y_true, y_pred, num_classes):
    assert(len(y_true) == len(y_pred))
    conf_mat = np.zeros(shape=(num_classes, num_classes), dtype=np.int32)
    for true, pred in zip(y_true, y_pred):
        conf_mat[true][pred] += 1
    return conf_mat
