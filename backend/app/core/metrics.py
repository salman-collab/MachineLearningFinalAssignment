def confusion_matrix(y_true, y_pred):
    tp = tn = fp = fn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


def accuracy(y_true, y_pred):
    correct = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1

    return correct / len(y_true)


def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm["TP"]
    fp = cm["FP"]

    if tp + fp == 0:
        return 0

    return tp / (tp + fp)


def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm["TP"]
    fn = cm["FN"]

    if tp + fn == 0:
        return 0

    return tp / (tp + fn)