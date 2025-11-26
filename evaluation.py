from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Some custom models may not have predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = None

    results = {
        "roc_auc": roc_auc,
        "classification_report": classification_report(y_test, y_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return results
