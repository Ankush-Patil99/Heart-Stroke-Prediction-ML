from sklearn.metrics import brier_score_loss


def compute_brier(model, X_test, y_test):
    if not hasattr(model, "predict_proba"):
        print("Model does not support predict_proba. Skipping Brier score.")
        return None

    y_prob = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_prob)

    return brier