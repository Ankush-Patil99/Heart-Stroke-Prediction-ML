from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
import joblib


def get_library_models():

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=None
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=4,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.01,
        scale_pos_weight=19,
        subsample=1.0,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )

    return lr, rf, xgb


def build_ensembles(lr, rf, xgb):

    voting = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("rf", rf),
            ("xgb", xgb)
        ],
        voting="soft"
    )

    stacking = StackingClassifier(
        estimators=[
            ("lr", lr),
            ("rf", rf)
        ],
        final_estimator=xgb,
        passthrough=True,
        n_jobs=-1
    )

    return voting, stacking


def save_model(model, path):

    joblib.dump(model, path)


def load_model(path):

    return joblib.load(path)
