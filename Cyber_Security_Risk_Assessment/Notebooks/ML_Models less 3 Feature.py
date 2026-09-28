# ==========================================
# DT + RF + XGBoost
# WITHOUT Training, MFA and Backup
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Data/Processed/risk_scored_dataset.csv"
)

print("Dataset Shape:", df.shape)

# ==========================================
# FEATURES (WITHOUT 3 FEATURES)
# ==========================================

X = df[
    [
        "Financial_Loss",
        "Operational_Disruption",
        "Reputation_Damage_Score",
        "Incident_Response_Time",
        "Recovery_Time",
        "Cybersecurity_Budget",
        "Technical_Score",
        "Human_Score",
        "Org_Score"
    ]
]

# ==========================================
# TARGET
# ==========================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df["Risk_Level"]
)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# DECISION TREE
# ==========================================

dt_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\n==============================")
print("DECISION TREE")
print("==============================")

print(
    "Accuracy:",
    round(
        accuracy_score(y_test, dt_pred) * 100,
        2
    ),
    "%"
)

print(
    classification_report(
        y_test,
        dt_pred,
        target_names=label_encoder.classes_
    )
)

# DT CONFUSION MATRIX

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    dt_pred,
    cmap="Blues"
)

disp.figure_.savefig(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Outputs/Confusion_Matrices/E2_Original_Reduced_DT.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# DT FEATURE IMPORTANCE

dt_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dt_model.feature_importances_
})

dt_importance = dt_importance.sort_values(
    by="Importance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(10,6))

ax.barh(
    dt_importance["Feature"],
    dt_importance["Importance"]
)

ax.set_title(
    "E2 Original Reduced DT Feature Importance"
)

fig.tight_layout()

fig.savefig(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Outputs/Feature_Importance/E2_Original_Reduced_DT_Feature_Importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# RANDOM FOREST
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

print(
    "Accuracy:",
    round(
        accuracy_score(y_test, rf_pred) * 100,
        2
    ),
    "%"
)

print(
    classification_report(
        y_test,
        rf_pred,
        target_names=label_encoder.classes_
    )
)

# RF CONFUSION MATRIX

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_pred,
    cmap="Blues"
)

disp.figure_.savefig(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Outputs/Confusion_Matrices/E2_Original_Reduced_RF.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# RF FEATURE IMPORTANCE

rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(10,6))

ax.barh(
    rf_importance["Feature"],
    rf_importance["Importance"]
)

ax.set_title(
    "E2 Original Reduced RF Feature Importance"
)

fig.tight_layout()

fig.savefig(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Outputs/Feature_Importance/E2_Original_Reduced_RF_Feature_Importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# XGBOOST
# ==========================================

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss"
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_pred = xgb_model.predict(
    X_test
)

print("\n==============================")
print("XGBOOST")
print("==============================")

print(
    "Accuracy:",
    round(
        accuracy_score(y_test, xgb_pred) * 100,
        2
    ),
    "%"
)

print(
    classification_report(
        y_test,
        xgb_pred,
        target_names=label_encoder.classes_
    )
)

# XGB CONFUSION MATRIX

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    xgb_pred,
    cmap="Blues"
)

disp.figure_.savefig(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Outputs/Confusion_Matrices/E2_Original_Reduced_XGB.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# XGB FEATURE IMPORTANCE

xgb_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
})

xgb_importance = xgb_importance.sort_values(
    by="Importance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(10,6))

ax.barh(
    xgb_importance["Feature"],
    xgb_importance["Importance"]
)

ax.set_title(
    "E2 Original Reduced XGB Feature Importance"
)

fig.tight_layout()

fig.savefig(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Outputs/Feature_Importance/E2_Original_Reduced_XGB_Feature_Importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


results = pd.DataFrame({
    "Experiment": ["E2_Original_Reduced"],
    "DT": [accuracy_score(y_test, dt_pred) * 100],
    "RF": [accuracy_score(y_test, rf_pred) * 100],
    "XGB": [accuracy_score(y_test, xgb_pred) * 100]
})

# Export to CSV

results = pd.DataFrame({
    "Experiment": ["E2_Original_Reduced"],
    "DT": [accuracy_score(y_test, dt_pred) * 100],
    "RF": [accuracy_score(y_test, rf_pred) * 100],
    "XGB": [accuracy_score(y_test, xgb_pred) * 100]
})

print(results)

results.to_csv(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Outputs/Tables/All_Results.csv",
    mode="a",
    header=False,
    index=False
)

print("E2 Results Saved Successfully")