# ==========================================
# DT + RF + XGBoost
# WITHOUT Training, MFA and Backup
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report

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

# ==========================================
# RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print(
    rf_importance.sort_values(
        by="Importance",
        ascending=False
    )
)

# ==========================================
# XGBOOST FEATURE IMPORTANCE
# ==========================================

xgb_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
})

print("\nXGBOOST FEATURE IMPORTANCE")
print(
    xgb_importance.sort_values(
        by="Importance",
        ascending=False
    )
)