# ==========================================
# STACKING ENSEMBLE
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.inspection import permutation_importance

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Data/Processed/risk_scored_dataset_augmented.csv"
)

print("\nDataset Loaded Successfully")
print(df.shape)

# ==========================================
# FEATURES
# ==========================================
# NO RISK_SCORE = Avoid Data Leakage

features = [
    "Financial_Loss",
    "Operational_Disruption",
    "Reputation_Damage_Score",
    "Incident_Response_Time",
    "Recovery_Time",
    "Cybersecurity_Budget",
    "Employee_Training",
    "Use_of_MFA",
    "Data_Backup_Availability",
    "Technical_Score",
    "Human_Score",
    "Org_Score"
]

X = df[features]

# ==========================================
# TARGET
# ==========================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df["Risk_Level"]
)

print("\nClasses:")
print(label_encoder.classes_)

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

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# ==========================================
# BASE MODELS
# ==========================================

dt = DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    eval_metric="mlogloss"
)

# ==========================================
# STACKING MODEL
# ==========================================

estimators = [
    ("dt", dt),
    ("rf", rf),
    ("xgb", xgb)
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

# ==========================================
# TRAIN
# ==========================================

print("\nTraining Stacking Ensemble...")

stack_model.fit(
    X_train,
    y_train
)

print("Training Complete")

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = stack_model.predict(
    X_test
)

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("STACKING ENSEMBLE")
print("==============================")

print(
    "Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues"
)

disp.figure_.savefig(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Outputs/Confusion_Matrices/E4_Augmented_Stacking.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Confusion Matrix Saved")

# ==========================================
# FEATURE IMPORTANCE
# PERMUTATION IMPORTANCE
# ==========================================

print("\nCalculating Feature Importance...")

importance = permutation_importance(
    stack_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)

stack_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance.importances_mean
})

stack_importance = stack_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(stack_importance)

# ==========================================
# SAVE FEATURE IMPORTANCE CSV
# ==========================================

stack_importance.to_csv(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Outputs/Feature_Importance/E4_Augmented_Stacking_Feature_Importance.csv",
    index=False
)

# ==========================================
# SAVE FEATURE IMPORTANCE FIGURE
# ==========================================

fig, ax = plt.subplots(
    figsize=(10,6)
)

ax.barh(
    stack_importance["Feature"],
    stack_importance["Importance"]
)

ax.set_title(
    "E4 Augmented Stacking Feature Importance"
)

ax.invert_yaxis()

fig.tight_layout()

fig.savefig(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Outputs/Feature_Importance/E4_Augmented_Stacking_Feature_Importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Feature Importance Saved")

# ==========================================
# SAVE CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

report_df.to_csv(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Outputs/Tables/E4_Augmented_Stacking_Report.csv"
)

print("Classification Report Saved")

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    stack_model,
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Models/Augmented_stacking_model.pkl"
)

joblib.dump(
    label_encoder,
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Models/AugStacking_label_encoder.pkl"
    
)

print("Model Saved")

# ==========================================
# SAVE RESULTS
# ==========================================

results = pd.DataFrame({
    "Experiment": ["E4_Augmented_Stacking"],
    "Stacking": [
        accuracy * 100
    ]
})

print(results)

results.to_csv(
    "/Users/prakashpanta/Desktop/Cyber_Security_Risk_Assessment/Outputs/Tables/AugStacking_All_Results.csv",
    mode="a",
    header=False,
    index=False
)

print("Results Saved Successfully")