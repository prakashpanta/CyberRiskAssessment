# Cyber Security Risk Assessment for SMEs

This project presents a machine learning-based cyber risk assessment framework designed for Small and Medium Enterprises (SMEs).

The framework integrates technical, human, and organisational cybersecurity factors into a unified risk assessment model. A synthetic dataset is used to construct measurable risk scores and classify cybersecurity risk into three categories:

- Low Risk
- Medium Risk
- High Risk

The project evaluates multiple machine learning approaches, including:

- Decision Tree
- Random Forest
- XGBoost
- Stacking Ensemble

The Stacking Ensemble combines Decision Tree, Random Forest, and XGBoost using a Logistic Regression meta-learner.

The project also includes a prototype dashboard for interacting with the risk assessment framework and presenting the resulting risk classification in an accessible format.

## Research Focus

The project investigates how an integrated and interpretable machine learning approach can support cybersecurity risk assessment for SMEs, particularly where organisations may have limited cybersecurity resources and technical expertise.

## Methodology

The overall workflow consists of:

1. Literature review and research gap identification
2. Dataset preparation
3. Feature selection
4. Data preprocessing
5. Synthetic risk-score construction
6. Risk-level classification
7. Machine learning model development
8. Comparative model evaluation
9. Stacking ensemble development
10. Framework implementation and dashboard development

## Models

| Model | Purpose |
|---|---|
| Decision Tree | Primary interpretable classification model |
| Random Forest | Comparative ensemble model |
| XGBoost | Comparative boosting model |
| Stacking Ensemble | Combines DT, RF and XGBoost using Logistic Regression |

## Risk Dimensions

The framework considers three main cybersecurity dimensions:

- **Technical factors**
- **Human factors**
- **Organisational factors**

These factors are incorporated into the proposed risk-scoring framework to produce Low, Medium, and High risk classifications.

## Project Structure

```text
Cyber_Security_Risk_Assessment/
│
├── Data/
│   ├── Raw/
│   └── Processed/
│
├── Models/
│
├── Notebooks/
│
├── Dashboard/
│
├── Outputs/
│   ├── Figures/
│   └── Tables/
│
└── README.md
