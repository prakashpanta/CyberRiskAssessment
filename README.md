# Cyber Security Risk Assessment Framework for SMEs

## Overview

This project presents a machine learning-based cyber risk assessment framework for Small and Medium Enterprises (SMEs). The framework integrates technical, human and organisational cybersecurity factors into a unified cyber risk assessment model.

A synthetic cyber risk scoring methodology was developed to quantify cybersecurity risk and classify organisations into three risk categories:

- Low Risk
- Medium Risk
- High Risk

The framework was evaluated using multiple machine learning algorithms, including Decision Tree, Random Forest, XGBoost and Stacking Ensemble models.

The Stacking Ensemble combines Decision Tree, Random Forest and XGBoost classifiers through a Logistic Regression meta-learner to improve cyber risk prediction performance.

In addition, a Streamlit-based dashboard was developed to provide an interactive interface for cyber risk assessment, visualisation and recommendation generation.

---

## Research Objectives

The research aims to:

- Develop a socio-technical cyber risk assessment framework for SMEs.
- Integrate technical, human and organisational cybersecurity factors into a unified assessment model.
- Apply machine learning techniques for cyber risk classification.
- Evaluate and compare multiple machine learning models.
- Develop a practical dashboard for cyber risk visualisation and decision support.

---

## Methodology

The project follows the following workflow:

1. Literature Review and Research Gap Identification
2. Dataset Preparation
3. Feature Selection
4. Data Preprocessing
5. Synthetic Cyber Risk Score Development
6. Cyber Risk Classification
7. Machine Learning Model Development
8. Comparative Model Evaluation
9. Stacking Ensemble Development
10. Dashboard Development and Implementation

---

## Machine Learning Models

The following machine learning models were evaluated:

- Decision Tree
- Random Forest
- XGBoost
- Stacking Ensemble

### Best Model Performance

- Decision Tree: 76.44%
- Random Forest: 92.25%
- XGBoost: 92.19%
- Stacking Ensemble: 95.75%

---

## Cyber Risk Dimensions

The framework incorporates three key cybersecurity dimensions:

### Technical Factors

- Multi-Factor Authentication (MFA)
- Backup Availability
- Incident Response Capability
- Technical Security Controls

### Human Factors

- Employee Training
- Security Awareness
- Human Resilience Score

### Organisational Factors

- Cybersecurity Budget
- Organisational Readiness
- Governance and Policy Measures

These factors are integrated within the cyber risk scoring framework to produce Low, Medium and High cyber risk classifications.

---

## Project Structure

```text
Cyber_Security_Risk_Assessment/
│
├── Dashboard/
│   └── application.py
│
├── Data/
│   ├── Raw/
│   └── Processed/
│
├── Models/
│
├── Notebooks/
│
├── Outputs/
│   ├── Figures/
│   └── Tables/
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

Launch the Streamlit dashboard using:

```bash
streamlit run Dashboard/application.py
```

---

## Research Contributions

This research contributes to SME cybersecurity by:

- Developing a socio-technical cyber risk assessment framework.
- Introducing a synthetic cyber risk scoring methodology.
- Demonstrating the effectiveness of machine learning for cyber risk prediction.
- Evaluating Decision Tree, Random Forest, XGBoost and Stacking Ensemble models.
- Developing a dashboard-based decision-support tool for SMEs.

---

## Author

**Prakash Panta**  
Master of ICT Research  
Melbourne Institute of Technology (MIT)

---

## Repository

This repository contains:

- Source code
- Datasets
- Trained machine learning models
- Experimental notebooks
- Dashboard implementation
- Results and outputs used in the research
