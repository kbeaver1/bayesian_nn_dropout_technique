![](docs/uofa.png)
# Bayesian Neural Network & Monte Carlo Dropout for Loan Approval Prediction

## Overview

This project applies Bayesian Neural Networks (BNNs) and Monte Carlo Dropout (MC Dropout) to predict loan approval decisions using the Bank Loan Approval dataset from Kaggle. The goals are:
1. Evaluate the effectiveness of BNNs for loan approval prediction.
2. Assess how MC Dropout prevents overfitting and improves generalization.

By fine-tuning dropout rates (30% to 60%), we achieved 90% accuracy and a loan approval rate of ~8.5%, improving model performance.

## Dataset

The Bank Loan Approval dataset contains 5,000 observations and 12 features, including:
- Age, Years of Work Experience, Income, Credit Score, etc.
- **Target**: Binary outcome for loan approval (1 = approved, 0 = denied).

## Methodology

### Model Architecture:
- **4 hidden layers**: 128, 64, 32, 16 units.
- **Output layer**: Sigmoid activation for binary classification.

### Training:
- **Training/Test Split**: 80%/20%
- **Epochs**: 30
- **Batch Size**: 5
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam

### Regularization:
- Applied Monte Carlo Dropout with dropout rates from 30% to 60% to prevent overfitting.

## Results

- **Without MC Dropout**: Overfitting observed with high training accuracy (~95%) and low validation loss (~7.83%).
- **With MC Dropout**: Training loss increased to ~20%, and accuracy remained near 90%, with loan approval rate reduced to 8.5%, more aligned with real-world expectations.

## Conclusion

BNNs and MC Dropout effectively prevent overfitting and improve loan approval prediction. The model can be adapted for other classification tasks with similar datasets.

## Future Work

- Expand to fraud detection, customer churn, and healthcare diagnostics.
- Scale with distributed training for larger datasets.
- Explore model explainability for transparent decision-making.

## Authors

- **Gubbala Durga Prasanth**: Machine Learning, AI
- **Kendall Beaver**: Data Science, Machine Learning
- **V.S. Murali Krishna Chittlu**: Machine Learning, AI
- **Dr. Kunal Arekar**: Bayesian Modeling, Deep Learning

For questions, contact any of above authors or Dr. Kunal Arekar at [kunalarekar@arizona.edu](mailto:kunalarekar@arizona.edu).

