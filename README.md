## Credit Scoring Business Understanding

### Basel II Accord and Model Interpretability

The Basel II Capital Accord emphasizes accurate risk measurement, transparency, and strong internal controls in credit risk management. As a result, financial institutions are required to justify how credit decisions are made and ensure models are well-documented, auditable, and explainable. This makes interpretability a critical requirement in our project, as regulators and stakeholders must be able to understand the drivers behind risk predictions, not just the final score.

### Proxy Variable for Credit Risk

The dataset does not contain an explicit loan default label. Therefore, a proxy target variable must be engineered to represent credit risk. In this project, customer behavioral patterns derived from transaction data are used to approximate default risk. While this approach enables model development, it introduces business risks such as misclassification, where good customers may be labeled as high-risk or vice versa. These risks must be mitigated through careful validation, conservative decision thresholds, and continuous model monitoring.

### Trade-offs Between Simple and Complex Models

Simple models such as Logistic Regression combined with Weight of Evidence (WoE) are highly interpretable, stable, and widely accepted by regulators. However, they may have limited predictive power. More complex models such as Gradient Boosting can capture non-linear relationships and improve performance but are harder to interpret and explain. In a regulated financial context, this creates a trade-off between predictive accuracy and regulatory compliance. A balanced approach involves benchmarking complex models against simpler, interpretable ones and selecting a model that meets both performance and governance requirements.
