# Credit Card Fraud Detection with Scalable Machine Learning

Built during the **AI4ALL Ignite Accelerator**, this project establishes a scalable machine learning and deep learning fraud detection pipeline evaluated on over 590,000 real-world e-commerce transactions. By addressing extreme class imbalance through custom focal loss architectures and cost-sensitive boosting, the system achieves an **82.80% fraud recall** and a **0.9515 ROC-AUC**, protecting consumers while minimizing false-positive operational overhead.

* **Live Interactive Demo:** [Streamlit Dashboard](https://ai4all2a-hg8xmydpthagr86f3bubye.streamlit.app/)
* **Project Poster:** [Google Slides Poster](https://docs.google.com/presentation/d/1F_3GB4kGSd-pOoXvJ9uBWLfZNyNURko65H8SXACZynY/edit)
* **Final Presentation:** [Showcase Deck](https://docs.google.com/presentation/d/1Isa8MpG_EKpsnCZTqUqPVIQoa6r8-AW0bQSv9GLldBg/edit)

---

## Problem Statement & Motivation

Digital payment fraud inflicts billions of dollars in losses annually and disrupts financial integrity across global payment rails. Traditional fraud prevention relies on rigid rule-based systems that struggle against novel attack patterns or raw accuracy-optimized classifiers that overlook fraudulent minority classes. In production environments where fraudulent activity accounts for only ~3.5% of total volume, optimizing for accuracy results in high false-negative rates that directly harm consumers. This project automates proactive, real-time fraud defense, prioritizing minority-class recall and ethical accountability.

### Research Question
> *"Can we accurately predict whether a transaction is fraudulent based on anonymized transaction and identity signatures using supervised and deep learning classification models?"*

---

## Key Results

1. **High Fraud Detection Rate:** XGBoost delivered the strongest overall detection performance, capturing **82.80% of fraudulent transactions** with a **0.9515 ROC-AUC**.
2. **Imbalanced Deep Learning:** Engineered a Feed-Forward Deep Neural Network with **Binary Focal Cross-Entropy Loss**, achieving **78.44% recall** and a **0.9400 ROC-AUC** without synthetic oversampling.
3. **Data Quality & Dimensionality Optimization:** Resolved high data sparsity by shifting missingness thresholds to 50% (marking missing values explicitly as `'missing'`), streamlining the feature space from 394 to 339 predictive columns.
4. **Live Streamlit Deployment:** Deployed an interactive risk evaluation interface featuring single transaction simulation, synthetic parameter manipulation, and batch data inference.

---

## Methodologies & Technical Architecture

### 1. Data Cleaning & Feature Engineering
* **Missing Value Imputation:** Features with $>50\%$ missing values were retained and tagged with explicit `'missing'` categorical indicators to prevent skewing numerical distributions.
* **Velocity & Identity Signals:** Extracted transaction frequency spikes over short sliding time windows and engineered `email_domain_match` flags to isolate credential-stuffing and identity-theft patterns.

### 2. Multi-Model Benchmarking

| Model Architecture | ROC-AUC | Recall (Fraud) | Accuracy | F1-Score | Key Trade-Off Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Selected)** | **0.9515** | **82.80%** | 92.54% | **0.4373** | Highest recall; rapid inference on tabular data; tuned positive class weights. |
| **Deep Neural Network** | 0.9400 | 78.44% | 92.48% | 0.4220 | Multi-layer dense net with Binary Focal Loss; captures complex non-linear interactions. |
| **Random Forest** | 0.9120 | 63.00% | 97.10% | 0.5400 | High precision (0.54), but missed 37% of fraud cases due to conservative thresholding. |
| **Support Vector Machine** | — | — | — | — | Computationally infeasible; $O(n^2)$ complexity unscalable on 590k samples. |

* **Deep Neural Network Topology:**
  * **Layer 1:** 128 units (ReLU) + Dropout (0.3)
  * **Layer 2:** 64 units (ReLU) + Dropout (0.2)
  * **Output Layer:** 1 unit (Sigmoid)
  * **Loss & Optimization:** Binary Focal Cross-Entropy Loss, 25 Epochs, Batch Size 512, Exponential Learning Rate Decay.

---

## Visualizations & Live Demo

The pipeline is integrated with an interactive [Streamlit Application](https://ai4all2a-hg8xmydpthagr86f3bubye.streamlit.app/) allowing users to:
* Generate randomized transactions from the IEEE-CIS test split.
* Synthesize feature values ($V_1–V_{339}$, $C_1–C_{14}$, $D_1–D_{15}$) in real time to observe model decision boundaries.
* Stream synthetic batches to evaluate throughput and false-positive flags.

### Streamlit Dashboard Interface
![Streamlit Dashboard Preview](dashboard_preview.png)

*Our interactive dashboard allows users to synthesize feature values and stream batches to evaluate fraud detection in real-time.*
---

## Limitations, Ethics & Future Work

* **Algorithmic Accountability:** In fraud detection, missing a fraudulent transaction (false negative) inflicts direct financial harm on victims, whereas a false alarm (false positive) introduces manageable review friction. Our models deliberately optimize for high recall to minimize real-world harm.
* **Cold-Start Identity Sparsity:** Anonymized identity attributes exhibit high sparsity for first-time cardholders.
* **Future Directions:**
  * Implementing **Graph Neural Networks (GNNs)** to model shared device and IP relationship subgraphs.
  * Building low-latency streaming pipelines using Apache Kafka and Redis for sub-millisecond edge scoring.

---

## Data Sources

* **IEEE-CIS Fraud Detection Dataset:** [Kaggle Competition Data](https://www.kaggle.com/c/ieee-fraud-detection/data) (590,540 e-commerce transactions across linked `train_transaction.csv` and `train_identity.csv`).

## Technologies Used

* **Languages & Core Libraries:** Python, Pandas, NumPy, Scikit-learn
* **Machine Learning & Deep Learning:** XGBoost, TensorFlow / Keras (Binary Focal Loss)
* **Visualization & Deployment:** Seaborn, Matplotlib, Streamlit, Google Colab

---

## Authors

Completed by **Group 2A** during the **AI4ALL Ignite Summer Accelerator**:

* **Annika Bhatia** — *Rutgers University* (Computer Science & Data Science)
* **Maiyun Zhang** — *AI4ALL Fellow*
* **Patrick Selby** — *Grambling State University* (Cybersecurity & CIS)
* **Jolaoluwa Amodu** — *Fisk University* (Computer Science)
