# 🌌 AIR-E: Aadhaar Integrity & Risk Engine
### *National Risk Overview & Real-Time Anomaly Intelligence Command Center*

---

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-111111?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

**AIR-E (Aadhaar Integrity & Risk Engine)** is a privacy-safe, high-performance administrative intelligence layer designed to detect structural anomalies and predict risk patterns using anonymized demographic and enrollment telemetry. Inspired by modern space-tech command centers, it enables proactive administrative reviews without increasing surveillance or exposing personally identifiable information (PII).

> [!NOTE]  
> **Hackathon Concept Project:** This application is a fully functional simulation built for hackathon demonstration. All datasets, risk indices, and geographic profiles are synthetically generated. It is not affiliated with nor based on actual production systems or internal data of the **Unique Identification Authority of India (UIDAI)**.

---

## 🚀 Key Features

*   **🌌 High-Fidelity Space-Tech UI**: A premium dark-themed administrative dashboard complete with glassmorphic cards, dynamic linear slider gradients, retro scanline styling, and pulsing state indicators.
*   **🌐 3D Orbital Globe Reconnaissance**: An interactive, custom-shaded **Three.js 3D Globe** rotating in real-time, plotting geographical anomalies and risk clusters on its outer shell.
*   **🤖 Unsupervised Anomaly Detection**: Integrates **Isolation Forest** algorithms to scan demographic metrics (biometric failures, update velocities, and document rejections) for anomalous behavior.
*   **📈 Supervised Risk Forecasting**: Employs high-performance **XGBoost Regressor** pipelines to assign early-stage risk scores across registrar operations.
*   **⚖️ Dynamic Policy Simulator**: An interactive "what-if" planning workbench. Adjust administrative rigors and immediately view projected leakage prevention (in Crores) balanced against citizen exclusion risks.
*   **🔍 Explainable AI (XAI) Dashboard**: Powered by simulated **SHAP waterfall attributions** and active Pearson correlation matrices to explain precisely *why* a particular region's risk index has spiked.
*   **🔒 DPDP & Privacy Aligned**: Z-Score engines run purely on anonymized and aggregated counts. Zero personal data (PII) is exposed or processed, ensuring full alignment with the **Digital Personal Data Protection (DPDP) Act**.

---

## 🛠️ 3-Tier Architecture

To maintain maximum code cleanliness, clean modular design, and execution speeds, the codebase is consolidated into a robust, recruiter-friendly **3-script engine layout**:

```
AIR-E-main/
├── data_engine.py      # Tier 1: Vectorized synthetic data generator (1M+ rows)
├── ml_engine.py        # Tier 2: Machine Learning pipeline (Isolation Forest + XGBoost)
├── app.py              # Tier 3: Premium space-tech Streamlit command dashboard
├── style.css           # Styling: Deep-space global CSS override configurations
├── aire_logo.png       # Brand asset logo
├── requirements.txt    # Python dependencies config
└── LICENSE             # MIT License
```

---

## ⚡ Quick Start

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/yocybroarnv/AIR-E.git
cd AIR-E-main

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Telemetry
To simulate standard administrative workloads (1 million rows of registrar datasets), run the **Data Engine**:
```bash
python data_engine.py
```
*Outputs `raw_data.parquet` (compressed columnar storage for 100x faster execution than standard CSV).*

### 3. Run the Machine Learning Pipeline
To execute the Isolation Forest and XGBoost model training and generate features:
```bash
python ml_engine.py
```
*Outputs `processed_data.parquet` containing calculated risk levels, anomaly scores, and projected indicators.*

### 4. Launch the Command Center
Launch the Streamlit web application:
```bash
streamlit run app.py
```

---

## 📊 Deep-Dive: Machine Learning Pipeline

### Unsupervised Isolation Forest (Anomaly Detection)
*   **Objective**: Detect structural outliers in registration logs (e.g., massive spikes in updates, unusual document rejection velocities).
*   **Features Used**: `enrollments`, `biometric_failure_rate`, `document_rejection_rate`.
*   **Contamination Rate**: Set at a defensive `0.05` to target extreme anomalies.

### Supervised XGBoost (Risk Forecasting)
*   **Objective**: Score administrative units on a scale of `0.00` to `1.00` to forecast the likelihood of operator credentials compromise.
*   **Training Target**: Synthesized via a weighted composite index of biometric failures, document rejections, and structural anomalies.
*   **Architecture**: 50 gradient-boosted decision trees with a max depth of 4 for absolute latency containment in real-time Streamlit charts.

---

## 👥 Developer Credits

Developed with passion by **Arnav Raj (Cybroarnv)**:
*   **GitHub**: [@yocybroarnv](https://github.com/yocybroarnv)
*   **LinkedIn**: [Arnav Raj](https://www.linkedin.com/in/arnav-raj-professional)
*   **Organization**: UIDAI Risk Intelligence Division (Hackathon Concept)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
