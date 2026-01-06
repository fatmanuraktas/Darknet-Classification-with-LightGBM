# 🛡️ AI-Powered Real-Time Darknet Traffic Detection

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Language](https://img.shields.io/badge/C%23-ML.NET-purple)
![Language](https://img.shields.io/badge/Python-Pandas%2FScikit--Learn-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-98.05%25-success)

This project is a high-performance, hybrid AI system designed to detect and classify **Darknet (Tor, VPN)** traffic without relying on Deep Packet Inspection (DPI) or IP-based blocking. It utilizes **behavioral flow analysis** to identify encrypted traffic patterns with **98% accuracy**.

## 🚀 Key Features
* **Privacy-Preserving:** Detects threats using only traffic flow statistics (Time, Packet Size, Jitter) without decrypting payloads.
* **Hybrid Architecture:** * **Python:** Advanced data preprocessing, feature engineering, and statistical analysis.
    * **C# (.NET 8):** Production-grade inference engine using **ML.NET** and **LightGBM** for ultra-fast, real-time detection.
* **High Precision:** Achieved **97.77% F1-Score**, minimizing false alarms in identifying benign VPNs vs. malicious Darknet channels.
* **Dataset Used:** SafeSurf Darknet [https://data.mendeley.com/datasets/kcrnj6z4rm/2]

## 📊 Performance Results
The model was trained and tested on a dataset of +250k network flows.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **98.05%** | Overall correctness of the model. |
| **F1 Score** | **97.77%** | Harmonic mean of Precision and Recall (Critical for security). |
| **AUC** | **99.2%** | Excellent separation capability between classes. |

## 🛠️ Tech Stack
* **Data Analysis:** Python, Pandas, NumPy, Scikit-learn (Random Forest for Feature Selection).
* **Machine Learning:** Microsoft ML.NET, LightGBM (Gradient Boosting Machine).
* **Runtime:** .NET 8.0 Console Application (Linux/Windows cross-platform).

## ⚙️ How It Works
1.  **Ingestion:** Network flow data (CICFlowMeter format) is processed.
2.  **Feature Selection:** Irrelevant data (IPs, Ports) and noise are removed using a Random Forest importance threshold.
3.  **Inference:** The C# engine loads the trained LightGBM model and classifies the flow as `Normal` or `Darknet` in milliseconds.

## 📂 Project Structure
```bash
├── 📁 Data_Preprocessing    # Python scripts for cleaning & feature selection
├── 📁 DarknetAI             # C# .NET Console Application (Inference Engine)
├── 📄 DarknetModel.zip      # Trained ML Model (Serialized)
└── 📄 README.md             # Documentation
