#  IEEE-CIS Fraud Detection Machine Learning Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange) ![Optuna](https://img.shields.io/badge/Tuning-Optuna-blueviolet) ![MLflow](https://img.shields.io/badge/Tracking-MLflow-00b4d8) ![SMOTE](https://img.shields.io/badge/Sampling-SMOTE-green)

---

## Purpose of This Repository

### Primary Purpose UTS Machine Learning
Repositori ini dibuat sebagai bagian dari **Ujian Tengah Semester (UTS)** mata kuliah Machine Learning. Implementasi mencakup end-to-end pipeline klasifikasi mulai dari eksplorasi data, feature engineering, training model, hingga evaluasi performa.

### Secondary Purpose Deeper Learning
Di luar kebutuhan UTS, proyek ini juga menjadi media eksplorasi dan pembelajaran mendalam terkait berbagai aspek Machine Learning, di antaranya:

- **Hyperparameter Tuning** Memahami dampak setiap hyperparameter (`max_depth`, `learning_rate`, `n_estimators`, dll.) terhadap performa model menggunakan Optuna
- **Pengaruh Kompleksitas Model** Mengamati bagaimana kedalaman pohon (`max_depth=13`) mempengaruhi akurasi, waktu training, dan konsumsi RAM
- **Imbalanced Learning** Menangani dataset yang sangat tidak seimbang (96.5% vs 3.5%) menggunakan teknik SMOTE
- **Experiment Tracking** Memahami pentingnya mencatat setiap eksperimen secara sistematis dengan MLflow
- **Feature Engineering** Mereduksi ratusan fitur mentah menjadi representasi yang lebih ringkas dan informatif

---

## Project Overview

Dataset yang digunakan adalah **IEEE-CIS Fraud Detection** dataset transaksi keuangan nyata yang berisi ratusan fitur anonim dari sistem pembayaran Vesta.

| Informasi | Detail |
|-----------|--------|
| **Dataset** | `train_transaction.csv` + `test_transaction.csv` |
| **Jumlah Data Train** | 590.540 transaksi |
| **Fitur Awal** | 394 kolom |
| **Fitur Setelah Engineering** | 31 kolom |
| **Target** | `isFraud` (0 = Aman, 1 = Fraud) |
| **Rasio Fraud** | ~3.5% (sangat tidak seimbang) |

### Alur Pipeline

```
Raw Data (394 col)
    ↓
Feature Engineering
(Agregasi C/D/V, Drop kolom tidak relevan)
    ↓
Train-Test Split (90:10)
    ↓
Pipeline: Imputer → Scaler → Encoder → SMOTE → XGBoost
    ↓
Hyperparameter Tuning (Optuna, 50 trials)
    ↓
Final Model → Evaluasi → Prediksi Data Test
```

---

## Model & Evaluation Results

### Model yang Digunakan
**XGBoost Classifier** dikombinasikan dengan **SMOTE** untuk menangani class imbalance, dioptimasi menggunakan **Optuna** sebanyak 50 trials dan dilacak dengan **MLflow**.

### Hyperparameter Terbaik

| Parameter | Nilai |
|-----------|-------|
| `max_depth` | 13 |
| `learning_rate` | 0.1268 |
| `n_estimators` | 349 |
| `min_child_weight` | 4 |
| `smote_strategy` | 0.3598 |

### Perbandingan Performa

| Metrik | Baseline | Tuned (Final) | Δ |
|--------|:--------:|:-------------:|:-:|
| **Accuracy** | 97% | 98% | ▲ +1% |
| **AUC-ROC** | 0.8890 | **0.9626** | ▲ +7.36% |
| **Precision (Fraud)** | 0.57 | **0.86** | ▲ +0.29 |
| **Recall (Fraud)** | 0.46 | **0.63** | ▲ +0.17 |
| **F1-Score (Fraud)** | 0.51 | **0.73** | ▲ +0.22 |

### Confusion Matrix (Final Model)

```
                  Predicted
                  Aman    Fraud
Actual  Aman    [ 56725    220 ]
        Fraud   [   778   1331 ]
```

### Overfitting Check

| Split | AUC |
|-------|-----|
| Train | 0.9987 |
| Test  | 0.9626 |
| Gap   | 0.0360 (overfitting ringan) |

---

## How to Navigate This Repository

```
root
 ├── Classification.ipynb     # Notebook utama
 ├── hasil_prediksi.csv       # Output prediksi untuk data test
 ├── README.md                # You are here
 └── mlflow_eksperimen.db     # Database hasil tracking 50 trials Optuna
```

### Urutan Baca Notebook

Notebook `Classification.ipynb` disusun secara berurutan dari atas ke bawah:

| Bagian | Isi |
|--------|-----|
| **1** | Load data, eksplorasi awal, cek distribusi kelas |
| **2** | Feature engineering agregasi grup C, D, V & reduksi dimensi |
| **3** | Korelasi antar fitur, split data, definisi preprocessing pipeline |
| **4** | Training baseline model + evaluasi awal |
| **5** | Hyperparameter tuning dengan Optuna (50 trials) + MLflow tracking |
| **6** | Training final model dengan parameter terbaik + logging ke MLflow |
| **7** | Visualisasi Feature Importance (Top 20) |
| **8** | Pengecekan overfitting, Confusion Matrix, ROC Curve |

### Cara Menjalankan

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn xgboost imbalanced-learn optuna mlflow seaborn matplotlib

# 2. Pastikan file dataset tersedia di direktori yang sama
#    - train_transaction.csv
#    - test_transaction.csv

# 3. Jalankan notebook dari atas ke bawah secara berurutan
jupyter notebook Classification.ipynb
```

> Proses tuning Optuna memakan waktu cukup lama (~2–3 jam) karena menjalankan 50 trials dengan 3-fold cross validation. Lewati cell ini dan langsung gunakan parameter terbaik yang sudah tercatat jika hanya ingin menjalankan final model.

---

## Identification

| | |
|--|--|
| **Nama** | Muhamad Kevin |
| **Kelas** | ML-TK-46-GAB |
| **NIM** | 101032300243 |
| **Mata Kuliah** | Machine Learning |
