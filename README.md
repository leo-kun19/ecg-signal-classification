# ECG Signal Classification

Pipeline klasifikasi sinyal **elektrokardiogram (ECG)** menggunakan dataset **PTB-XL**, dengan pendekatan **hybrid**: model **LSTM** untuk sinyal ECG mentah dan model berbasis **BERT/RoBERTa** untuk laporan teks diagnosis. Repo ini berisi **kode saja** — data dan model terlatih tidak disertakan (lihat `.gitignore`).

## 🎯 Pendekatan

- **LSTM (TensorFlow/Keras)** — mempelajari pola dari sinyal ECG multi-lead untuk memprediksi kelas diagnostik (SCP codes).
- **BERT/RoBERTa (PyTorch + Transformers)** — mengklasifikasikan laporan teks klinis yang menyertai tiap rekaman.
- Kedua modalitas (sinyal + teks) dieksplorasi untuk membandingkan dan menggabungkan performa.

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow/Keras (LSTM), PyTorch + HuggingFace Transformers (BERT/RoBERTa)
- **Data ECG:** wfdb, scipy, numpy, pandas
- **Evaluasi & Visualisasi:** scikit-learn, matplotlib, seaborn, plotly
- **Eksplorasi:** Jupyter Notebook

## 📁 Struktur

```
.
├── run_pipeline.py            # Entry point: jalankan pipeline end-to-end
├── requirements.txt
├── src/
│   ├── download_data.py       # Unduh/siapkan dataset PTB-XL
│   ├── analyze_dataset.py     # Analisis & statistik dataset
│   ├── preprocessing.py       # Preprocessing sinyal ECG
│   ├── lstm_model.py          # Model LSTM untuk sinyal
│   ├── bert_model.py          # Model BERT/RoBERTa untuk laporan teks
│   ├── train_pipeline.py      # Orkestrasi training
│   └── utils.py               # Loader PTB-XL & helper
├── notebooks/                 # Eksplorasi data & eksperimen model
└── results/                   # Hasil analisis & visualisasi dataset
```

## 🚀 Menjalankan

```bash
# 1. Install dependensi (disarankan pakai virtual environment)
pip install -r requirements.txt

# 2. Siapkan dataset PTB-XL
python src/download_data.py

# 3. Jalankan pipeline (preprocessing + training + evaluasi)
python run_pipeline.py
```

> Dataset PTB-XL berukuran besar dan tidak disertakan di repo. Jalankan `src/download_data.py` atau unduh manual dari [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/) lalu tempatkan di folder `data/`.

## 📊 Dataset

[PTB-XL](https://physionet.org/content/ptb-xl/) — dataset ECG 12-lead berlabel klinis berskala besar (PhysioNet). Label diagnostik menggunakan SCP codes, disertai laporan teks tiap rekaman.

## 📄 Lisensi

Proyek ini dibuat untuk keperluan pembelajaran. Penggunaan dataset PTB-XL mengikuti lisensi PhysioNet.
