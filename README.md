# DecepTech: Multimodal Deception Detection

**DecepTech** is an end-to-end pipeline for automatically detecting deception by fusing audio and facial cues. It combines pretrained speech embeddings, handcrafted audio features, and normalized facial‐landmark vectors into an ensemble of machine-learning classifiers.

---

## 🚀 Features

- **Audio Pipeline**  
  - Wav2Vec2 → 768-dim embeddings  
  - MFCC extraction (13-dim)  
  - Random Forest & Logistic Regression benchmarks  
- **Vision Pipeline**  
  - dlib 68-point facial landmarks → 136-dim vectors  
  - Random Forest & XGBoost classifiers  
- **Multimodal Ensemble**  
  - Stacked XGBoost on face-score + MFCC features  
  - Near-perfect test performance  
- **UI for Batch Processing**  
  - Lightweight Tkinter interface (`pipelineUI.py`)  
  - Real-time logs, progress bars, and result export  
- **Jupyter Notebooks** for exploratory analysis and visualization  

---

## 📁 Repository Structure

```text
DecepTech/
├── notebooks/
│   ├── TrainModel.ipynb
│   ├── VideoAndAudioModel.ipynb
│   ├── VideoExtract.ipynb
│   └── VideoModelScript.ipynb
├── scripts/
│   ├── audio_pipeline.py
│   ├── video_pipeline.py
│   ├── train_models.py
│   └── pipelineUI.py
├── data/
│   ├── audio_clips/            # raw .wav files
│   ├── CroppedClips/           # raw video clips
│   ├── transcripts/            # optional transcripts
│   └── csv/                    # metadata CSVs
├── models/
│   ├── deceptech_model.pkl
│   ├── deceptech_logreg_model.pkl
│   ├── rf_face_model.pkl
│   ├── xgb_face_model.json
│   └── stacked_model.pkl
├── requirements.txt
└── README.md
