# 🎵 Emotion Analysis from Speech Data – Unsupervised Learning  
*Capstone Project 2: Building and Comparing ML/DL Models*  

## 📌 Overview  
This project focuses on **emotion recognition from speech audio** using the **RAVDESS dataset**.  
The objective was to explore **unsupervised learning approaches** for clustering emotional states in speech without relying on explicit labels.  

Key highlights:  
- Extracted **comprehensive audio features** (MFCCs, spectral features, pitch, energy, tempo, chroma, mel spectrograms).  
- Applied multiple **clustering algorithms**: K-Means, DBSCAN, Agglomerative Clustering, Gaussian Mixture Models, Spectral Clustering, and Ensemble Clustering.  
- Used **dimensionality reduction techniques** (PCA, t-SNE, TruncatedSVD) for visualization and noise reduction.  
- Evaluated clustering quality with **Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Score**.  
- Built a **real-time emotion detection pipeline** for batch or single audio prediction.  

---

## 📂 Dataset  
- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**  
- Contains speech recordings across 8 emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.  
- Each file includes actor ID, emotion label, and modality in its filename.  
- Dataset available on [Kaggle](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) or directly from [RAVDESS official site](https://zenodo.org/record/1188976).  

---

## ⚙️ Project Workflow  

1. **Data Loading & Exploration**  
   - Audio preprocessing using **Librosa**.  
   - Extraction of basic info: sample rate, duration, waveform.  

2. **Feature Extraction**  
   - Spectral centroid, rolloff, bandwidth  
   - Zero crossing rate (ZCR)  
   - MFCCs (13 coefficients)  
   - Chroma features, mel spectrogram  
   - Pitch, RMS energy, tempo  
   - Spectral contrast  

3. **Data Preprocessing**  
   - Variance thresholding  
   - Feature scaling (**StandardScaler**)  
   - Dimensionality reduction (PCA, t-SNE)  

4. **Clustering & Evaluation**  
   - Algorithms: K-Means, Agglomerative, DBSCAN, Gaussian Mixture, Spectral Clustering  
   - Evaluation metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin  
   - Ensemble clustering for consensus  

5. **Visualization**  
   - PCA scatter plots  
   - t-SNE plots of clusters  
   - Heatmaps and feature correlations  

6. **Pipeline for Real-time Emotion Detection**  
   - Feature extraction → Scaling → PCA → Cluster prediction → Emotion mapping  

---

## 🚀 Installation & Usage  

### Prerequisites  
- Python 3.8+  
- Install dependencies:  

```bash
pip install -r requirements.txt
