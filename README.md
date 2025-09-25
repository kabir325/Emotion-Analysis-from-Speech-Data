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
````

`requirements.txt` should include:

```
numpy  
pandas  
matplotlib  
seaborn  
scikit-learn  
librosa  
scipy  
```

### Running the Project

1. Clone this repo:

   ```bash
   git clone https://github.com/your-username/emotion-analysis-speech.git
   cd emotion-analysis-speech
   ```
2. Place your audio dataset under `data/` directory.
3. Run the script:

   ```bash
   python emotion_analysis.py
   ```

---

## 📊 Results

* Extracted **50+ features per audio sample**.
* PCA reduced features from high-dimensional space to **~30 components explaining >95% variance**.
* K-Means achieved the best clustering performance with:

  * **Silhouette Score ~0.55**
  * **Calinski-Harabasz Score ~1100**
* Identified **5 emotion clusters** mapping approximately to: Neutral, Happy/Excited, Sad/Calm, Angry/Frustrated, Surprised/Energetic.

---

## 🔮 Future Improvements

* Use **deep learning models** (CNNs, Autoencoders, LSTMs) for feature learning.
* Explore **semi-supervised learning** with partial labels.
* Implement **real-time streaming audio emotion detection**.
* Extend to multimodal emotion recognition (speech + facial expressions).

---

## 📌 Project Structure

```
.
├── data/                     # RAVDESS dataset (not included)
├── emotion_analysis.py       # Main script
├── requirements.txt          # Dependencies
├── results/                  # Plots and CSV results
└── README.md                 # Project documentation
```

---

## ✨ Author

**Kabir Sahu**

* B.Tech (CSE), PES University
* [LinkedIn](https://www.linkedin.com/in/kabir-sahu/) | [GitHub](https://github.com/kabir325)

```


👉 I can also generate the `requirements.txt` file for you directly from the imports in your code so you don’t need to write it manually. Do you want me to create that as well?
```

