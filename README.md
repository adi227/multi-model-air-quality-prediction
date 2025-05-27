#  Multi-Model CO₂ Emission Prediction Using Agro-Food and Demographic Data

This project presents a comprehensive analysis and modeling framework to predict national CO₂ emissions using agro-food activities, demographic indicators, and environmental covariates across 197 countries from 1990–2020. Leveraging a rich country-year panel dataset, the work explores a spectrum of predictive techniques including machine learning and deep learning models to extract insights, reduce multicollinearity, and enable policy-oriented forecasts.

---

##  Objective

**Can total CO₂ emissions of a country be accurately predicted from its agro-food activities and population characteristics?**

---

##  Dataset

- **Source**: [Gigasheet Agro-Food CO₂ Emission Dataset](https://www.gigasheet.com/sample-data/agri-food-co2-emission-dataset---forecasting-ml)
- **Size**: 1,200 observations × 31–35 features
- **Features**:
  - Emission drivers (fertiliser use, food transport, manure, crop residue)
  - Demographic indicators (population counts, urban ratio)
  - Environmental (temperature, region)

---

##  Data Preparation & Cleaning

- Created a **Region** variable for cluster-aware imputation.
- Addressed missing values using:
  - **Linear Interpolation** (short-term gaps)
  - **KNN Imputation** (context-aware)
- Removed phantom zeros and zero-variance columns.
- Applied **StandardScaler** for normalization.

---

##  Exploratory Data Analysis & PCA

- Visualized emissions by year and region.
- Identified multicollinearity (r > 0.8) between major variables.
- Applied **Principal Component Analysis (PCA)**:
  - Reduced to 5–6 components (capturing ~80% variance)
  - Enabled clustering (K-Means) and dimensionality insights

---

##  Models Implemented

### 🔹 Machine Learning Models
- **Random Forest**: R² = 0.9958 | RMSE ≈ 13,419 kt
- **XGBoost**: R² = 0.9916 | RMSE ≈ 17,028 kt
- **Support Vector Machine (SVM)**: R² = 0.9928 | RMSE ≈ 18,563 kt
- **Decision Tree**: R² = 0.9643 | RMSE ≈ 31,082 kt

### 🔹 Deep Learning Models
- **Multilayer Perceptron (MLP)**: R² = 0.9894 | RMSE ≈ 23,505 kt
- **Autoencoder + MLP**: R² = 0.9634 | RMSE ≈ 23,000–24,000 kt
- **Deep Neural Network (DNN)**: R² = 0.9667 | RMSE ≈ 40,870 kt
- **LSTM with Embeddings**: R² = 0.9835 | RMSE ≈ 40,115 kt

---

## 📈 Model Comparison

| Model              | R² Score | RMSE (kt)  | Notes                            |
|-------------------|----------|------------|----------------------------------|
| Random Forest      | 0.9958   | 13,419     | Most accurate and interpretable |
| XGBoost            | 0.9916   | 17,028     | Fast and scalable                |
| SVM                | 0.9928   | 18,563     | Strong, scale-sensitive          |
| Decision Tree      | 0.9643   | 31,082     | Simple but overfits              |
| MLP                | 0.9894   | 23,505     | Best DL performance              |
| Autoencoder + MLP  | 0.9634   | ~24,000    | Interpretable latent space       |
| DNN                | 0.9667   | 40,870     | Deeper but higher error          |
| LSTM               | 0.9835   | 40,115     | Captures temporal dynamics       |

---

##  Tech Stack

- **Languages**: Python, R  
- **Python Libraries**: `scikit-learn`, `tensorflow.keras`, `pandas`, `matplotlib`, `seaborn`  
- **R Packages**: `dplyr`, `ggplot2`, `randomForest`  
- **Tools Used**: PCA, K-Means, KNN Imputation, Grid Search, Early Stopping, t-SNE

---

##  Conclusion

All models demonstrated high performance in predicting national CO₂ emissions, with **Random Forest** being the top performer in accuracy, robustness, and interpretability. Deep learning methods like **MLP** and **Autoencoder + MLP** offered valuable insights through latent representations, while **LSTM** showed potential for temporal modeling.

This multi-model study confirms that agro-food and demographic variables are reliable predictors of emissions and supports the use of ensemble and hybrid modeling strategies for environmental forecasting.

---


##  References

- [KNN Imputer – scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)  
- [PCA – scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)  
- [Random Forest – scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)  
- [TensorFlow MLP Tutorial](https://www.tensorflow.org/tutorials/keras/classification)

---

Note:-All source code and the dataset used for this project are organized and stored in the master branch of this repository.

