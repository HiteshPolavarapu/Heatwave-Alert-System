# Summer Heat Waves Mobile Alert System - Vijayawada

## Project Overview

This repository contains the data analysis and prototype machine learning model for a **Summer Heat Waves Mobile Alert System** designed for the city of Vijayawada, Andhra Pradesh. With the increasing risks posed by extreme heat events due to climate change, this project aims to create a system that can predict the likelihood of a heatwave and provide timely, actionable alerts to the public.

This project covers the complete data science lifecycle, from initial data exploration to the development of a deployable predictive model and the architectural design for a full-scale mobile alert platform.

---

##  Methodology & Project Phases

The project was executed in three main phases:

1.  **Data Analysis & Exploration:** A historical weather dataset for Vijayawada was acquired, cleaned, and analyzed. The key insight from this phase was the critical impact of humidity, making the **"Feels Like" temperature** a more important metric for public health than the actual temperature.

2.  **Predictive Model Development:** Using the cleaned data, a machine learning model was developed to predict the daily `alert_level` based on historical weather patterns. The alert levels are defined as:
    * **Normal:** "Feels Like" temperature below 40°C.
    * **Heatwave (Orange Alert):** "Feels Like" temperature between 40°C and 44.9°C.
    * **Severe Heatwave (Red Alert):** "Feels Like" temperature is 45°C or higher.

3.  **System Architecture Design:** A high-level blueprint was designed for a production system to integrate the model with mobile platforms for widespread dissemination of alerts via Push Notifications and SMS.

---

## Dataset

* **Source:** The dataset was sourced from [Visual Crossing](https://www.visualcrossing.com/), providing daily historical weather data for Vijayawada from September 2024 to September 2025.
* **Filename:** `Vijayawada24_25.csv`
* **Key Features Used:** `datetime`, `tempmax`, `tempmin`, `humidity`, `feelslikemax`, `solarradiation`.

---

## Model Performance (Prototype V1)

A **Random Forest Classifier** was trained to predict the daily alert level. The model was evaluated on a test set (20% of the data) that it had never seen before.

* **Overall Accuracy:** **75.34%**
* **Key Finding:** The model performs well in predicting "Normal" and "Heatwave" days. However, the analysis revealed a key challenge with **class imbalance**; the model struggled to predict the rare but most critical "Severe Heatwave" events, achieving a recall of only 25% for that category.
* **Conclusion:** The prototype successfully proves the viability of the concept. Future work must focus on techniques to improve the model's performance on rare, high-risk events, such as using `class_weight` balancing or gathering more years of data.

---

## Proposed System Architecture

The end-to-end system is designed to be fully automated and cloud-based:

1.  **Backend:** A Python script deployed on a cloud server (e.g., AWS Lambda) runs daily. It fetches the latest weather data, feeds it to our trained model, and gets a prediction.
2.  **Dissemination:** If a "Heatwave" or "Severe Heatwave" is predicted, the backend triggers alerts through:
    * **Firebase Cloud Messaging (FCM)** for mobile app push notifications.
    * An **SMS Gateway** (e.g., Twilio) for text message alerts.
3.  **Frontend:** A simple, lightweight mobile app for Android and iOS that displays the current alert status, provides safety tips, and shows a map of public cooling centers.

---

## Technologies Used

* **Data Analysis:** Python, Pandas, Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, LightGBM, Joblib
* **Development Environment:** Visual Studio Code, Conda
* **Collaboration:** Git & GitHub

---

## How to Use This Repository

To replicate this analysis, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/HiteshPolavarapu/Heatwave-Alert-System.git](https://github.com/HiteshPolavarapu/Heatwave-Alert-System.git)
    cd Heatwave-Alert-System
    ```
2.  **Set up the Conda environment:**
    ```bash
    conda create --name heatwave_env python=3.11
    conda activate heatwave_env
    conda install pandas matplotlib seaborn scikit-learn jupyter joblib lightgbm -c conda-forge
    ```
3.  **Launch the notebook:**
    Open the `initial_analysis.ipynb` file in VS Code or a Jupyter Notebook environment to view the complete data analysis and model training process.
