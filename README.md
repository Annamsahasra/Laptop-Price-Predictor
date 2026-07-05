# Laptop Price Prediction Web Application

A premium, production-ready Flask web application that predicts retail laptop prices in Euros based on hardware specifications. The application uses a machine learning inference pipeline built directly from a trained gradient-boosted decision tree (`XGBoost`) model.

## 🌐 Live Demo

https://your-render-url.onrender.com

---

## 🛠️ Technologies Used

### Machine Learning
- **Python 3.11.11** - Core programming language.
- **XGBoost Regressor** - High-performance gradient boosting library.
- **Pandas & NumPy** - Data manipulation and vector math.
- **Scikit-Learn** - Model training, cross-validation, and splitting.
- **Joblib** - Binary serialization of the trained estimator and features.

### Web Development
- **Flask** - Lightweight WSGI web application framework.
- **Jinja2** - Templating engine for dynamic spec summaries and predictions.
- **Vanilla CSS (Grid & Flexbox)** - Modern custom dark-themed styling.
- **FontAwesome CDN** - SVG icon toolkit.

---

## 📁 Folder Structure

```
Laptop Price Prediction/
│
├── app.py
├── verify_model.py
├── LaptopPriceModel.pkl
├── FeatureNames.pkl
├── requirements.txt
├── Procfile
├── runtime.txt
├── .gitignore
├── README.md
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    ├── style.css
    └── images/
```

---

## ⚙️ Core Preprocessing Pipeline (Notebook Replication)

To guarantee that predictions match the Jupyter Notebook exactly, the backend performs the following steps on form submission:
1. **Physical Specs**: Extracts RAM (GB), weight (kg), and screen size (inches).
2. **Dynamic Screen Details**: Computes Pixel Density (PPI) using:
   $$\text{PPI} = \frac{\sqrt{X_{\text{res}}^2 + Y_{\text{res}}^2}}{\text{Inches}}$$
3. **Storage Computation**: Automatically calculates `Total_Storage` as the sum of SSD, HDD, Flash Storage, and Hybrid capacities.
4. **Exact Feature Vector Reconstruction**: 
   - Loads the exact column layout from `FeatureNames.pkl`.
   - Initializes a single-row DataFrame with all 43 columns set to `0`.
   - Maps inputs to their respective columns.
   - For categorical values, sanitizes strings (spaces replaced with `_`, special characters removed) and activates **only** the matched one-hot column by setting it to `1`.
   - Dropped categorical baselines (`Acer`, `2 in 1 Convertible`, `AMD`, `Linux`) automatically result in all-zero dummy flags.
   - Reorders columns to match the trained model's structure exactly before inference.

---

## 📈 Model Performance

These are the evaluation metrics for the final tuned machine learning model:

- **Model**: XGBoost Regressor
- **R² Score**: 0.8204
- **Cross Validation R²**: 0.8357
- **MAE**: 178.89 Euros
- **RMSE**: 302.05 Euros
- **MAPE**: 17.10%

---

## 💻 Local Installation

1. **Clone or Extract the Project Directory**:
   Ensure you are in the workspace root.

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Running Locally

To run the Flask application locally:

```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

To run using the production gunicorn container locally (Unix/Linux environments):
```bash
gunicorn app:app
```

---

## 📦 GitHub Push Instructions

Push the project files to your GitHub repository by running:

1. **Initialize Git Repository**:
   ```bash
   git init
   ```

2. **Stage and Commit Project Files**:
   ```bash
   git add .
   git commit -m "Initial commit of Laptop Price Prediction project"
   ```

3. **Link to GitHub Repository and Push**:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   git push -u origin main
   ```

---

## ☁️ Render Deployment Guide

Deploy your Flask app to Render in minutes using these steps:

1. **Create a New Web Service on Render**:
   - Connect your GitHub repository.
2. **Configure Service Details**:
   - **Environment**: `Python`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     gunicorn app:app
     ```
3. **Select Plan**: Choose the **Free** or **Starter** tier.
4. **Deploy**: Click **Create Web Service**.

---

## 🎨 Project Screenshots

The application uses a **Glassmorphism Dark Theme** designed to feel futuristic, clean, and premium:
- **Color Palette**: Cyberpunk neon blues (`#00f0ff`) and dark slates.
- **Hover Transitions**: Dynamic micro-animations on interactive selectors and buttons.
- **Responsive Layout**: Adapts gracefully to desktop, tablet, and mobile screens.

---

## 🔮 Future Improvements
- **Continuous Learning**: Add a pipeline to log inputs and retrain the model when target actual prices are submitted.
- **Multilingual Support**: Support dynamic localization for international markets.
- **Currency Converter**: Connect with real-time exchange rates to display predictions in USD, GBP, or INR dynamically.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
