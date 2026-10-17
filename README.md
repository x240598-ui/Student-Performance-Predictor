# Student Performance Prediction 🎓

A beginner-friendly Machine Learning project that predicts a student's
**Performance Index** from five study-related features using **Linear Regression**.

---

## 📁 Project Files

| File | Purpose |
|------|---------|
| `Student_Performance.csv` | Dataset (10 000 rows, 6 columns) |
| `student_performance.py` | Full ML pipeline — EDA, training, graphs, model save |
| `app.py` | Streamlit web interface |
| `requirements.txt` | Python package dependencies |
| `outputs/` | Auto-generated graphs (created when you run the pipeline) |
| `linear_regression_model.pkl` | Saved model (created when you run the pipeline) |

---

## ⚙️ Step-by-Step Setup in VS Code

### 1 · Open the Project Folder
```
File → Open Folder → select the folder containing these files
```

### 2 · Create a Virtual Environment (recommended)
Open the **VS Code integrated terminal** (`Ctrl + backtick`) and run:
```bash
python -m venv venv
```
Activate it:
- **Windows:**  `venv\Scripts\activate`
- **Mac / Linux:** `source venv/bin/activate`

VS Code may prompt *"We noticed a new virtual environment was created…"* — click **Yes** to select it as the interpreter.

### 3 · Install Dependencies
```bash
pip install -r requirements.txt
```

### 4 · Run the ML Pipeline (EDA + Training + Graphs)
```bash
python student_performance.py
```
This will:
- Print EDA results (missing values, duplicates, statistics)
- Train the Linear Regression model
- Save four graphs to the `outputs/` folder
- Save `linear_regression_model.pkl`

### 5 · Launch the Streamlit Web App
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

---

## 📊 Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `Hours Studied` | int | Daily study hours (1–9) |
| `Previous Scores` | int | Previous exam score (40–99) |
| `Extracurricular Activities` | str (Yes/No) | Participates in activities |
| `Sleep Hours` | int | Daily sleep hours (4–9) |
| `Sample Question Papers Practiced` | int | Papers practised (0–9) |
| `Performance Index` | float | **Target** — overall score (0–100) |

---

## 📈 Evaluation Metrics (expected range)

| Metric | Meaning |
|--------|---------|
| MAE | Average absolute prediction error |
| MSE | Mean of squared errors |
| RMSE | Square root of MSE (same unit as target) |
| R² | How much variance the model explains (closer to 1 is better) |

---

## 🔮 Using the Web App

1. Adjust the five sliders/radio buttons to your values.
2. Click **Predict Performance Index**.
3. Your estimated score and performance level appear instantly.
4. Sidebar shows dataset graphs (after running the pipeline once).

---

## 🐍 Python Version
Tested with **Python 3.10 / 3.11**. Any Python ≥ 3.9 should work.
