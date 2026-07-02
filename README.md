# 🎓 Student Performance Predictor

A machine learning web app that predicts student grade categories (**High / Medium / Low**) based on daily habits — and warns students before it's too late.

---

## 📁 Project Files

```
student-performance-predictor/
├── app.py                   ← Main Streamlit application
├── student_data_200.csv     ← Sample dataset (200 students)
├── requirements.txt         ← Python dependencies
└── README.md                ← This file
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## How It Works

| Feature | Description |
|---|---|
| **Attendance %** | How often the student attends class |
| **Study Hours/Day** | Daily self-study time |
| **Assignments Completed %** | Percentage of homework submitted |
| **Sleep Hours** | Nightly sleep duration |
| **Extracurricular Hours** | Time spent on activities beyond studies |
| **Previous Score** | Last exam score (0–100) |
| **Motivation Score** | Self-rated motivation (1–10) |

### ML Model: Random Forest Classifier
- Trained on the student dataset
- Predicts: **High** (80+) · **Medium** (60–79) · **Low** (below 60)
- Shows prediction probability for each category

---

## ✨ Features

### 📊 Dashboard Tab
- Grade distribution donut chart
- Attendance vs Score scatter plot
- Study hours histogram by grade
- Feature importance chart
- Habit averages by grade category

### 🔮 Predict Tab
- Enter any student's habits with sliders
- Get instant grade prediction with confidence scores
- Personalised warnings (attendance, study, sleep, etc.)
- Compare student habits vs class average bar chart

### ⚠️ At-Risk Tab
- Lists all Low and Medium performers with warnings
- Correlation heatmap of all habit factors

### 📋 Data & Report Tab
- Full filterable and searchable dataset
- Grade-wise summary report cards
- CSV download of filtered data

---

## 📊 Dataset Columns

```
student_id, name, attendance_pct, study_hours_per_day,
assignments_completed_pct, sleep_hours, extracurricular_hours,
previous_score, motivation_score, grade_category
```

You can upload your own CSV using the sidebar — it must contain these same columns.

---

## ⚠️ Warning Rules

| Condition | Warning Level |
|---|---|
| Attendance < 75% | 🔴 Critical |
| Attendance 75–85% | 🟡 Warning |
| Study < 1.5 hrs/day | 🔴 Critical |
| Study 1.5–2.5 hrs/day | 🟡 Warning |
| Assignments < 60% | 🔴 Critical |
| Sleep < 5.5 hrs | 🟡 Warning |
| Motivation < 4 | 🟡 Warning |
| Extracurricular > 5 hrs | 🟡 Warning |

---

## 🛠 Tech Stack

- **Python** — Core language
- **Pandas** — Data manipulation
- **Scikit-Learn** — Random Forest classification
- **Matplotlib / Seaborn** — Visualizations
- **Streamlit** — Web application framework
