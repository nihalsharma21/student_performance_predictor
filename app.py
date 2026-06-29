import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

#  PAGE CONFIG
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  CUSTOM CSS
st.markdown("""
<style>
    /* ── Base & Font ─────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ──────────────────────────── */
    .main { background-color: #F0F4F8; }
    .block-container { padding: 2rem 2.5rem 3rem 2.5rem; }

    /* ── Header banner ───────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #1A4A6B 0%, #0E7490 50%, #10B981 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        color: white;
        margin-bottom: 1.8rem;
    }
    .hero-banner h1 {
        font-family: 'Sora', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        font-size: 1.0rem;
        opacity: 0.88;
        margin: 0;
    }

    /* ── Metric cards ────────────────────────── */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #0E7490;
        margin-bottom: 1rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.78rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.8px; }
    .metric-card .big-num { font-family: 'Sora', sans-serif; font-size: 2rem; font-weight: 700; color: #0E7490; margin: 0; }

    /* ── Warning cards ───────────────────────── */
    .warn-card {
        background: #FFF7ED;
        border: 1px solid #FDBA74;
        border-left: 4px solid #F97316;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
    }
    .warn-card b { color: #C2410C; }

    .danger-card {
        background: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-left: 4px solid #EF4444;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
    }
    .danger-card b { color: #B91C1C; }

    .success-card {
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-left: 4px solid #22C55E;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
    }
    .success-card b { color: #15803D; }

    /* ── Section headings ────────────────────── */
    .section-title {
        font-family: 'Sora', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E3A5F;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #E2E8F0;
        margin-bottom: 1rem;
    }

    /* ── Predict result pill ─────────────────── */
    .pill-high   { background:#D1FAE5; color:#065F46; padding:0.5rem 1.4rem; border-radius:50px; font-weight:700; font-size:1.1rem; display:inline-block; }
    .pill-medium { background:#FEF3C7; color:#92400E; padding:0.5rem 1.4rem; border-radius:50px; font-weight:700; font-size:1.1rem; display:inline-block; }
    .pill-low    { background:#FEE2E2; color:#991B1B; padding:0.5rem 1.4rem; border-radius:50px; font-weight:700; font-size:1.1rem; display:inline-block; }

    /* ── Sidebar ─────────────────────────────── */
    [data-testid="stSidebar"] { background-color: #1A4A6B; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSlider > div > div { background: #0E7490 !important; }

    /* ── Tab styling ─────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] { background-color: white; color: #0E7490 !important; }

    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

#  HELPERS
GRADE_COLORS = {"High": "#10B981", "Medium": "#F59E0B", "Low": "#EF4444"}

def load_data(uploaded=None):
    if uploaded:
        return pd.read_csv(uploaded)
    return pd.read_csv("student_data_200.csv")

def train_model(df):
    features = ["attendance_pct", "study_hours_per_day", "assignments_completed_pct",
                 "sleep_hours", "extracurricular_hours", "previous_score", "motivation_score"]
    X = df[features]
    y = df["grade_category"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, le, features, acc

def get_warnings(row):
    warnings_list = []
    if row.get("attendance_pct", 100) < 75:
        warnings_list.append(("danger", "⚠️ Attendance below 75% — at serious risk of failing. Must attend regularly."))
    elif row.get("attendance_pct", 100) < 85:
        warnings_list.append(("warn", "📅 Attendance is moderate. Try to attend at least 85% of classes."))
    if row.get("study_hours_per_day", 3) < 1.5:
        warnings_list.append(("danger", "📚 Study hours critically low (< 1.5 hrs/day). Increase immediately."))
    elif row.get("study_hours_per_day", 3) < 2.5:
        warnings_list.append(("warn", "📖 Studying less than 2.5 hrs/day. Aim for at least 3 hours for good results."))
    if row.get("assignments_completed_pct", 100) < 60:
        warnings_list.append(("danger", "📝 Assignment completion is very low. Missing assignments hurt your grade directly."))
    elif row.get("assignments_completed_pct", 100) < 80:
        warnings_list.append(("warn", "📋 Assignment completion under 80%. Catch up on pending work."))
    if row.get("sleep_hours", 7) < 5.5:
        warnings_list.append(("warn", "😴 Less than 5.5 hrs of sleep reduces focus and retention. Aim for 7-8 hrs."))
    if row.get("motivation_score", 5) < 4:
        warnings_list.append(("warn", "💡 Motivation is low. Set short-term goals and celebrate small wins."))
    if row.get("extracurricular_hours", 2) > 5:
        warnings_list.append(("warn", "⚽ High extracurricular hours (> 5 hrs/day). Balance activities with study time."))
    return warnings_list

#  SIDEBAR
with st.sidebar:
    st.markdown("## 🎓 Student Predictor")
    st.markdown("---")
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader("CSV with student data", type=["csv"])
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
This tool uses **Random Forest** classification to predict student grade categories.
**Grade Categories:**
- 🟢 **High** → Score 80–100
- 🟡 **Medium** → Score 60–79
- 🔴 **Low** → Score below 60

Built with Python · Scikit-Learn · Streamlit
    """)

#  LOAD DATA & TRAIN
df = load_data(uploaded_file)
clf, le, features, model_acc = train_model(df)

#  HERO
st.markdown("""
<div class="hero-banner">
  <h1>🎓 Student Performance Predictor</h1>
  <p>Machine learning tool that predicts grade categories (High / Medium / Low) based on attendance, study habits, and daily routines — giving teachers and students early warnings before it's too late.</p>
</div>
""", unsafe_allow_html=True)

#  TOP METRICS
total = len(df)
high_n   = (df["grade_category"] == "High").sum()
medium_n = (df["grade_category"] == "Medium").sum()
low_n    = (df["grade_category"] == "Low").sum()

c1, c2, c3, c4, c5 = st.columns(5)
for col, label, val, color in zip(
    [c1, c2, c3, c4, c5],
    ["Total Students", "🟢 High Performers", "🟡 Medium Performers", "🔴 At Risk (Low)", "Model Accuracy"],
    [total, high_n, medium_n, low_n, f"{model_acc*100:.1f}%"],
    ["#0E7490", "#10B981", "#F59E0B", "#EF4444", "#6366F1"]
):
    col.markdown(f"""
    <div class="metric-card" style="border-left-color:{color}">
        <h3>{label}</h3>
        <p class="big-num" style="color:{color}">{val}</p>
    </div>""", unsafe_allow_html=True)

#  TABS
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predict Student", "⚠️ At-Risk Students", "📋 Data & Report"])

#  TAB 1 — DASHBOARD
with tab1:
    st.markdown('<p class="section-title">📈 Performance Overview</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # Grade distribution donut
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df["grade_category"].value_counts()
        colors = [GRADE_COLORS.get(k, "#CBD5E1") for k in counts.index]
        wedges, texts, autotexts = ax.pie(
            counts, labels=counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90,
            wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
            textprops={'fontsize': 11, 'fontweight': '600'}
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_color('white')
            at.set_fontweight('bold')
        ax.set_title("Grade Category Distribution", fontsize=13, fontweight='bold', pad=14, color='#1E3A5F')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('none')
        st.pyplot(fig)

    # Attendance vs Score scatter
    with col_b:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        for cat, color in GRADE_COLORS.items():
            sub = df[df["grade_category"] == cat]
            ax2.scatter(sub["attendance_pct"], sub["previous_score"],
                        c=color, label=cat, alpha=0.8, s=70, edgecolors='white', linewidth=0.5)
        ax2.set_xlabel("Attendance %", fontsize=10, color='#475569')
        ax2.set_ylabel("Previous Score", fontsize=10, color='#475569')
        ax2.set_title("Attendance vs Previous Score", fontsize=13, fontweight='bold', color='#1E3A5F')
        ax2.legend(title="Grade", fontsize=9)
        ax2.grid(True, alpha=0.15)
        ax2.set_facecolor('#F8FAFC')
        fig.patch.set_facecolor('white')
        st.pyplot(fig2)

    col_c, col_d = st.columns(2)

    # Study hours distribution per category
    with col_c:
        fig3, ax3 = plt.subplots(figsize=(5, 3.8))
        for cat, color in GRADE_COLORS.items():
            sub = df[df["grade_category"] == cat]["study_hours_per_day"]
            ax3.hist(sub, bins=8, alpha=0.65, color=color, label=cat, edgecolor='white')
        ax3.set_xlabel("Study Hours / Day", fontsize=10, color='#475569')
        ax3.set_ylabel("Number of Students", fontsize=10, color='#475569')
        ax3.set_title("Study Hours Distribution by Grade", fontsize=12, fontweight='bold', color='#1E3A5F')
        ax3.legend(fontsize=9)
        ax3.set_facecolor('#F8FAFC')
        fig.patch.set_facecolor('white')
        st.pyplot(fig3)

    # Feature importance
    with col_d:
        fi = pd.Series(clf.feature_importances_, index=features).sort_values()
        fig4, ax4 = plt.subplots(figsize=(5, 3.8))
        colors_fi = ['#10B981' if v > 0.15 else '#0E7490' if v > 0.10 else '#94A3B8' for v in fi.values]
        bars = ax4.barh(fi.index, fi.values, color=colors_fi, edgecolor='white', height=0.6)
        ax4.set_xlabel("Importance", fontsize=10, color='#475569')
        ax4.set_title("Feature Importance (Random Forest)", fontsize=12, fontweight='bold', color='#1E3A5F')
        ax4.set_facecolor('#F8FAFC')
        for bar, val in zip(bars, fi.values):
            ax4.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                     va='center', fontsize=8.5, color='#334155')
        fig.patch.set_facecolor('white')
        st.pyplot(fig4)

    # Avg metrics by category
    st.markdown('<p class="section-title">📊 Average Habits by Grade Category</p>', unsafe_allow_html=True)
    avg = df.groupby("grade_category")[["attendance_pct", "study_hours_per_day",
                                         "assignments_completed_pct", "sleep_hours",
                                         "motivation_score"]].mean().round(2)
    avg.columns = ["Avg Attendance %", "Study Hrs/Day", "Assignments Done %", "Sleep Hrs", "Motivation (1–10)"]
    st.dataframe(avg.style.background_gradient(cmap="YlGn", axis=0), use_container_width=True)

#  TAB 2 — PREDICT
with tab2:
    st.markdown('<p class="section-title">🔮 Predict a Student\'s Grade Category</p>', unsafe_allow_html=True)
    st.markdown("Enter a student's habits below to predict their performance and receive personalised warnings.")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            student_name = st.text_input("Student Name", placeholder="e.g. Aarav Sharma")
            attendance   = st.slider("Attendance %", 30, 100, 80)
            study_hrs    = st.slider("Study Hours / Day", 0.0, 8.0, 3.0, 0.5)
            assignments  = st.slider("Assignments Completed %", 0, 100, 80)
        with col2:
            sleep_hrs    = st.slider("Sleep Hours / Night", 3.0, 10.0, 7.0, 0.5)
            extra_hrs    = st.slider("Extracurricular Hours / Day", 0.0, 8.0, 2.0, 0.5)
            prev_score   = st.slider("Previous Exam Score", 0, 100, 70)
            motivation   = st.slider("Motivation Score (1–10)", 1, 10, 6)

        submitted = st.form_submit_button("🎯 Predict Performance", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([[attendance, study_hrs, assignments,
                                     sleep_hrs, extra_hrs, prev_score, motivation]],
                                   columns=features)
        pred_enc = clf.predict(input_data)[0]
        pred_proba = clf.predict_proba(input_data)[0]
        pred_label = le.inverse_transform([pred_enc])[0]

        pill_class = f"pill-{pred_label.lower()}"
        emoji = "🟢" if pred_label == "High" else "🟡" if pred_label == "Medium" else "🔴"

        st.markdown("---")
        r1, r2 = st.columns([1, 2])
        with r1:
            name_display = student_name if student_name else "Student"
            st.markdown(f"### Result for **{name_display}**")
            st.markdown(f'<div style="margin:0.5rem 0"><span class="{pill_class}">{emoji} {pred_label} Performer</span></div>', unsafe_allow_html=True)

            for i, cat in enumerate(le.classes_):
                prob_pct = pred_proba[i] * 100
                color = GRADE_COLORS.get(cat, "#94A3B8")
                st.markdown(f"""
                <div style="margin:4px 0; font-size:0.85rem; color:#475569">
                    <span style="font-weight:600; color:{color}">{cat}</span>
                    <div style="background:#E2E8F0;border-radius:4px;height:8px;margin-top:3px">
                        <div style="width:{prob_pct:.1f}%;background:{color};height:8px;border-radius:4px"></div>
                    </div>
                    <span style="font-size:0.78rem">{prob_pct:.1f}%</span>
                </div>""", unsafe_allow_html=True)

        with r2:
            row_data = {
                "attendance_pct": attendance, "study_hours_per_day": study_hrs,
                "assignments_completed_pct": assignments, "sleep_hours": sleep_hrs,
                "extracurricular_hours": extra_hrs, "motivation_score": motivation
            }
            warns = get_warnings(row_data)
            if warns:
                st.markdown("### ⚠️ Personalised Warnings")
                for wtype, msg in warns:
                    card_cls = "danger-card" if wtype == "danger" else "warn-card"
                    st.markdown(f'<div class="{card_cls}">{msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-card">
                    ✅ <b>All habits look great!</b> Keep up the excellent study routine, attendance, and sleep schedule.
                </div>""", unsafe_allow_html=True)

            # Radar-style bar chart of student vs class averages
            st.markdown("#### Your Habits vs Class Average")
            metrics_labels = ["Attendance %", "Study Hrs×10", "Assignments %", "Sleep×10", "Motivation×10"]
            student_vals = [attendance, study_hrs * 10, assignments, sleep_hrs * 10, motivation * 10]
            avg_vals = [
                df["attendance_pct"].mean(),
                df["study_hours_per_day"].mean() * 10,
                df["assignments_completed_pct"].mean(),
                df["sleep_hours"].mean() * 10,
                df["motivation_score"].mean() * 10,
            ]
            x = np.arange(len(metrics_labels))
            fig5, ax5 = plt.subplots(figsize=(6, 3))
            ax5.bar(x - 0.2, student_vals, 0.38, label=name_display or "Student",
                    color='#0E7490', alpha=0.85, edgecolor='white')
            ax5.bar(x + 0.2, avg_vals, 0.38, label="Class Average",
                    color='#94A3B8', alpha=0.7, edgecolor='white')
            ax5.set_xticks(x)
            ax5.set_xticklabels(metrics_labels, fontsize=8.5, rotation=15, ha='right')
            ax5.legend(fontsize=9)
            ax5.set_ylim(0, 110)
            ax5.set_facecolor('#F8FAFC')
            fig.patch.set_facecolor('white')
            fig5.tight_layout()
            st.pyplot(fig5)

#  TAB 3 — AT-RISK STUDENTS
with tab3:
    st.markdown('<p class="section-title">⚠️ Students Who Need Attention</p>', unsafe_allow_html=True)

    # Score-based risk
    low_students = df[df["grade_category"] == "Low"].copy()
    medium_students = df[df["grade_category"] == "Medium"].copy()

    st.markdown(f"#### 🔴 Critical Risk — Low Performers ({len(low_students)} students)")
    if not low_students.empty:
        for _, row in low_students.iterrows():
            name = row.get("name", row.get("student_id", "Unknown"))
            warns = get_warnings(row)
            warn_texts = " · ".join([msg for _, msg in warns[:2]]) if warns else "Low overall performance."
            st.markdown(f"""
            <div class="danger-card">
                <b>{name}</b> — Attendance: {row['attendance_pct']:.0f}% · Study: {row['study_hours_per_day']:.1f} hrs/day · Score: {row['previous_score']}<br>
                <span style="font-size:0.83rem; color:#7F1D1D">{warn_texts}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.success("No students in the Low category.")

    st.markdown(f"#### 🟡 Medium Performers — Need Improvement ({len(medium_students)} students)")
    if not medium_students.empty:
        for _, row in medium_students.iterrows():
            name = row.get("name", row.get("student_id", "Unknown"))
            warns = get_warnings(row)
            warn_texts = " · ".join([msg for _, msg in warns[:1]]) if warns else "On track — push a bit more."
            st.markdown(f"""
            <div class="warn-card">
                <b>{name}</b> — Attendance: {row['attendance_pct']:.0f}% · Study: {row['study_hours_per_day']:.1f} hrs/day · Score: {row['previous_score']}<br>
                <span style="font-size:0.83rem; color:#92400E">{warn_texts}</span>
            </div>""", unsafe_allow_html=True)

    # Attendance heatmap
    st.markdown("---")
    st.markdown('<p class="section-title">📊 Correlation Heatmap — Habit Factors</p>', unsafe_allow_html=True)
    num_cols = ["attendance_pct", "study_hours_per_day", "assignments_completed_pct",
                "sleep_hours", "extracurricular_hours", "previous_score", "motivation_score"]
    corr = df[num_cols].corr()
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=0.5, linecolor='white',
                annot_kws={"size": 9}, ax=ax6)
    ax6.set_title("Correlation Between Student Habits & Scores", fontsize=13, fontweight='bold',
                  color='#1E3A5F', pad=12)
    fig.patch.set_facecolor('white')
    fig6.tight_layout()
    st.pyplot(fig6)

#  TAB 4 — DATA & REPORT
with tab4:
    st.markdown('<p class="section-title">📋 Full Student Dataset</p>', unsafe_allow_html=True)

    # Filters
    f1, f2 = st.columns(2)
    grade_filter = f1.multiselect("Filter by Grade", ["High", "Medium", "Low"],
                                   default=["High", "Medium", "Low"])
    name_search = f2.text_input("Search by Name / ID", placeholder="Type student name…")

    filtered = df[df["grade_category"].isin(grade_filter)]
    if name_search:
        search_col = "name" if "name" in df.columns else "student_id"
        filtered = filtered[filtered[search_col].str.contains(name_search, case=False, na=False)]

    def color_grade(val):
        c = {"High": "background-color:#D1FAE5;color:#065F46",
             "Medium": "background-color:#FEF3C7;color:#92400E",
             "Low": "background-color:#FEE2E2;color:#991B1B"}.get(val, "")
        return c

    styled_df = filtered.style.map(color_grade, subset=["grade_category"])
    st.dataframe(styled_df, use_container_width=True, height=320)

    # Summary report
    st.markdown("---")
    st.markdown('<p class="section-title">📄 Summary Report</p>', unsafe_allow_html=True)

    report_cols = st.columns(3)
    for col, cat in zip(report_cols, ["High", "Medium", "Low"]):
        sub = df[df["grade_category"] == cat]
        color = GRADE_COLORS[cat]
        emoji_map = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
        col.markdown(f"""
        <div style="background:white;border-radius:12px;padding:1.2rem;box-shadow:0 2px 10px rgba(0,0,0,0.06);border-top:4px solid {color}">
            <div style="font-family:'Sora',sans-serif;font-size:1.05rem;font-weight:700;color:{color};margin-bottom:0.8rem">
                {emoji_map[cat]} {cat} Performers ({len(sub)})
            </div>
            <div style="font-size:0.84rem;color:#475569;line-height:1.9">
                📅 Avg Attendance: <b>{sub['attendance_pct'].mean():.1f}%</b><br>
                📚 Avg Study Hrs: <b>{sub['study_hours_per_day'].mean():.1f} hrs/day</b><br>
                📝 Assignments: <b>{sub['assignments_completed_pct'].mean():.1f}%</b><br>
                😴 Sleep: <b>{sub['sleep_hours'].mean():.1f} hrs</b><br>
                💡 Motivation: <b>{sub['motivation_score'].mean():.1f}/10</b><br>
                🎯 Avg Score: <b>{sub['previous_score'].mean():.1f}</b>
            </div>
        </div>""", unsafe_allow_html=True)

    # CSV download
    st.markdown("---")
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=filtered.to_csv(index=False),
        file_name="student_performance_filtered.csv",
        mime="text/csv",
        use_container_width=True
    )
