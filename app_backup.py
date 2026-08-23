import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Student Academic Analyzer",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/students.csv")

    df["SGPA"] = pd.to_numeric(df["SGPA"], errors="coerce")
    df["Credits_Earned"] = pd.to_numeric(
        df["Credits_Earned"],
        errors="coerce"
    )

    return df


df = load_data()

# -----------------------------
# TITLE
# -----------------------------
st.title("🎓 Student Academic Performance Analyzer")
st.caption(
    "B.Tech Mechanical Engineering — Semester 1"
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🔎 Student Search")

search = st.sidebar.text_input(
    "Search student name or roll number"
)

if search:
    filtered = df[
        df["Name"].astype(str).str.contains(
            search,
            case=False,
            na=False
        )
        |
        df["Roll_No"].astype(str).str.contains(
            search,
            case=False,
            na=False
        )
    ]
else:
    filtered = df.copy()

# -----------------------------
# KPI CARDS
# -----------------------------
total_students = len(df)
average_sgpa = df["SGPA"].mean()
highest_sgpa = df["SGPA"].max()
full_credits = (df["Credits_Earned"] >= 20).sum()

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "👨‍🎓 Total Students",
    total_students
)

c2.metric(
    "📊 Average SGPA",
    f"{average_sgpa:.2f}"
)

c3.metric(
    "🏆 Highest SGPA",
    f"{highest_sgpa:.2f}"
)

c4.metric(
    "🎯 20 Credits Earned",
    full_credits
)

st.divider()

# -----------------------------
# STUDENT SEARCH RESULT
# -----------------------------
if search:

    st.subheader("🔍 Search Result")

    if len(filtered) == 0:
        st.warning("No student found.")

    else:
        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

st.subheader("📈 SGPA Distribution")

fig = px.histogram(
    df,
    x="SGPA",
    nbins=10,
    title="Distribution of SGPA"
)

fig.update_layout(
    xaxis_title="SGPA",
    yaxis_title="Number of Students"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# SUBJECT PERFORMANCE
# -----------------------------
st.subheader("📚 Subject-wise Grade Analysis")

subject_columns = [
    "BSC_101",
    "ESC_101",
    "BSC_102",
    "ESC_102",
    "BSC_103",
    "ESC_103",
    "HSMC_103"
]

subject_names = {
    "BSC_101": "BSC-101",
    "ESC_101": "ESC-101",
    "BSC_102": "BSC-102",
    "ESC_102": "ESC-102",
    "BSC_103": "BSC-103",
    "ESC_103": "ESC-103",
    "HSMC_103": "HSMC-103"
}

grade_data = []

for subject in subject_columns:

    counts = df[subject].value_counts()

    for grade, count in counts.items():

        grade_data.append({
            "Subject": subject_names[subject],
            "Grade": str(grade),
            "Students": int(count)
        })

grade_df = pd.DataFrame(grade_data)

fig2 = px.bar(
    grade_df,
    x="Subject",
    y="Students",
    color="Grade",
    barmode="group",
    title="Grade Distribution by Subject"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# -----------------------------
# SGPA TOP STUDENTS
# -----------------------------
st.subheader("🏆 Top Performers")

top_students = (
    df.sort_values(
        "SGPA",
        ascending=False
    )
    .head(10)
)

st.dataframe(
    top_students[
        [
            "Roll_No",
            "Name",
            "SGPA",
            "Credits_Earned"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# BACKLOG / CREDIT ANALYSIS
# -----------------------------
st.subheader("📌 Academic Status")

status_df = pd.DataFrame({
    "Status": [
        "20 Credits",
        "Less than 20 Credits"
    ],
    "Students": [
        (df["Credits_Earned"] >= 20).sum(),
        (df["Credits_Earned"] < 20).sum()
    ]
})

fig3 = px.pie(
    status_df,
    names="Status",
    values="Students",
    title="Credit Completion Status"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# -----------------------------
# COMPLETE DATA
# -----------------------------
st.subheader("📋 Complete Student Results")

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.caption(
    "Student Academic Performance Analyzer | "
    "Semester 1 Result Analysis"
)
