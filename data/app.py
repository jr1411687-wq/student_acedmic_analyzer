import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import re
import io

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Student Academic Analyzer",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# GENDER DETECTION MODEL CLASS
# =========================================================

class GenderDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        # Internal database - never displayed
        self.name_database = {
            'male': [
                'abhi', 'anmol', 'anshveer', 'anuj', 'armaandeep',
                'arnav', 'arun', 'ayush', 'biplove', 'devansh',
                'gaganpreet', 'gurbaaj', 'gurbaksh', 'gurkanwar',
                'gurkirat', 'gurnoor', 'gurpreet', 'gursimar',
                'gurtej', 'hardaksh', 'harmanpreet', 'harsimrat',
                'ishan', 'karan', 'karanveer', 'keshav', 'krishan',
                'mankaran', 'manmeet', 'mauli', 'md sahil', 'navjot',
                'nikhil', 'prabpreet', 'rakesh', 'reetinder', 'rishu',
                'sahibpreet', 'sahil', 'samarveer', 'samridh',
                'sanyam', 'sharanjit', 'shubh', 'snimerbir',
                'snover', 'subham', 'swarnava', 'tajvir', 'tamish',
                'tripanjeet', 'udeybir', 'varun', 'vikramjeet',
                'vinay', 'gurshaan', 'arshdeep', 'dipanshu',
                'harman', 'jasnoor', 'manish', 'parmeet',
                'prabhjot', 'rhythm', 'prabhsimran'
            ],
            'female': [
                'akriti', 'alpana', 'bhavya', 'chanpreet', 'chetna',
                'gunsimranjot', 'gurunaamdeep', 'harmandeep', 'harpreet',
                'harsimar', 'navneet', 'nisha', 'prabhleen', 'rashi',
                'rajwant', 'ramandeep', 'ravinder', 'simran',
                'sukhmeet', 'taniya', 'tanu', 'anshika', 'priya',
                'neha', 'pooja', 'anjali', 'sneha', 'divya', 'isha',
                'riya', 'shreya', 'kajal', 'jasmine'
            ]
        }
        self.suffix_patterns = {
            'male': ['singh', 'kumar', 'sharma', 'verma', 'gupta', 'jain', 
                    'agarwal', 'malhotra', 'mehra', 'kapoor', 'khanna', 'chopra',
                    'arora', 'gill', 'sodhi', 'galib', 'sahni', 'sofat', 
                    'sehgal', 'bhangu', 'rattu', 'mourya', 'bhaskar', 'ahuja',
                    'pal', 'thakur', 'tagger', 'yadav', 'patel', 'shah',
                    'joshi', 'pandey', 'tiwari', 'dubey', 'mishra'],
            'female': ['kaur', 'devi', 'rani', 'kumari']
        }
        self.training_count = {
            'male': len(self.name_database['male']),
            'female': len(self.name_database['female'])
        }
        # Auto-train on initialization if model doesn't exist
        if not os.path.exists('gender_model.pkl'):
            self.auto_train()
        else:
            self.load_model('gender_model.pkl')
    
    def auto_train(self):
        """Automatically train the model with existing data"""
        try:
            accuracy = self.train_model()
            if accuracy:
                self.save_model('gender_model.pkl')
            return accuracy
        except:
            return None
    
    def detect_gender(self, name):
        """Automatically detect gender - no names displayed"""
        if pd.isna(name) or name == '':
            return 'Unknown'
        
        name = str(name).strip().lower()
        # Clean the name
        name = name.replace('/', '').replace('\\', '').strip()
        
        # Split name to get first name for better detection
        name_parts = name.split()
        first_name = name_parts[0] if name_parts else name
        
        # Method 1: Check in database (check full name and first name)
        for gender, names in self.name_database.items():
            if name in names or first_name in names:
                return 'Male' if gender == 'male' else 'Female'
        
        # Method 2: Check for 'KAUR' (Female) vs 'SINGH' (Male)
        if 'kaur' in name:
            return 'Female'
        if 'singh' in name:
            # Check if it's a female name with Singh
            for female in self.name_database['female']:
                if female in name or female in first_name:
                    return 'Female'
            return 'Male'
        
        # Method 3: Check for suffixes
        for gender, suffixes in self.suffix_patterns.items():
            for suffix in suffixes:
                if name.endswith(suffix) or first_name.endswith(suffix):
                    return 'Male' if gender == 'male' else 'Female'
        
        # Method 4: Pattern recognition - Female names often end with 'a', 'i', 'e'
        if first_name.endswith('a') or first_name.endswith('i') or first_name.endswith('e'):
            male_exceptions = ['arjun', 'karan', 'mohan', 'rahul', 'ankit', 'vikram', 
                              'raj', 'amit', 'suresh', 'ramesh', 'mahesh']
            if first_name not in male_exceptions:
                if first_name not in self.name_database['male']:
                    return 'Female'
        
        # Method 5: First letter patterns
        male_first_letters = ['a', 'b', 'd', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'r', 's', 't', 'u', 'v']
        
        if first_name:
            first_letter = first_name[0]
            
            if first_letter in male_first_letters:
                # Names ending with 'eet' or 'een' or 'aj' are often male
                if first_name.endswith('eet') or first_name.endswith('een') or first_name.endswith('aj'):
                    return 'Male'
                
                # Check for male name endings
                male_endings = ['bir', 'jeet', 'preet', 'deep', 'inder', 'veer', 'pal']
                for ending in male_endings:
                    if first_name.endswith(ending):
                        return 'Male'
        
        # Method 6: Use trained model if available
        if self.model is not None and self.vectorizer is not None:
            try:
                name_vectorized = self.vectorizer.transform([name])
                prediction = self.model.predict(name_vectorized)
                return prediction[0]
            except:
                pass
        
        return 'Unknown'
    
    def train_model(self):
        """Train the machine learning model"""
        names = []
        genders = []
        
        # Add existing database
        for gender, names_list in self.name_database.items():
            for name in names_list:
                names.append(name)
                genders.append('Male' if gender == 'male' else 'Female')
        
        # Add common patterns
        male_suffixes = ['singh', 'kumar', 'sharma', 'verma', 'gupta', 'jain',
                        'yadav', 'patel', 'shah', 'joshi', 'pandey']
        for suffix in male_suffixes:
            names.append(suffix)
            genders.append('Male')
        
        female_suffixes = ['kaur', 'devi', 'kumari']
        for suffix in female_suffixes:
            names.append(suffix)
            genders.append('Female')
        
        # Create feature vectors
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4))
        X = self.vectorizer.fit_transform(names)
        y = np.array(genders)
        
        # Split data
        if len(np.unique(y)) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.model = MultinomialNB()
            self.model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            return accuracy
        return None
    
    def save_model(self, filepath):
        """Save the trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'database': self.name_database,
                'suffixes': self.suffix_patterns,
                'training_count': self.training_count
            }, f)
    
    def load_model(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.vectorizer = data['vectorizer']
                    self.name_database = data['database']
                    self.suffix_patterns = data['suffixes']
                    self.training_count = data.get('training_count', 
                        {'male': len(self.name_database['male']), 
                         'female': len(self.name_database['female'])})
                return True
            except:
                return False
        return False
    
    def get_stats(self):
        """Get statistics without exposing names"""
        return {
            'male_count': self.training_count['male'],
            'female_count': self.training_count['female'],
            'total': self.training_count['male'] + self.training_count['female'],
            'model_trained': self.model is not None
        }

# Initialize the gender detector
gender_detector = GenderDetector()

# =========================================================
# DATA FILES
# =========================================================

DATASETS = {
    "B.Tech Mechanical Engineering": {
        "Semester 1": "data/students.csv"
    },

    "B.Tech Electronics & Communication Engineering": {
        "Semester 3": "data/ece_sem3.csv"
    }
}

# =========================================================
# FUNCTION TO PROCESS CSV WITH GENDER
# =========================================================

def process_csv_with_gender(df):
    """Process CSV and add gender column"""
    # Convert numeric columns if they exist
    if "SGPA" in df.columns:
        df["SGPA"] = pd.to_numeric(df["SGPA"], errors="coerce")
    
    if "Credits_Earned" in df.columns:
        df["Credits_Earned"] = pd.to_numeric(df["Credits_Earned"], errors="coerce")
    
    # Ensure existing gender column is properly formatted
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip()
        df["Gender"] = df["Gender"].replace({
            'M': 'Male',
            'F': 'Female',
            'm': 'Male',
            'f': 'Female',
            'Male': 'Male',
            'Female': 'Female'
        })
    else:
        # Add Gender column if Name column exists
        if "Name" in df.columns:
            df["Gender"] = df["Name"].apply(gender_detector.detect_gender)
    
    return df

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = process_csv_with_gender(df)
    return df

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 Student Academic Analyzer")

st.sidebar.subheader("Academic Selection")

branch = st.sidebar.selectbox(
    "Select Branch",
    list(DATASETS.keys())
)

semester = st.sidebar.selectbox(
    "Select Semester",
    list(DATASETS[branch].keys())
)

file_path = DATASETS[branch][semester]

# Load selected dataset
df = load_data(file_path)

st.sidebar.success(
    f"Loaded: {len(df)} students"
)

st.sidebar.divider()

# =========================================================
# STUDENT SEARCH
# =========================================================

st.sidebar.subheader("🔎 Student Search")

search = st.sidebar.text_input(
    "Search student name or roll number"
)

# =========================================================
# CSV UPLOAD SECTION IN SIDEBAR
# =========================================================

st.sidebar.divider()
st.sidebar.subheader("📤 Upload Custom CSV")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=['csv'],
    help="Upload any CSV with student data"
)

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        uploaded_df = process_csv_with_gender(uploaded_df)
        st.sidebar.success(f"✅ Loaded: {len(uploaded_df)} students")
        
        # Use uploaded data instead of default
        df = uploaded_df
        branch = "Custom Dataset"
        semester = "Uploaded CSV"
        
        # Show gender stats in sidebar
        if "Gender" in df.columns:
            gender_counts = df["Gender"].value_counts()
            st.sidebar.info(f"""
            **Gender Distribution:**
            - 👨 Male: {gender_counts.get('Male', 0)}
            - 👩 Female: {gender_counts.get('Female', 0)}
            - ❓ Unknown: {gender_counts.get('Unknown', 0)}
            """)
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

# =========================================================
# TITLE
# =========================================================

st.title("🎓 Student Academic Performance Analyzer")

st.caption(
    f"{branch} — {semester}"
)

# =========================================================
# SEARCH FILTER
# =========================================================

if search:

    name_match = df["Name"].astype(str).str.contains(
        search,
        case=False,
        na=False
    )

    roll_match = df["Roll_No"].astype(str).str.contains(
        search,
        case=False,
        na=False
    )

    filtered = df[name_match | roll_match]

else:

    filtered = df.copy()


# =========================================================
# KPI DATA
# =========================================================

total_students = len(filtered)

average_sgpa = filtered["SGPA"].mean()

highest_sgpa = filtered["SGPA"].max()

full_credits = (
    filtered["Credits_Earned"] >= 20
).sum()


# =========================================================
# KPI CARDS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "👨‍🎓 Total Students",
    total_students
)

c2.metric(
    "📊 Average SGPA",
    f"{average_sgpa:.2f}" if pd.notna(average_sgpa) else "N/A"
)

c3.metric(
    "🏆 Highest SGPA",
    f"{highest_sgpa:.2f}" if pd.notna(highest_sgpa) else "N/A"
)

c4.metric(
    "🎯 20 Credits Earned",
    full_credits
)

st.divider()

# =========================================================
# GENDER RATIO ANALYSIS
# =========================================================

st.subheader("👥 Gender Ratio Analysis")

if "Gender" in filtered.columns and not filtered["Gender"].isna().all():

    # Calculate gender counts
    gender_counts = filtered["Gender"].value_counts()
    male_count = gender_counts.get('Male', 0)
    female_count = gender_counts.get('Female', 0)
    unknown_count = gender_counts.get('Unknown', 0)
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📈 Statistics", "📋 Gender-Separated Data", "🎯 Academic Performance"])
    
    with tab1:
        # Create columns for side-by-side display
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Pie Chart
            fig_gender_pie = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title=f"Gender Distribution (Total: {len(filtered)} students)",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#95A5A6'],
                hole=0.3
            )
            fig_gender_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=14
            )
            st.plotly_chart(fig_gender_pie, use_container_width=True)
        
        with col2:
            # Bar Chart
            gender_df = pd.DataFrame({
                'Gender': gender_counts.index,
                'Count': gender_counts.values,
                'Percentage': (gender_counts.values / len(filtered) * 100).round(1)
            })
            
            fig_gender_bar = px.bar(
                gender_df,
                x='Gender',
                y='Count',
                title="Gender Distribution (Count)",
                labels={'x': 'Gender', 'y': 'Number of Students'},
                color='Gender',
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#95A5A6'],
                text=gender_df['Percentage'].apply(lambda x: f"{x}%")
            )
            fig_gender_bar.update_traces(textposition='outside')
            fig_gender_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_gender_bar, use_container_width=True)
    
    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👨‍🎓 Total Students", len(filtered))
        
        with col2:
            male_percentage = (male_count / len(filtered) * 100) if len(filtered) > 0 else 0
            st.metric("👨 Male Students", f"{male_count} ({male_percentage:.1f}%)")
        
        with col3:
            female_percentage = (female_count / len(filtered) * 100) if len(filtered) > 0 else 0
            st.metric("👩 Female Students", f"{female_count} ({female_percentage:.1f}%)")
        
        with col4:
            ratio = f"{male_count}:{female_count}" if female_count > 0 else f"{male_count}:0"
            st.metric("⚖️ Male:Female Ratio", ratio)
        
        if unknown_count > 0:
            st.info(f"ℹ️ {unknown_count} students ({unknown_count/len(filtered)*100:.1f}%) have unknown gender")
        
        # Gender Distribution Summary Table
        st.subheader("📊 Gender Distribution Summary")
        gender_summary = pd.DataFrame({
            "Gender": ["Male", "Female", "Unknown", "Total"],
            "Count": [male_count, female_count, unknown_count, len(filtered)],
            "Percentage": [
                f"{male_count/len(filtered)*100:.1f}%" if len(filtered) > 0 else "0%",
                f"{female_count/len(filtered)*100:.1f}%" if len(filtered) > 0 else "0%",
                f"{unknown_count/len(filtered)*100:.1f}%" if len(filtered) > 0 else "0%",
                "100%"
            ]
        })
        st.dataframe(gender_summary, use_container_width=True, hide_index=True)
    
    with tab3:
        # Gender-Separated Data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👨 Male Students")
            male_df = filtered[filtered['Gender'] == 'Male']
            if not male_df.empty:
                display_cols = [col for col in male_df.columns if col != 'Name']
                st.dataframe(
                    male_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(f"Total: {len(male_df)} male students")
            else:
                st.info("No male students found")
        
        with col2:
            st.markdown("### 👩 Female Students")
            female_df = filtered[filtered['Gender'] == 'Female']
            if not female_df.empty:
                display_cols = [col for col in female_df.columns if col != 'Name']
                st.dataframe(
                    female_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(f"Total: {len(female_df)} female students")
            else:
                st.info("No female students found")
        
        if unknown_count > 0:
            st.divider()
            st.markdown("### ❓ Unknown Gender Students")
            unknown_df = filtered[filtered['Gender'] == 'Unknown']
            if not unknown_df.empty:
                display_cols = [col for col in unknown_df.columns if col != 'Name']
                st.dataframe(
                    unknown_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(f"Total: {len(unknown_df)} students with unknown gender")
    
    with tab4:
        # Academic Performance by Gender
        st.subheader("📊 Academic Performance by Gender")
        
        if "SGPA" in filtered.columns:
            # Calculate statistics by gender
            gender_performance = filtered.groupby("Gender").agg({
                "SGPA": ["count", "mean", "min", "max", "std"],
                "Credits_Earned": ["mean", "min", "max"]
            }).round(2)
            
            # Flatten column names
            gender_performance.columns = [
                'Count', 'Avg SGPA', 'Min SGPA', 'Max SGPA', 'Std SGPA', 
                'Avg Credits', 'Min Credits', 'Max Credits'
            ]
            
            # Display performance table
            st.dataframe(gender_performance, use_container_width=True)
            
            # Visualize average SGPA by gender
            col1, col2 = st.columns(2)
            
            with col1:
                avg_sgpa_df = gender_performance[['Avg SGPA']].reset_index()
                # Filter out Unknown if needed
                avg_sgpa_df = avg_sgpa_df[avg_sgpa_df['Gender'] != 'Unknown']
                
                if not avg_sgpa_df.empty:
                    fig_avg_sgpa = px.bar(
                        avg_sgpa_df,
                        x='Gender',
                        y='Avg SGPA',
                        title="Average SGPA by Gender",
                        labels={'x': 'Gender', 'y': 'Average SGPA'},
                        color='Gender',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#95A5A6'],
                        text='Avg SGPA'
                    )
                    fig_avg_sgpa.update_traces(textposition='outside')
                    fig_avg_sgpa.update_layout(showlegend=False)
                    st.plotly_chart(fig_avg_sgpa, use_container_width=True)
            
            with col2:
                # Box plot for SGPA distribution by gender
                df_filtered = filtered[filtered['Gender'] != 'Unknown']
                if not df_filtered.empty and 'SGPA' in df_filtered.columns:
                    fig_box = px.box(
                        df_filtered,
                        x='Gender',
                        y='SGPA',
                        title="SGPA Distribution by Gender",
                        color='Gender',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#95A5A6']
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
            
            # Credit completion by gender
            st.subheader("📊 Credit Completion by Gender")
            if "Credits_Earned" in filtered.columns:
                credit_by_gender = filtered.groupby("Gender").apply(
                    lambda x: (x["Credits_Earned"] >= 20).sum()
                ).reset_index(name='Students with 20+ Credits')
                
                st.dataframe(credit_by_gender, use_container_width=True, hide_index=True)
        else:
            st.info("No SGPA data available for performance analysis")

else:
    st.warning("Gender data is not available in the current dataset. Please ensure the CSV file contains a 'Name' column for gender detection.")

st.divider()

# =========================================================
# SEARCH RESULT
# =========================================================

if search:

    st.subheader("🔎 Search Result")

    if len(filtered) == 0:

        st.warning("No student found.")

    else:

        # we can hide any column we want for privacy or any other reason in if col = "Name"  
        display_cols = [col for col in filtered.columns if col ]
        st.dataframe(
            filtered[display_cols],
            use_container_width=True,
            hide_index=True
        )

        st.divider()


# =========================================================
# SGPA DISTRIBUTION
# =========================================================

st.subheader("📊 SGPA Distribution")

if filtered["SGPA"].notna().any():

    fig = px.histogram(
        filtered,
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

else:

    st.info("No SGPA data available.")


# =========================================================
# SUBJECT-WISE GRADE ANALYSIS
# =========================================================

st.subheader("📚 Subject-wise Grade Analysis")

subject_columns = [
    col for col in [
        "BSC_101",
        "ESC_101",
        "BSC_102",
        "ESC_102",
        "BSC_103",
        "ESC_103",
        "HSMC_103"
    ]
    if col in filtered.columns
]

if subject_columns:

    grade_data = []

    for subject in subject_columns:

        counts = filtered[subject].fillna(
            "Not Available"
        ).value_counts()

        for grade, count in counts.items():

            grade_data.append({
                "Subject": subject,
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

else:

    st.info("Subject grade columns not available.")


# =========================================================
# TOP PERFORMERS
# =========================================================

st.subheader("🏆 Top Performers")

top_students = (
    filtered
    .sort_values(
        "SGPA",
        ascending=False
    )
    .head(10)
)

display_columns = [
    col for col in [
        "Roll_No",
        "Name",
        "Gender",
        "SGPA",
        "Credits_Earned"
    ]
    if col in top_students.columns
]

st.dataframe(
    top_students[display_columns],
    use_container_width=True,
    hide_index=True
)


# =========================================================
# ACADEMIC STATUS
# =========================================================

st.subheader("📈 Academic Status")

status_df = pd.DataFrame({
    "Status": [
        "20 Credits",
        "Less than 20 Credits"
    ],
    "Students": [
        (filtered["Credits_Earned"] >= 20).sum(),
        (filtered["Credits_Earned"] < 20).sum()
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


# =========================================================
# COMPLETE STUDENT RESULTS
# =========================================================

st.subheader("📋 Complete Student Results")

# Hide Name column for privacy
display_cols = [col for col in filtered.columns if col != 'Name']
if display_cols:
    # Ensure Gender is displayed near the front
    if 'Gender' in display_cols:
        display_cols.remove('Gender')
        if 'Roll_No' in display_cols:
            roll_idx = display_cols.index('Roll_No')
            display_cols.insert(roll_idx + 1, 'Gender')
        else:
            display_cols.insert(1, 'Gender')
    
    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# DOWNLOAD SECTION
# =========================================================

if uploaded_file is not None:
    st.divider()
    st.subheader("📥 Download Processed CSV")
    
    # Complete CSV
    csv_buffer = io.StringIO()
    filtered.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download CSV with Gender Column",
            data=csv_string,
            file_name="students_with_gender.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download Male-only CSV
        male_df = filtered[filtered['Gender'] == 'Male']
        if not male_df.empty:
            male_csv = male_df.to_csv(index=False)
            st.download_button(
                label="👨 Download Male Students Only",
                data=male_csv,
                file_name="male_students.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Download Female-only CSV
        female_df = filtered[filtered['Gender'] == 'Female']
        if not female_df.empty:
            female_csv = female_df.to_csv(index=False)
            st.download_button(
                label="👩 Download Female Students Only",
                data=female_csv,
                file_name="female_students.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col4:
        # Download Unknown-only CSV
        unknown_df = filtered[filtered['Gender'] == 'Unknown']
        if not unknown_df.empty:
            unknown_csv = unknown_df.to_csv(index=False)
            st.download_button(
                label="❓ Download Unknown Gender Students",
                data=unknown_csv,
                file_name="unknown_gender_students.csv",
                mime="text/csv",
                use_container_width=True
            )

# =========================================================
# GENDER VS ACADEMIC PERFORMANCE - ADDITIONAL ANALYSIS
# =========================================================

st.divider()
st.subheader("📊 Gender vs Academic Performance Analysis")

if "Gender" in filtered.columns and "SGPA" in filtered.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot: SGPA vs Gender
        fig_scatter = px.strip(
            filtered[filtered['Gender'] != 'Unknown'],
            x='Gender',
            y='SGPA',
            title="SGPA Distribution by Gender",
            color='Gender',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
            stripmode='overlay'
        )
        fig_scatter.update_layout(
            xaxis_title="Gender",
            yaxis_title="SGPA"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Violin plot for SGPA by gender
        fig_violin = px.violin(
            filtered[filtered['Gender'] != 'Unknown'],
            x='Gender',
            y='SGPA',
            title="SGPA Distribution (Violin Plot)",
            color='Gender',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4'],
            box=True,
            points='all'
        )
        st.plotly_chart(fig_violin, use_container_width=True)

# =========================================================
# GENDER-WISE ACADEMIC PERFORMANCE TABLE
# =========================================================

if "Gender" in filtered.columns:
    st.subheader("📊 Gender-wise Academic Performance Summary")
    
    gender_stats = filtered.groupby('Gender').agg({
        'Roll_No': 'count',
        'SGPA': ['mean', 'min', 'max', 'std'],
        'Credits_Earned': ['mean', 'min', 'max']
    }).round(2)
    
    # Rename columns for better display
    gender_stats.columns = ['Count', 'Avg SGPA', 'Min SGPA', 'Max SGPA', 'Std SGPA', 
                           'Avg Credits', 'Min Credits', 'Max Credits']
    
    st.dataframe(gender_stats, use_container_width=True)
    
    # Show count of students with 20+ credits by gender
    if "Credits_Earned" in filtered.columns:
        credit_completion = filtered.groupby('Gender').apply(
            lambda x: (x['Credits_Earned'] >= 20).sum()
        ).reset_index(name='Students with 20+ Credits')
        
        st.subheader("📊 Credit Completion by Gender")
        st.dataframe(credit_completion, use_container_width=True, hide_index=True)

#

#=========================================================
# =========================================================
# =========================================================
# OVERALL CLASS PERFORMANCE - PASS/FAIL ANALYSIS (FIXED)
# =========================================================

st.divider()
st.subheader("📊 Overall Class Performance - Pass/Fail Analysis")

# Define passing criteria (SGPA >= 5.0 is typically passing)
PASSING_SGPA = 5.0

if "SGPA" in filtered.columns:
    
    # FIXED: Treat Unknown/Null SGPA as Fail
    def get_pass_status(sgpa):
        if pd.isna(sgpa):
            return "❌ Fail"  # Treat missing SGPA as Fail
        elif sgpa >= PASSING_SGPA:
            return "✅ Pass"
        else:
            return "❌ Fail"  # SGPA < 5.0
    
    filtered["Pass_Status"] = filtered["SGPA"].apply(get_pass_status)
    
    # Get counts
    pass_count = (filtered["Pass_Status"] == "✅ Pass").sum()
    fail_count = (filtered["Pass_Status"] == "❌ Fail").sum()
    total_students = len(filtered)
    
    # Calculate rates
    pass_rate = (pass_count / total_students * 100) if total_students > 0 else 0
    fail_rate = (fail_count / total_students * 100) if total_students > 0 else 0
    
    # Create tabs for different views
    perf_tab1, perf_tab2, perf_tab3, perf_tab4 = st.tabs([
        "📊 Overview", 
        "📈 Charts", 
        "📋 Detailed Analysis",
        "🎯 Performance by Gender"
    ])
    
    with perf_tab1:
        # KPI Cards for Pass/Fail
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "✅ Pass Rate",
                f"{pass_rate:.1f}%",
                f"{pass_count} students"
            )
        
        with col2:
            st.metric(
                "❌ Fail Rate",
                f"{fail_rate:.1f}%",
                f"{fail_count} students"
            )
        
        with col3:
            st.metric(
                "📊 Total Students",
                total_students
            )
        
        with col4:
            st.metric(
                "📈 Pass:Fail Ratio",
                f"{pass_count}:{fail_count}"
            )
        
        # Pass/Fail Summary
        st.info(f"""
        **📊 Performance Summary:**
        - ✅ **Pass**: {pass_count} students ({pass_rate:.1f}%)
        - ❌ **Fail**: {fail_count} students ({fail_rate:.1f}%)
        - 📊 **Total**: {total_students} students
        - 📈 **Pass:Fail Ratio**: {pass_count}:{fail_count}
        - 🎯 **Minimum SGPA**: {filtered['SGPA'].min():.2f} if available else "N/A"
        - 🏆 **Maximum SGPA**: {filtered['SGPA'].max():.2f} if available else "N/A"
        - 📊 **Average SGPA**: {filtered['SGPA'].mean():.2f} if available else "N/A"
        """)
        
        # Show SGPA missing count
        missing_sgpa = filtered['SGPA'].isna().sum()
        if missing_sgpa > 0:
            st.warning(f"⚠️ {missing_sgpa} students have missing SGPA and are counted as Fail")
    
    with perf_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie Chart for Pass/Fail
            status_data = []
            if pass_count > 0:
                status_data.append({"Status": "✅ Pass", "Count": pass_count})
            if fail_count > 0:
                status_data.append({"Status": "❌ Fail", "Count": fail_count})
            
            if status_data:
                status_df = pd.DataFrame(status_data)
                fig_pass_fail = px.pie(
                    status_df,
                    values="Count",
                    names="Status",
                    title=f"Pass/Fail Distribution (Total: {total_students} students)",
                    color_discrete_sequence=['#2ECC71', '#E74C3C'],
                    hole=0.3
                )
                fig_pass_fail.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont_size=14
                )
                st.plotly_chart(fig_pass_fail, use_container_width=True)
        
        with col2:
            # Bar Chart for Pass/Fail
            if status_data:
                fig_bar_status = px.bar(
                    status_df,
                    x="Status",
                    y="Count",
                    title="Pass/Fail Distribution",
                    labels={'x': 'Status', 'y': 'Number of Students'},
                    color="Status",
                    color_discrete_sequence=['#2ECC71', '#E74C3C'],
                    text="Count"
                )
                fig_bar_status.update_traces(textposition='outside')
                fig_bar_status.update_layout(showlegend=False)
                st.plotly_chart(fig_bar_status, use_container_width=True)
        
        # SGPA Distribution with Pass/Fail threshold
        st.subheader("📊 SGPA Distribution with Pass/Fail Threshold")
        
        # Only show students with SGPA values
        sgpa_data = filtered[filtered["SGPA"].notna()]
        if not sgpa_data.empty:
            fig_sgpa_threshold = px.histogram(
                sgpa_data,
                x="SGPA",
                nbins=20,
                title=f"SGPA Distribution (Passing threshold: {PASSING_SGPA})",
                color="Pass_Status",
                color_discrete_map={'✅ Pass': '#2ECC71', '❌ Fail': '#E74C3C'}
            )
            # Add vertical line for passing threshold
            fig_sgpa_threshold.add_vline(
                x=PASSING_SGPA, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Passing Threshold: {PASSING_SGPA}",
                annotation_position="top right"
            )
            fig_sgpa_threshold.update_layout(
                xaxis_title="SGPA",
                yaxis_title="Number of Students"
            )
            st.plotly_chart(fig_sgpa_threshold, use_container_width=True)
        else:
            st.info("No SGPA data available for histogram")
    
    with perf_tab3:
        # Detailed Pass/Fail Analysis
        st.subheader("📋 Detailed Pass/Fail Analysis")
        
        # Create detailed dataframe (only students with SGPA)
        detail_df = filtered[filtered["SGPA"].notna()].copy()
        if not detail_df.empty:
            detail_df["SGPA_Range"] = pd.cut(
                detail_df["SGPA"],
                bins=[0, 3, 4, 5, 6, 7, 8, 9, 10],
                labels=["0-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"]
            )
            
            # SGPA distribution table
            sgpa_distribution = detail_df["SGPA_Range"].value_counts().sort_index().reset_index()
            sgpa_distribution.columns = ["SGPA Range", "Number of Students"]
            sgpa_distribution["Percentage"] = (sgpa_distribution["Number of Students"] / len(detail_df) * 100).round(1)
            
            st.subheader("📊 SGPA Distribution by Range")
            st.dataframe(sgpa_distribution, use_container_width=True, hide_index=True)
        
        # Performance categories
        st.subheader("🏆 Performance Categories")
        
        def get_performance_category(sgpa):
            if pd.isna(sgpa):
                return "❌ No SGPA Data"
            elif sgpa >= 8.5:
                return "🌟 Excellent"
            elif sgpa >= 7.5:
                return "⭐ Very Good"
            elif sgpa >= 6.5:
                return "✅ Good"
            elif sgpa >= 5.0:
                return "📘 Satisfactory"
            else:
                return "❌ Needs Improvement"
        
        filtered["Performance_Category"] = filtered["SGPA"].apply(get_performance_category)
        category_counts = filtered["Performance_Category"].value_counts()
        
        # Display category distribution
        category_df = pd.DataFrame({
            "Category": category_counts.index,
            "Count": category_counts.values,
            "Percentage": (category_counts.values / total_students * 100).round(1)
        })
        
        st.dataframe(category_df, use_container_width=True, hide_index=True)
        
        # Bar chart for performance categories
        fig_categories = px.bar(
            category_df,
            x="Category",
            y="Count",
            title="Student Performance Distribution",
            labels={'x': 'Performance Category', 'y': 'Number of Students'},
            color="Category",
            text="Count"
        )
        fig_categories.update_traces(textposition='outside')
        fig_categories.update_layout(showlegend=False)
        st.plotly_chart(fig_categories, use_container_width=True)
    
    with perf_tab4:
        # Performance by Gender
        st.subheader("📊 Pass/Fail by Gender")
        
        if "Gender" in filtered.columns:
            # Create pass/fail data by gender
            gender_performance = filtered.groupby("Gender").agg({
                "SGPA": ["count", "mean", "min", "max"],
                "Pass_Status": lambda x: (x == "✅ Pass").sum()
            }).round(2)
            
            # Flatten columns
            gender_performance.columns = ['Count', 'Avg SGPA', 'Min SGPA', 'Max SGPA', 'Pass Count']
            
            # Calculate fail count and rates
            gender_performance["Fail Count"] = gender_performance["Count"] - gender_performance["Pass Count"]
            gender_performance["Pass Rate"] = (gender_performance["Pass Count"] / gender_performance["Count"] * 100).round(1)
            gender_performance["Fail Rate"] = (gender_performance["Fail Count"] / gender_performance["Count"] * 100).round(1)
            
            st.dataframe(gender_performance, use_container_width=True)
            
            # Visualize pass/fail by gender
            col1, col2 = st.columns(2)
            
            with col1:
                # Pass rate by gender
                fig_pass_rate = px.bar(
                    gender_performance.reset_index(),
                    x="Gender",
                    y="Pass Rate",
                    title="Pass Rate by Gender",
                    labels={'x': 'Gender', 'y': 'Pass Rate (%)'},
                    color="Gender",
                    color_discrete_sequence=['#2ECC71', '#3498DB'],
                    text="Pass Rate"
                )
                fig_pass_rate.update_traces(textposition='outside')
                fig_pass_rate.update_layout(showlegend=False)
                st.plotly_chart(fig_pass_rate, use_container_width=True)
            
            with col2:
                # Stacked bar for Pass/Fail by Gender
                gender_stacked = gender_performance.reset_index()
                
                # Prepare data for stacked bar chart
                stacked_data = []
                for _, row in gender_stacked.iterrows():
                    stacked_data.append({
                        'Gender': row['Gender'],
                        'Status': 'Pass',
                        'Count': row['Pass Count']
                    })
                    stacked_data.append({
                        'Gender': row['Gender'],
                        'Status': 'Fail',
                        'Count': row['Fail Count']
                    })
                
                gender_stacked_melted = pd.DataFrame(stacked_data)
                
                fig_stacked = px.bar(
                    gender_stacked_melted,
                    x="Gender",
                    y="Count",
                    color="Status",
                    title="Pass/Fail Distribution by Gender",
                    color_discrete_map={'Pass': '#2ECC71', 'Fail': '#E74C3C'},
                    barmode="stack",
                    text="Count"
                )
                fig_stacked.update_traces(textposition='inside')
                st.plotly_chart(fig_stacked, use_container_width=True)
            
            # Average SGPA by gender with pass/fail
            st.subheader("📊 Average SGPA by Gender")
            
            # Box plot with pass/fail
            sgpa_data = filtered[filtered["SGPA"].notna()]
            if not sgpa_data.empty:
                fig_box_gender = px.box(
                    sgpa_data,
                    x="Gender",
                    y="SGPA",
                    title="SGPA Distribution by Gender",
                    color="Gender",
                    color_discrete_sequence=['#2ECC71', '#3498DB']
                )
                # Add horizontal line for passing threshold
                fig_box_gender.add_hline(
                    y=PASSING_SGPA, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Passing Threshold: {PASSING_SGPA}",
                    annotation_position="bottom right"
                )
                st.plotly_chart(fig_box_gender, use_container_width=True)

else:
    st.warning("No SGPA data available for performance analysis")

# =========================================================
# PASS/FAIL SUMMARY TABLE
# =========================================================

if "SGPA" in filtered.columns:
    st.subheader("📋 Pass/Fail Summary Table")
    
    # Create summary table
    summary_data = {
        "Metric": ["Total Students", "Passed", "Failed", "Pass Rate", "Fail Rate"],
        "Value": [
            total_students,
            pass_count,
            fail_count,
            f"{pass_rate:.1f}%",
            f"{fail_rate:.1f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Show list of failed students (without names for privacy)
    if fail_count > 0:
        st.subheader(f"❌ Failed Students ({fail_count} students)")
        failed_df = filtered[filtered["Pass_Status"] == "❌ Fail"]
        # Remove Name and Pass_Status columns for privacy
        display_cols = [col for col in failed_df.columns if col not in ['Name', 'Pass_Status', 'Performance_Category']]
        st.dataframe(
            failed_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"Total failed students: {len(failed_df)}")

# =========================================================
# EXPORT PASS/FAIL DATA
# =========================================================

if "SGPA" in filtered.columns:
    st.divider()
    st.subheader("📥 Download Pass/Fail Reports")
    
    col1, col2, col3 = st.columns(3)
    
    # Prepare data for download
    complete_data = filtered.copy()
    if "Pass_Status" not in complete_data.columns:
        complete_data["Pass_Status"] = complete_data["SGPA"].apply(
            lambda x: "Pass" if pd.notna(x) and x >= PASSING_SGPA else "Fail"
        )
    
    with col1:
        # Complete data with pass/fail
        csv_complete = complete_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Data",
            data=csv_complete,
            file_name="complete_student_performance.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Failed students only
        failed_only = complete_data[complete_data["Pass_Status"] == "Fail"]
        if not failed_only.empty:
            csv_failed = failed_only.to_csv(index=False)
            st.download_button(
                label="❌ Download Failed Students",
                data=csv_failed,
                file_name="failed_students.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No failed students")
    
    with col3:
        # Passed students only
        passed_only = complete_data[complete_data["Pass_Status"] == "Pass"]
        if not passed_only.empty:
            csv_passed = passed_only.to_csv(index=False)
            st.download_button(
                label="✅ Download Passed Students",
                data=csv_passed,
                file_name="passed_students.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No passed students")
#=========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    f"Student Academic Performance Analyzer | "
    f"{branch} | {semester}"
)

# =========================================================
# HOW IT WORKS EXPANDER
# =========================================================

with st.expander("ℹ️ About Gender Detection & Analysis"):
    st.markdown("""
    ### Automatic Gender Detection & Academic Analysis System
    
    **What this app does:**
    1. **Gender Detection**: Automatically detects gender from student names
    2. **Gender Analysis**: Shows gender distribution with charts and statistics
    3. **Academic Performance**: Compares academic performance by gender
    4. **Data Export**: Download gender-separated CSV files
    
    **Features:**
    - 📊 **Gender Distribution**: Pie and bar charts showing gender breakdown
    - 📈 **Statistics**: Male/Female counts, percentages, and ratios
    - 🎯 **Academic Performance**: SGPA comparison by gender
    - 📋 **Gender-Separated Data**: View male and female students separately
    - 📥 **Download Options**: Download CSVs with gender column
    
    **Detection Methods:**
    - **Name Database**: Built-in database of common Indian names
    - **Suffix Detection**: 'Singh', 'Kumar' → Male | 'Kaur' → Female
    - **Pattern Recognition**: Names ending with 'a', 'i', 'e' → Often Female
    - **Machine Learning**: Trained model for accurate detection
    
    **Privacy Protected:**
    - Student names are NEVER displayed
    - Only statistical data and charts are shown
    - All processing happens locally on your machine
    """)