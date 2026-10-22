import streamlit as st
import pandas as pd
import plotly.express as px
import os
import io

from analyzer.catalog import apply_majority_batch_year, collapse_exam_attempts, infer_exam_type, sidebar_branch_year_sem_picker
from analyzer.nav import BRANCH_VIEWS, selected_view
from analyzer.parsers import detect_subject_columns, load_dataset_smart
from analyzer.store import DATASETS, _data_mtime_signature, build_dataset_catalog, load_data, process_csv_with_gender
from analyzer.ui import kpi_cards, show_chart

def render_branch_analysis_page():
    """One branch and semester, with Regular, Reappear, and Makeup together."""
    st.sidebar.subheader("Dataset Selection")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or PDF (optional — overrides selection below)",
        type=['csv', 'pdf'],
        help="Upload a result CSV or PDF; format, subjects, and gender are auto-detected"
    )
    subject_columns = []
    dataset_label = "No dataset"
    df = pd.DataFrame()

    if uploaded_file is not None:
        try:
            with st.spinner("Parsing file..."):
                df, subject_columns = load_dataset_smart(uploaded_file, uploaded_file.name)
                df = process_csv_with_gender(df)
            dataset_label = uploaded_file.name
            if df.empty:
                st.sidebar.error("Could not extract student rows from this file. Check format.")
            else:
                st.sidebar.success(f"✅ Loaded: {len(df)} students from upload")
                if uploaded_file.name.lower().endswith(".pdf"):
                    st.sidebar.caption("Parsed from PDF")
                if subject_columns:
                    st.sidebar.caption(f"📚 Auto-detected {len(subject_columns)} subject columns")
                if "Gender" in df.columns:
                    gender_counts = df["Gender"].value_counts()
                    st.sidebar.info(
                        f"**Gender Distribution:**\n"
                        f"- 👨 Male: {gender_counts.get('Male', 0)}\n"
                        f"- 👩 Female: {gender_counts.get('Female', 0)}\n"
                        f"- ❓ Unknown: {gender_counts.get('Unknown', 0)}"
                    )
        except Exception as e:
            st.sidebar.error(f"Error loading upload: {str(e)}")
    elif DATASETS:
        catalog = build_dataset_catalog(_data_mtime_signature())
        pick = sidebar_branch_year_sem_picker(catalog, key_prefix="branch_analysis")
        if pick is None:
            st.sidebar.warning("Choose Branch, Year, and Semester to load a dataset.")
        else:
            dataset_label = pick["display"]
            matches = pick.get("matches") or [pick]
            try:
                frames = []
                subject_columns = []
                with st.spinner("Loading all exam types for this semester…"):
                    for item in matches:
                        file_path = item["path"]
                        frame, subs = load_data(file_path, os.path.getmtime(file_path))
                        if frame is None or frame.empty:
                            continue
                        tagged = frame.copy()
                        tagged["Branch"] = item.get("branch", pick.get("branch"))
                        tagged["Semester"] = item.get("semester", pick.get("semester"))
                        tagged["Semester_Label"] = item.get("semester_label")
                        tagged["Exam_Type"] = item.get("exam_type") or infer_exam_type(item.get("filename", ""))
                        tagged["Batch_Year_Label"] = item.get("batch_year_label")
                        tagged["Study_Year_Label"] = item.get("study_year_label")
                        tagged = apply_majority_batch_year(tagged)
                        frames.append(tagged)
                        subject_columns.extend(subs or [])
                if not frames:
                    st.sidebar.error(f"Could not parse data for '{dataset_label}'. Check file format.")
                else:
                    df = collapse_exam_attempts(pd.concat(frames, ignore_index=True, sort=False))
                    subject_columns = list(dict.fromkeys(c for c in subject_columns if c in df.columns))
                    types = ", ".join(pick.get("exam_types") or sorted(df["Exam_Type"].dropna().unique()))
                    st.sidebar.success(
                        f"Loaded: {len(df)} students · {pick.get('branch', '')} · "
                        f"Batch {pick.get('batch_year_label', '')} · {pick.get('study_year_label', '')} · "
                        f"Sem {pick.get('semester_label', '')} · {types}"
                    )
                    if subject_columns:
                        st.sidebar.caption(f"📚 Auto-detected {len(subject_columns)} subject columns")
                    if "Gender" in df.columns:
                        gender_counts = df["Gender"].value_counts()
                        st.sidebar.info(
                            f"**Gender Distribution:**\n"
                            f"- 👨 Male: {gender_counts.get('Male', 0)}\n"
                            f"- 👩 Female: {gender_counts.get('Female', 0)}\n"
                            f"- ❓ Unknown: {gender_counts.get('Unknown', 0)}"
                        )
            except Exception as e:
                st.sidebar.error(f"Error loading dataset: {str(e)}")
    else:
        st.sidebar.warning("No CSV/PDF files found in data folder. Upload a file to begin.")

    st.sidebar.divider()

    # =========================================================
    # STUDENT SEARCH
    # =========================================================

    st.sidebar.subheader("🔎 Student Search")

    search = st.sidebar.text_input(
        "Search student name or roll number",
        key="branch_student_search",
    )

    # =========================================================
    # TITLE
    # =========================================================

    view = selected_view(BRANCH_VIEWS)
    st.title(view)
    st.caption(f"Branch Analysis · {dataset_label}")

    if df.empty:
        st.warning("Please upload a CSV/PDF file or add files to the data folder to begin analysis.")
        st.stop()

    # =========================================================
    # SEARCH FILTER
    # =========================================================

    if search:
        name_match = pd.Series([False] * len(df), index=df.index)
        roll_match = pd.Series([False] * len(df), index=df.index)

        if "Name" in df.columns:
            name_match = df["Name"].astype(str).str.contains(search, case=False, na=False)
        if "Roll_No" in df.columns:
            roll_match = df["Roll_No"].astype(str).str.contains(search, case=False, na=False)

        filtered = df[name_match | roll_match]
    else:
        filtered = df.copy()


    # =========================================================
    # KPI DATA
    # =========================================================

    total_students = len(filtered)

    average_sgpa = filtered["SGPA"].mean() if "SGPA" in filtered.columns else None
    highest_sgpa = filtered["SGPA"].max() if "SGPA" in filtered.columns else None
    full_credits = (
        (filtered["Credits_Earned"] >= 20).sum()
        if "Credits_Earned" in filtered.columns else 0
    )


    # =========================================================
    # KPI CARDS
    # =========================================================

    kpi_cards([
        ("Total Students", total_students),
        ("Average SGPA", f"{average_sgpa:.2f}" if pd.notna(average_sgpa) else "N/A"),
        ("Highest SGPA", f"{highest_sgpa:.2f}" if pd.notna(highest_sgpa) else "N/A"),
        ("20 Credits Earned", full_credits),
    ])

    st.divider()

    if view == "Gender Ratio":

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
                    show_chart(fig_gender_pie, use_container_width=True)
        
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
                    show_chart(fig_gender_bar, use_container_width=True)
    
            with tab2:
                male_percentage = (male_count / len(filtered) * 100) if len(filtered) > 0 else 0
                female_percentage = (female_count / len(filtered) * 100) if len(filtered) > 0 else 0
                ratio = f"{male_count}:{female_count}" if female_count > 0 else f"{male_count}:0"
                kpi_cards([
                    ("Total Students", len(filtered)),
                    ("Male Students", f"{male_count} ({male_percentage:.1f}%)"),
                    ("Female Students", f"{female_count} ({female_percentage:.1f}%)"),
                    ("Male:Female Ratio", ratio),
                ])
        
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
                        display_cols = list(male_df.columns)
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
                        display_cols = list(female_df.columns)
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
                        display_cols = list(unknown_df.columns)
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
                            show_chart(fig_avg_sgpa, use_container_width=True)
            
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
                            show_chart(fig_box, use_container_width=True)
            
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


    if view == "SGPA Distribution":
        # =========================================================
        # SGPA DISTRIBUTION
        # =========================================================

        st.subheader("📊 SGPA Distribution")

        if filtered["SGPA"].notna().any() if "SGPA" in filtered.columns else False:

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

            show_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info("No SGPA data available.")


    # =========================================================
    if view == "Subject-wise Grades":
        # SUBJECT-WISE GRADE ANALYSIS
        # =========================================================

        st.subheader("📚 Subject-wise Grade Analysis")

        if not subject_columns:
            subject_columns = detect_subject_columns(filtered)

        if subject_columns:
            with st.expander("📋 Auto-detected subject columns", expanded=False):
                st.write(", ".join(subject_columns))

            grade_data = []
            for subject in subject_columns:
                counts = filtered[subject].fillna("Not Available").value_counts()
                for grade, count in counts.items():
                    grade_data.append({
                        "Subject": subject,
                        "Grade": str(grade),
                        "Students": int(count)
                    })

            grade_df = pd.DataFrame(grade_data)

            if grade_df.empty:
                st.info("No subject grades for the current search and filters.")
            else:
                fig2 = px.bar(
                    grade_df,
                    x="Subject",
                    y="Students",
                    color="Grade",
                    barmode="group",
                    title="Grade Distribution by Subject (Auto-detected)"
                )

                show_chart(fig2, use_container_width=True)
        else:
            st.info("No subject grade columns detected. Grades are auto-detected from column names and values.")


    # =========================================================
    if view == "Top Performers":
        # TOP PERFORMERS
        # =========================================================

        st.subheader("🏆 Top Performers")

        if "SGPA" in filtered.columns:
            top_students = filtered.sort_values("SGPA", ascending=False).head(10)
            display_columns = [
                col for col in ["Roll_No", "Name", "Gender", "SGPA", "Credits_Earned"]
                if col in top_students.columns
            ]
            st.dataframe(
                top_students[display_columns],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No SGPA column detected for ranking top performers.")


    # =========================================================
    if view == "Academic Status":
        # ACADEMIC STATUS
        # =========================================================

        st.subheader("📈 Academic Status")

        if "Credits_Earned" in filtered.columns:
            status_df = pd.DataFrame({
                "Status": ["20 Credits", "Less than 20 Credits"],
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

            show_chart(fig3, use_container_width=True)
        else:
            st.info("No credits column detected for academic status analysis.")


    # =========================================================
    if view == "Student Results":
        # COMPLETE STUDENT RESULTS
        # =========================================================

        st.subheader("📋 Complete Student Results")

        display_cols = list(filtered.columns)
        if display_cols:
            # Keep Roll_No, Name, then Gender together at the front of the results table
            for col in ("Name", "Gender"):
                if col in display_cols and "Roll_No" in display_cols:
                    display_cols.remove(col)
                    display_cols.insert(display_cols.index("Roll_No") + 1, col)
            if "Name" in display_cols and "Gender" in display_cols:
                display_cols.remove("Name")
                display_cols.insert(display_cols.index("Gender"), "Name")
    
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

    if view == "Pass / Fail":
        # =========================================================
        # OVERALL CLASS PERFORMANCE - PASS/FAIL ANALYSIS
        # (Gender academics live under Gender Ratio Analysis → Academic Performance)
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
            perf_tab1, perf_tab2, perf_tab3 = st.tabs([
                "📊 Overview",
                "📈 Charts",
                "📋 Detailed Analysis",
            ])
    
            with perf_tab1:
                kpi_cards([
                    ("Pass Rate", f"{pass_rate:.1f}%"),
                    ("Fail Rate", f"{fail_rate:.1f}%"),
                    ("Total Students", total_students),
                    ("Pass:Fail Ratio", f"{pass_count}:{fail_count}"),
                ])
        
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
                        show_chart(fig_pass_fail, use_container_width=True)
        
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
                        show_chart(fig_bar_status, use_container_width=True)
        
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
                    show_chart(fig_sgpa_threshold, use_container_width=True)
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
                show_chart(fig_categories, use_container_width=True)

        else:
            st.warning("No SGPA data available for performance analysis")

        # =========================================================
        # FAILED STUDENTS LIST
        # =========================================================

        if "SGPA" in filtered.columns and "Pass_Status" in filtered.columns:
            if fail_count > 0:
                st.subheader(f"❌ Failed Students ({fail_count} students)")
                failed_df = filtered[filtered["Pass_Status"] == "❌ Fail"]
                display_cols = [col for col in failed_df.columns if col not in ['Pass_Status', 'Performance_Category']]
                if "Name" in display_cols and "Roll_No" in display_cols:
                    display_cols.remove("Name")
                    display_cols.insert(display_cols.index("Roll_No") + 1, "Name")
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
        f"Student Academic Performance Analyzer | {dataset_label}"
    )

    # =========================================================
    # HOW IT WORKS EXPANDER
    # =========================================================

    with st.expander("ℹ️ About this app"):
        st.markdown("""
        Parses branch result CSVs/PDFs, detects subjects and gender from names, and surfaces
        SGPA, pass/fail, and subject pressure.

        **Pages:** Overall CGPA · At-Risk · Subject Heatmap · Section Comparison · Branch Analysis · College Comparison
        """)
