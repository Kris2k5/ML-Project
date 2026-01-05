"""
Resume Screening Application - Streamlit UI
============================================

Interactive web application for AI-powered resume screening.
Upload resumes, enter job description, and get ranked candidates with ML matching.
"""

import streamlit as st
import os
import tempfile
from datetime import datetime
import pandas as pd
from resume_screener import ResumeScreener


# Page configuration
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .candidate-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .score-excellent {
        color: #28a745;
        font-weight: bold;
    }
    .score-good {
        color: #ffc107;
        font-weight: bold;
    }
    .score-fair {
        color: #ff6b6b;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


def get_score_color_class(score):
    """Return CSS class based on score value."""
    if score >= 70:
        return "score-excellent"
    elif score >= 50:
        return "score-good"
    else:
        return "score-fair"


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory and return path."""
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file {uploaded_file.name}: {str(e)}")
        return None


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📄 AI-Powered Resume Screener</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload resumes and find the best candidates using Machine Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        This application uses **Machine Learning** to match resumes with job descriptions.
        
        **How it works:**
        1. Upload candidate resumes (PDF/TXT)
        2. Enter job description
        3. Click 'Analyze Candidates'
        4. View ranked results
        
        **ML Approach:**
        - TF-IDF vectorization
        - Cosine similarity matching
        - Automated keyword extraction
        """)
        
        st.header("📊 Features")
        st.markdown("""
        ✅ Multi-file upload (PDF/TXT)  
        ✅ Real-time ML analysis  
        ✅ Candidate ranking  
        ✅ Skills matching  
        ✅ Export results to CSV  
        """)
    
    # Main content area
    st.header("1️⃣ Upload Resumes")
    st.markdown("Upload one or more resume files (PDF or TXT format)")
    
    uploaded_files = st.file_uploader(
        "Choose resume files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Select multiple resume files to analyze"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        with st.expander("📋 View uploaded files"):
            for idx, file in enumerate(uploaded_files, 1):
                st.write(f"{idx}. {file.name} ({file.size} bytes)")
    
    st.markdown("---")
    
    # Job description input
    st.header("2️⃣ Enter Job Description")
    st.markdown("Provide a detailed job description including required skills and qualifications")
    
    job_description = st.text_area(
        "Job Description",
        height=200,
        placeholder="""Example:
We are looking for a Senior Software Engineer with experience in Python and Machine Learning.

Required Skills:
- 5+ years of Python development
- Experience with ML frameworks (TensorFlow, PyTorch, scikit-learn)
- Strong understanding of data structures and algorithms
- Experience with REST APIs and web services

Preferred Qualifications:
- Master's degree in Computer Science
- Experience with cloud platforms (AWS, GCP, Azure)
- Strong communication skills
        """,
        help="Enter the job requirements, skills, and qualifications"
    )
    
    if job_description:
        st.info(f"📝 Job description: {len(job_description)} characters")
    
    st.markdown("---")
    
    # Analysis button
    st.header("3️⃣ Analyze Candidates")
    
    if st.button("🚀 Analyze Candidates", type="primary", use_container_width=True):
        
        # Validation
        if not uploaded_files:
            st.error("⚠️ Please upload at least one resume file!")
            return
        
        if not job_description or len(job_description.strip()) < 50:
            st.error("⚠️ Please enter a detailed job description (at least 50 characters)!")
            return
        
        # Process files
        with st.spinner("🔄 Processing resumes... This may take a moment."):
            try:
                # Save uploaded files to temporary location
                resume_files = []
                for uploaded_file in uploaded_files:
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        resume_files.append((uploaded_file.name, file_path))
                
                if not resume_files:
                    st.error("❌ No files could be processed. Please try again.")
                    return
                
                # Initialize screener
                screener = ResumeScreener()
                
                # Run analysis
                results_df = screener.analyze_resumes(job_description, resume_files)
                
                # Store results in session state
                st.session_state['results'] = results_df
                st.session_state['analysis_complete'] = True
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                return
    
    # Display results
    if 'analysis_complete' in st.session_state and st.session_state['analysis_complete']:
        st.markdown("---")
        st.header("4️⃣ Results Dashboard")
        
        results_df = st.session_state['results']
        
        if results_df.empty:
            st.warning("No results to display.")
            return
        
        # Summary metrics
        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Candidates", len(results_df))
        
        with col2:
            top_score = results_df['Match_Score'].max()
            st.metric("Top Match Score", f"{top_score:.1f}%")
        
        with col3:
            avg_score = results_df['Match_Score'].mean()
            st.metric("Average Score", f"{avg_score:.1f}%")
        
        with col4:
            high_matches = len(results_df[results_df['Match_Score'] >= 70])
            st.metric("Strong Matches (≥70%)", high_matches)
        
        st.markdown("---")
        
        # Top candidates section
        st.subheader("🏆 Top Candidates Ranking")
        
        # Show top 10 or all if less than 10
        display_count = min(10, len(results_df))
        
        for idx, row in results_df.head(display_count).iterrows():
            rank = row['Rank']
            name = row['Candidate_Name']
            score = row['Match_Score']
            keywords = row['Matched_Keywords']
            matched_count = row['Matched_Count']
            
            # Color coding based on score
            score_class = get_score_color_class(score)
            
            # Candidate card
            with st.container():
                col_rank, col_details = st.columns([1, 9])
                
                with col_rank:
                    # Medal icons for top 3
                    if rank == 1:
                        st.markdown("### 🥇")
                    elif rank == 2:
                        st.markdown("### 🥈")
                    elif rank == 3:
                        st.markdown("### 🥉")
                    else:
                        st.markdown(f"### #{rank}")
                
                with col_details:
                    st.markdown(f"**{name}**")
                    
                    # Progress bar for score
                    st.progress(score / 100)
                    
                    col_score, col_keywords = st.columns([1, 2])
                    with col_score:
                        st.markdown(f'<p class="{score_class}">Match Score: {score:.1f}%</p>', unsafe_allow_html=True)
                    with col_keywords:
                        st.markdown(f"**Keywords Matched:** {matched_count}")
                    
                    if keywords:
                        st.markdown(f"**🔑 Top Matched Skills:** {keywords}")
                
                st.markdown("---")
        
        # Full results table
        if len(results_df) > display_count:
            with st.expander(f"📋 View All {len(results_df)} Candidates"):
                st.dataframe(
                    results_df[['Rank', 'Candidate_Name', 'Match_Score', 'Matched_Count']],
                    use_container_width=True,
                    hide_index=True
                )
        
        # Export functionality
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        col_export1, col_export2 = st.columns([2, 1])
        
        with col_export1:
            # Prepare CSV
            export_df = results_df[['Rank', 'Candidate_Name', 'Match_Score', 'Matched_Keywords', 'Matched_Count']]
            csv = export_df.to_csv(index=False)
            
            # Timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resume_screening_results_{timestamp}.csv"
            
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
        
        with col_export2:
            if st.button("🔄 Start New Analysis", use_container_width=True):
                # Clear session state
                st.session_state['analysis_complete'] = False
                st.session_state['results'] = None
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #888; padding: 2rem;'>
            <p>Built with ❤️ using Streamlit and scikit-learn</p>
            <p>ML-Powered Resume Screening System | Version 1.0</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
