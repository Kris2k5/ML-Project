"""
Resume Screener - ML Engine for Resume Analysis
================================================

This module implements the core machine learning engine for resume screening.
It uses NLP techniques (TF-IDF and cosine similarity) to match resumes against
job descriptions and rank candidates.

Key Components:
    - Text extraction from PDF and TXT files
    - Text preprocessing (cleaning, tokenization, stopword removal)
    - TF-IDF vectorization for feature extraction
    - Cosine similarity for matching
    - Candidate ranking algorithm
"""

import os
import re
from typing import List, Dict, Tuple
from collections import Counter
import itertools
import PyPDF2
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class ResumeScreener:
    """
    Main class for resume screening using machine learning.
    
    This class handles the entire pipeline from text extraction to candidate ranking.
    """
    
    def __init__(self):
        """Initialize the resume screener with necessary NLP resources."""
        self._download_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        self.job_vector = None
        
    def _download_nltk_resources(self):
        """Download required NLTK data if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
            
        Raises:
            Exception: If PDF cannot be read
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """
        Extract text content from a TXT file.
        
        Args:
            txt_path (str): Path to the TXT file
            
        Returns:
            str: File content
            
        Raises:
            Exception: If file cannot be read
        """
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error reading TXT {txt_path}: {str(e)}")
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a file (auto-detects PDF or TXT based on extension).
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str: Extracted text content
        """
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.txt'):
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for ML analysis.
        
        Steps:
            1. Convert to lowercase
            2. Remove special characters and digits
            3. Remove extra whitespace
            4. Tokenize
            5. Remove stopwords
            6. Rejoin tokens
        
        Args:
            text (str): Raw text input
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words (< 2 characters)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Rejoin tokens
        return ' '.join(tokens)
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract top keywords from text based on frequency.
        
        Args:
            text (str): Preprocessed text
            top_n (int): Number of top keywords to extract
            
        Returns:
            List[str]: List of top keywords
        """
        tokens = text.split()
        
        # Count word frequencies using Counter for better performance
        word_freq = Counter(tokens)
        
        # Get top N most common words
        return [word for word, freq in word_freq.most_common(top_n)]
    
    def analyze_resumes(self, job_description: str, resume_files: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Analyze resumes against a job description and rank candidates.
        
        This is the main analysis function that:
            1. Preprocesses all texts
            2. Creates TF-IDF vectors
            3. Computes cosine similarity scores
            4. Ranks candidates
            5. Extracts matching keywords
        
        Args:
            job_description (str): The job description text
            resume_files (List[Tuple[str, str]]): List of tuples (filename, file_path)
            
        Returns:
            pd.DataFrame: Results dataframe with columns:
                - Rank
                - Candidate_Name
                - Match_Score
                - Matched_Keywords
                - File_Path
        """
        if not resume_files:
            return pd.DataFrame()
        
        # Step 1: Extract and preprocess job description
        print("Preprocessing job description...")
        job_desc_processed = self.preprocess_text(job_description)
        job_keywords = set(self.extract_keywords(job_desc_processed, top_n=50))
        
        # Step 2: Extract and preprocess all resumes
        print(f"Processing {len(resume_files)} resumes...")
        resume_data = []
        
        for filename, file_path in resume_files:
            try:
                # Extract text
                raw_text = self.extract_text(file_path)
                # Preprocess
                processed_text = self.preprocess_text(raw_text)
                resume_data.append({
                    'filename': filename,
                    'file_path': file_path,
                    'processed_text': processed_text
                })
            except Exception as e:
                print(f"Warning: Could not process {filename}: {str(e)}")
                continue
        
        if not resume_data:
            print("No resumes could be processed successfully.")
            return pd.DataFrame()
        
        # Step 3: Create TF-IDF vectors
        print("Computing TF-IDF vectors...")
        
        # Prepare documents: job description + all resumes
        documents = [job_desc_processed] + [r['processed_text'] for r in resume_data]
        
        # Create TF-IDF vectorizer
        # Using 1-2 word phrases (unigrams and bigrams) for better matching
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Job description vector is the first one
        self.job_vector = tfidf_matrix[0:1]
        
        # Resume vectors are the rest
        resume_vectors = tfidf_matrix[1:]
        
        # Step 4: Compute cosine similarity
        print("Computing similarity scores...")
        similarities = cosine_similarity(self.job_vector, resume_vectors)[0]
        
        # Step 5: Create results dataframe
        results = []
        for idx, resume in enumerate(resume_data):
            # Get similarity score as percentage
            score = round(similarities[idx] * 100, 2)
            
            # Extract keywords from this resume
            resume_keywords = set(self.extract_keywords(resume['processed_text'], top_n=50))
            
            # Find matching keywords with job description
            matched_keywords = job_keywords.intersection(resume_keywords)
            # Use itertools.islice for efficient slicing of sorted keywords
            matched_keywords_str = ', '.join(itertools.islice(sorted(matched_keywords), 10))
            
            results.append({
                'Candidate_Name': resume['filename'],
                'Match_Score': score,
                'Matched_Keywords': matched_keywords_str,
                'Matched_Count': len(matched_keywords),
                'File_Path': resume['file_path']
            })
        
        # Create dataframe and sort by score
        df = pd.DataFrame(results)
        df = df.sort_values('Match_Score', ascending=False).reset_index(drop=True)
        df['Rank'] = df.index + 1
        
        # Reorder columns
        df = df[['Rank', 'Candidate_Name', 'Match_Score', 'Matched_Keywords', 'Matched_Count', 'File_Path']]
        
        print(f"Analysis complete! Processed {len(results)} candidates.")
        return df
    
    def get_top_candidates(self, results_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N candidates from results.
        
        Args:
            results_df (pd.DataFrame): Results dataframe from analyze_resumes
            top_n (int): Number of top candidates to return
            
        Returns:
            pd.DataFrame: Top N candidates
        """
        return results_df.head(top_n)
    
    def export_results(self, results_df: pd.DataFrame, output_path: str):
        """
        Export results to CSV file.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            output_path (str): Path to save CSV file
        """
        # Select columns for export (exclude File_Path for privacy)
        export_df = results_df[['Rank', 'Candidate_Name', 'Match_Score', 'Matched_Keywords', 'Matched_Count']]
        export_df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")


# Example usage (for testing)
if __name__ == "__main__":
    # Create screener instance
    screener = ResumeScreener()
    
    # Example job description
    job_desc = """
    We are looking for a Senior Python Developer with experience in machine learning.
    Required skills: Python, TensorFlow, scikit-learn, pandas, NumPy.
    The candidate should have strong background in data science and AI.
    Experience with web development (Django/Flask) is a plus.
    """
    
    # Example resume files (would need actual files)
    # resume_files = [
    #     ('john_doe.pdf', 'path/to/john_doe.pdf'),
    #     ('jane_smith.txt', 'path/to/jane_smith.txt')
    # ]
    
    # Analyze
    # results = screener.analyze_resumes(job_desc, resume_files)
    # print(results)
    
    print("ResumeScreener module loaded successfully!")
