"""
Resume Screener - ML Engine for Resume Analysis
================================================

This module implements the core machine learning engine for resume screening.
It uses NLP techniques (TF-IDF and cosine similarity) to match resumes against
job descriptions and rank candidates.

**PYTHON 3.13 COMPATIBLE VERSION**
This implementation uses MANUAL TF-IDF and Cosine Similarity algorithms
built from scratch using pure Python and NumPy, avoiding scikit-learn
compilation issues with Python 3.13.

Key Components:
    - Text extraction from PDF and TXT files
    - Text preprocessing (cleaning, tokenization, stopword removal)
    - **MANUAL TF-IDF vectorization** for feature extraction (from scratch)
    - **MANUAL Cosine similarity** for matching (from scratch using NumPy)
    - Candidate ranking algorithm
    
Educational Value:
    All ML algorithms are implemented from scratch with detailed comments
    explaining the mathematics, making this perfect for learning and presentations.
"""

import os
import re
import math
from typing import List, Dict, Tuple
from collections import Counter
import itertools
import PyPDF2
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class ResumeScreener:
    """
    Main class for resume screening using machine learning.
    
    This class handles the entire pipeline from text extraction to candidate ranking.
    Uses MANUAL implementation of TF-IDF and cosine similarity (no scikit-learn).
    """
    
    def __init__(self):
        """Initialize the resume screener with necessary NLP resources."""
        self._download_nltk_resources()
        self.stop_words = set(stopwords.words('english'))
        self.vocabulary = []  # List of unique terms in corpus
        self.idf_dict = {}    # IDF values for each term
        
    def _download_nltk_resources(self):
        """Download required NLTK data if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
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
    
    def build_vocabulary(self, documents: List[str]) -> List[str]:
        """
        Build vocabulary from all documents.
        
        Args:
            documents (List[str]): List of preprocessed text documents
            
        Returns:
            List[str]: Sorted list of unique terms across all documents
        """
        # Extract all unique terms from all documents
        all_terms = set()
        for doc in documents:
            terms = doc.split()
            all_terms.update(terms)
        
        # Return sorted vocabulary for consistency
        return sorted(list(all_terms))
    
    def compute_term_frequency(self, document: str) -> Dict[str, float]:
        """
        Compute Term Frequency (TF) for a document.
        
        TF measures how frequently a term appears in a document.
        Formula: TF(term, doc) = (count of term in doc) / (total terms in doc)
        
        Args:
            document (str): Preprocessed text document
            
        Returns:
            Dict[str, float]: Dictionary mapping terms to their TF values
        """
        terms = document.split()
        total_terms = len(terms)
        
        if total_terms == 0:
            return {}
        
        # Count term frequencies
        term_counts = Counter(terms)
        
        # Calculate TF for each term
        tf_dict = {}
        for term, count in term_counts.items():
            tf_dict[term] = count / total_terms
        
        return tf_dict
    
    def compute_inverse_document_frequency(self, documents: List[str]) -> Dict[str, float]:
        """
        Compute Inverse Document Frequency (IDF) for all terms in the corpus.
        
        IDF measures how important a term is across all documents.
        Terms appearing in many documents get lower IDF scores.
        
        Formula: IDF(term) = log((1 + N) / (1 + df))
        where N = total documents, df = documents containing term
        
        Note: The +1 smoothing (similar to Laplace smoothing) prevents:
        - Division by zero
        - Log of zero
        - Over-penalization of terms appearing in all documents
        
        Args:
            documents (List[str]): List of preprocessed text documents
            
        Returns:
            Dict[str, float]: Dictionary mapping terms to their IDF values
        """
        total_docs = len(documents)
        
        # Count how many documents contain each term
        term_doc_count = {}
        for doc in documents:
            unique_terms = set(doc.split())
            for term in unique_terms:
                term_doc_count[term] = term_doc_count.get(term, 0) + 1
        
        # Calculate IDF for each term
        # Using standard IDF formula with smoothing to prevent log(0)
        # IDF = log((1 + N) / (1 + df)) where N is total docs, df is document frequency
        # The +1 smoothing is similar to Laplace smoothing
        idf_dict = {}
        for term, doc_count in term_doc_count.items():
            # Smooth IDF formula: prevents issues with rare/common terms
            idf_dict[term] = math.log((1 + total_docs) / (1 + doc_count))
        
        return idf_dict
    
    def compute_tfidf_vector(self, document: str, vocabulary: List[str], 
                            idf_dict: Dict[str, float]) -> np.ndarray:
        """
        Compute TF-IDF vector for a document.
        
        TF-IDF combines term frequency with inverse document frequency.
        Formula: TF-IDF(term, doc) = TF(term, doc) × IDF(term)
        
        Args:
            document (str): Preprocessed text document
            vocabulary (List[str]): List of all unique terms in corpus
            idf_dict (Dict[str, float]): IDF values for all terms
            
        Returns:
            np.ndarray: TF-IDF vector (NumPy array) for the document
        """
        # Compute term frequency for this document
        tf_dict = self.compute_term_frequency(document)
        
        # Initialize vector with zeros
        tfidf_vector = np.zeros(len(vocabulary))
        
        # Calculate TF-IDF for each term in vocabulary
        for idx, term in enumerate(vocabulary):
            if term in tf_dict:
                # TF-IDF = TF × IDF
                tf = tf_dict[term]
                idf = idf_dict.get(term, 0)
                tfidf_vector[idx] = tf * idf
        
        return tfidf_vector
    
    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Cosine similarity measures the cosine of the angle between two vectors.
        It ranges from 0 (completely dissimilar) to 1 (identical).
        
        Formula: cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
        where:
            - A · B is the dot product of vectors A and B
            - ||A|| is the Euclidean norm (magnitude) of vector A
        
        Args:
            vec1 (np.ndarray): First vector
            vec2 (np.ndarray): Second vector
            
        Returns:
            float: Cosine similarity score (0 to 1)
        """
        # Compute dot product using NumPy
        dot_product = np.dot(vec1, vec2)
        
        # Compute magnitudes (L2 norms) using NumPy
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity = dot product / (norm1 * norm2)
        similarity = dot_product / (norm1 * norm2)
        
        return similarity
    
    def analyze_resumes(self, job_description: str, resume_files: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Analyze resumes against a job description and rank candidates.
        
        This is the main analysis function that:
            1. Preprocesses all texts
            2. Creates TF-IDF vectors MANUALLY (from scratch)
            3. Computes cosine similarity scores MANUALLY (using NumPy)
            4. Ranks candidates
            5. Extracts matching keywords
        
        **MANUAL ML IMPLEMENTATION**:
        - No scikit-learn dependency
        - Pure Python + NumPy implementation
        - Python 3.13 compatible
        
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
        
        # Step 3: Build vocabulary and compute IDF (MANUAL IMPLEMENTATION)
        print("Building vocabulary and computing IDF values...")
        
        # Prepare documents: job description + all resumes
        documents = [job_desc_processed] + [r['processed_text'] for r in resume_data]
        
        # Build vocabulary from all documents
        self.vocabulary = self.build_vocabulary(documents)
        print(f"Vocabulary size: {len(self.vocabulary)} unique terms")
        
        # Compute IDF for all terms in vocabulary
        self.idf_dict = self.compute_inverse_document_frequency(documents)
        
        # Step 4: Create TF-IDF vectors (MANUAL IMPLEMENTATION)
        print("Computing TF-IDF vectors...")
        
        # Compute TF-IDF vector for job description
        job_vector = self.compute_tfidf_vector(job_desc_processed, self.vocabulary, self.idf_dict)
        
        # Compute TF-IDF vectors for all resumes
        resume_vectors = []
        for resume in resume_data:
            vector = self.compute_tfidf_vector(resume['processed_text'], self.vocabulary, self.idf_dict)
            resume_vectors.append(vector)
        
        # Step 5: Compute cosine similarity (MANUAL IMPLEMENTATION)
        print("Computing similarity scores...")
        similarities = []
        for resume_vector in resume_vectors:
            similarity = self.compute_cosine_similarity(job_vector, resume_vector)
            similarities.append(similarity)
        
        # Step 6: Create results dataframe
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
    print("Using MANUAL TF-IDF and Cosine Similarity implementation (Python 3.13 compatible)")
    print("No scikit-learn dependency - all ML implemented from scratch!")
