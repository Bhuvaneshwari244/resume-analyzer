import spacy
import PyPDF2
import os

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

def analyze_resume(resume_text, job_description):
    """Analyzes a resume against a job description and scores based on keyword matching."""
    resume_doc = nlp(resume_text.lower())
    job_doc = nlp(job_description.lower())

    resume_words = {token.lemma_ for token in resume_doc if token.is_alpha}
    job_words = {token.lemma_ for token in job_doc if token.is_alpha}

    matched_keywords = resume_words.intersection(job_words)
    score = len(matched_keywords) / len(job_words) * 100 if job_words else 0

    return {
        "score": round(score, 2),
        "matched_keywords": matched_keywords
    }

if __name__ == "__main__":
    # Example Usage
    job_description = """
    We are looking for a Python Developer with experience in machine learning, NLP, and web development.
    Skills: Python, NLP, spaCy, TensorFlow, Flask, REST APIs.
    """

    resume_path = "Sample_resume.pdf"  # Change to actual resume file
    if os.path.exists(resume_path):
        resume_text = extract_text_from_pdf(resume_path)
        result = analyze_resume(resume_text, job_description)
        print(f"Resume Match Score: {result['score']}%")
        print(f"Matched Keywords: {result['matched_keywords']}")
    else:
        print("Resume file not found!")
