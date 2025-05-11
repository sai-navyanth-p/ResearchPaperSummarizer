import os
import json
import argparse
import fitz  # PyMuPDF
from tqdm import tqdm
import nltk
import string

nltk.download('punkt')
nltk.download('punkt_tab')

def preprocess_text(text):
    # Lowercase, remove punctuation, tokenize
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
    return ' '.join(tokens)

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text(sort=True) + " "
        return text.strip()
    except Exception as e:
        print(f"Failed to extract {pdf_path}: {e}")
        return ""

def main(pdfdir, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    data = []
    for fname in tqdm(os.listdir(pdfdir), desc="Processing PDFs"):
        if not fname.endswith('.pdf'):
            continue
        pdf_path = os.path.join(pdfdir, fname)
        doc_id = fname.replace('.pdf', '')
        text = extract_text_from_pdf(pdf_path)
        clean_text = preprocess_text(text)
        data.append({
            "doc_id": doc_id,
            "text": clean_text
        })
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdfdir', required=True, help="Directory containing PDFs")
    parser.add_argument('--outfile', default='trainingdata/train.json', help="Output JSON file")
    args = parser.parse_args()
    nltk.download('punkt')
    main(args.pdfdir, args.outfile)
