from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import fitz  # PyMuPDF
import os

app = Flask(__name__)

local_dir = "/app/bart_model"  # Match this with save_path above
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["pdf"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Invalid file type"}), 400

    # Save and read file
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    # Extract text from PDF
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    # Summarize (truncate long input for BART)
    max_chunk = 1024
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summaries = [summarizer(chunk)[0]["summary_text"] for chunk in chunks]
    final_summary = " ".join(summaries)

    return jsonify({"summary": final_summary})

if __name__ == "__main__":
    app.run(debug=True)
