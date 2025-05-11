import pandas as pd
import requests
import os
import argparse
from tqdm import tqdm

def normalize_arxiv_id(arxiv_id):
    arxiv_id = str(arxiv_id).strip()
    # Remove 'arXiv:' or 'abs-' prefix
    if arxiv_id.lower().startswith('arxiv:'):
        arxiv_id = arxiv_id[6:]
    if arxiv_id.lower().startswith('abs-'):
        arxiv_id = arxiv_id[4:]
    # If it looks like 'cs-9308101v1', convert to 'cs/9308101v1'
    archive_prefixes = [
        'cs', 'math', 'hep-th', 'hep-ph', 'hep-ex', 'hep-lat', 'astro-ph', 'cond-mat',
        'gr-qc', 'nucl-ex', 'nucl-th', 'physics', 'quant-ph', 'nlin', 'q-bio', 'q-fin', 'stat', 'eess', 'econ'
    ]
    if '-' in arxiv_id and arxiv_id.split('-')[0] in archive_prefixes:
        parts = arxiv_id.split('-', 1)
        arxiv_id = f"{parts[0]}/{parts[1]}"
    return arxiv_id

def get_arxiv_url(arxiv_id):
    # Normalize the ID for all cases
    arxiv_id = normalize_arxiv_id(arxiv_id)
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

def main(csv_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        orig_arxiv_id = str(row['id'])
        arxiv_id = normalize_arxiv_id(orig_arxiv_id)
        url = get_arxiv_url(orig_arxiv_id)
        # Use a safe filename
        safe_id = arxiv_id.replace('/', '_')
        out_path = os.path.join(outdir, f"{safe_id}.pdf")
        if os.path.exists(out_path):
            continue
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(out_path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f"Failed to download {orig_arxiv_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--outdir', default='pdfs')
    args = parser.parse_args()
    main(args.csv, args.outdir)
