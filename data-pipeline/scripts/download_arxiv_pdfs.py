import pandas as pd
import requests
import os
import argparse
from tqdm import tqdm

def get_arxiv_url(arxiv_id):
    if '-' not in arxiv_id:
        return None
    category, rest = arxiv_id.split('-', 1)
    return f"https://arxiv.org/pdf/{category}/{rest}.pdf"

def main(csv_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        arxiv_id = str(row['id'])
        url = get_arxiv_url(arxiv_id)
        if not url:
            print(f"Skipping malformed id: {arxiv_id}")
            continue
        out_path = os.path.join(outdir, f"{arxiv_id}.pdf")
        if os.path.exists(out_path):
            continue
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(out_path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f"Failed to download {arxiv_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--outdir', default='pdfs')
    args = parser.parse_args()
    main(args.csv, args.outdir)
