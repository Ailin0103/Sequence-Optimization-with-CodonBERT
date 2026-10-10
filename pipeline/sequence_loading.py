import re
import io
import json
import boto3
import torch
import pandas as pd
from Bio.Seq import Seq
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForPreTraining
from boto3.session import Session
from .config import PROFILE_NAME, REGION_NAME
from tqdm import tqdm

# =========================
# Config
# =========================
BUCKET = "sequence-data-ztgg"

INPUT_KEY = "SRR4163265.fasta"
OUTPUT_PREFIX = "codonbert_inference/"
BATCH_SIZE = 10_000

session = Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
s3 = session.client("s3")

S3_BATCH_SIZE = 1000

MODEL_BATCH_SIZE = 8

MAX_LEN = 512

MODEL_NAME = "lhallee/CodonBERT"

# =====================

# Cleaning logic from teacher

# =====================

IUPAC_MAPPING = {

    "R": "A", "Y": "C", "D": "A", "H": "A", "N": "A",

    "B": "A", "V": "A", "W": "A", "S": "A", "M": "A", "K": "A",

}

def clean_sequence(seq):

    seq = seq.upper().strip()

    def repl(match):

        base = match.group(1)

        num = int(match.group(2))

        base_clean = IUPAC_MAPPING.get(base, base)

        return base_clean * num

    seq = re.sub(r"\(([A-Z])(\d+)\)", repl, seq)

    seq = re.sub(r"[\(\)\d]", "", seq)

    seq = "".join(IUPAC_MAPPING.get(b, b) for b in seq)

    return seq

def format_sequence_for_codonbert(seq):

    seq = clean_sequence(seq)

    if len(seq) == 0:

        return None

    if len(seq) % 3 != 0:

        return None

    if not all(c in "ATCG" for c in seq):

        return None

    codons = [seq[i:i + 3] for i in range(0, len(seq), 3)]

    return " ".join(codons)

# =====================

# Stream FASTA from S3

# =====================

def stream_fasta_from_s3(bucket, key):

    obj = s3.get_object(Bucket=bucket, Key=key)

    body = obj["Body"]

    current_id = None

    seq_lines = []

    for raw_line in body.iter_lines():

        line = raw_line.decode("utf-8").strip()

        if not line:

            continue

        if line.startswith(">"):

            if current_id is not None:

                yield current_id, "".join(seq_lines)

            current_id = line[1:].split()[0]

            seq_lines = []

        else:

            seq_lines.append(line)

    if current_id is not None:

        yield current_id, "".join(seq_lines)

# =====================

# CodonBERT inference logic

# =====================

def mean_pool_embedding(hidden_states, attention_mask):

    mask = attention_mask.unsqueeze(-1)

    summed = (hidden_states * mask).sum(dim=1)

    counts = mask.sum(dim=1).clamp(min=1)

    return summed / counts

def run_codonbert_inference(df, tokenizer, model, device):

    all_rows = []

    seqs = df["clean_seq"].tolist()

    ids = df["id"].tolist()

    for start in tqdm(range(0, len(seqs), MODEL_BATCH_SIZE), desc="Model inference"):

        batch_seqs = seqs[start:start + MODEL_BATCH_SIZE]

        batch_ids = ids[start:start + MODEL_BATCH_SIZE]

        inputs = tokenizer(

            batch_seqs,

            return_tensors="pt",

            truncation=True,

            padding=True,

            max_length=MAX_LEN,

        ).to(device)

        with torch.no_grad():

            outputs = model(**inputs, output_hidden_states=True)

       
        # Mean pooled embedding: one 768-d vector per sequence

        pooled = mean_pool_embedding(

            outputs.hidden_states[-1],

            inputs["attention_mask"]

        ).detach().cpu()

        for i, seq_id in enumerate(batch_ids):

            row = {

                "id": seq_id,

                "clean_seq": batch_seqs[i],

                "num_codons": len(batch_seqs[i].split()),

            }

            emb = pooled[i].numpy()

            for j, value in enumerate(emb):

                row[f"emb_{j}"] = float(value)

            # 不建议保存完整 MLM logits，因为非常大

            # 如果以后需要，可以单独存 torch 文件或 npz

            all_rows.append(row)

    return pd.DataFrame(all_rows)

# =====================

# Write parquet to S3

# =====================

def write_df_to_s3_parquet(df, bucket, key):

    buffer = io.BytesIO()

    df.to_parquet(buffer, index=False)

    buffer.seek(0)

    s3.put_object(

        Bucket=bucket,

        Key=key,

        Body=buffer.getvalue()

    )

# =====================

# Read parquet from S3

# =====================


def read_df_from_s3_parquet(bucket, key):

    obj = s3.get_object(Bucket=bucket, Key=key)

    body = obj["Body"].read()

    return pd.read_parquet(io.BytesIO(body))


def load_embedding_batch_from_s3(batch_num=1, bucket=BUCKET, prefix=OUTPUT_PREFIX):

    key = f"{prefix}embeddings_batch_{batch_num:05d}.parquet"

    df = read_df_from_s3_parquet(bucket, key)

    print(f"Loaded {len(df)} rows from s3://{bucket}/{key}")

    return df




# =====================

# Main pipeline

# =====================

def main():

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Using device: {device}")

    print("Loading tokenizer and CodonBERT...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForPreTraining.from_pretrained(MODEL_NAME).to(device)

    model.eval()

    batch_records = []

    batch_num = 0

    total_raw = 0

    total_kept = 0

    START_BATCH = 1554
    SKIP_KEPT = (START_BATCH - 1) * S3_BATCH_SIZE
    batch_num = START_BATCH - 1

    for seq_id, raw_seq in stream_fasta_from_s3(BUCKET, INPUT_KEY):

        total_raw += 1

        formatted_seq = format_sequence_for_codonbert(raw_seq)

        if formatted_seq is None:

            continue

        total_kept += 1
        if total_kept <= SKIP_KEPT:

            continue

        batch_records.append({

            "id": seq_id,

            "clean_seq": formatted_seq,

        })

        if len(batch_records) >= S3_BATCH_SIZE:

            batch_num += 1

            process_and_save_batch(batch_records, batch_num, tokenizer, model, device)

            batch_records = []

    if batch_records:
        
        batch_num += 1

        process_and_save_batch(batch_records, batch_num, tokenizer, model, device)

    summary = {

        "input": f"s3://{BUCKET}/{INPUT_KEY}",

        "output_prefix": f"s3://{BUCKET}/{OUTPUT_PREFIX}",

        "total_raw_sequences": total_raw,

        "total_kept_sequences": total_kept,

        "num_batches": batch_num,

        "s3_batch_size": S3_BATCH_SIZE,

        "model_batch_size": MODEL_BATCH_SIZE,

        "model_name": MODEL_NAME,

    }

    s3.put_object(

        Bucket=BUCKET,

        Key=f"{OUTPUT_PREFIX}summary.json",

        Body=json.dumps(summary, indent=2),

    )

    print("Done.")

    print(summary)

def process_and_save_batch(batch_records, batch_num, tokenizer, model, device):

    print(f"\nProcessing S3 batch {batch_num}, n={len(batch_records)}")

    df = pd.DataFrame(batch_records)

    df_out = run_codonbert_inference(df, tokenizer, model, device)

    output_key = f"{OUTPUT_PREFIX}embeddings_batch_{batch_num:05d}.parquet"

    write_df_to_s3_parquet(df_out, BUCKET, output_key)

    print(f"Saved: s3://{BUCKET}/{output_key}")

if __name__ == "__main__":

    main()