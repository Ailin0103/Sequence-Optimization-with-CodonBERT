from pathlib import Path
import numpy as np
import pandas as pd
import re
from Bio import SeqIO
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForPreTraining
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from transformers import AutoModel


def find_repo_root(start: Optional[Path] = None, marker: str = "pyproject.toml") -> Path:
    """
    Find repo root by walking up parents until we see `marker`.
    Works no matter where you run the script from.
    """
    p = (start or Path(__file__)).resolve()
    if p.is_file():
        p = p.parent
    for _ in range(8):
        if (p / marker).exists():
            return p
        p = p.parent
    raise RuntimeError(f"Repo root not found (marker={marker}). Run inside the repo tree.")

REPO_ROOT = find_repo_root(Path(__file__))
WORKSPACE_ROOT = REPO_ROOT.parent

DATA_RAW = WORKSPACE_ROOT / "data_raw"
DATA_CACHE = WORKSPACE_ROOT / "data_cache"


def assert_inputs() -> None:
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"DATA_RAW not found at: {DATA_RAW}")
    if not DATA_CACHE.exists():
        DATA_CACHE.mkdir(parents=True, exist_ok=True)


# =========================
# Cleaning FASTA → DataFrame
# =========================

IUPAC_MAPPING = {
    # keep it minimal based on your use-case; expand if needed
    "R": "A",  # A/G
    "Y": "C",  # C/T
    "D": "A",  # A/G/T
    "H": "A",  # A/C/T
    "N": "A",  # any
}

_REPEAT_PATTERN = re.compile(r"\(([A-Z])(\d+)\)")


def _expand_repeats_and_iupac(seq: str) -> str:
    """
    Expand patterns like (N20) -> 'A'*20 and replace ambiguous IUPAC bases.
    """
    seq = seq.upper().strip()

    def repl(m: re.Match) -> str:
        base = m.group(1)
        num = int(m.group(2))
        base_clean = IUPAC_MAPPING.get(base, base)
        return base_clean * num

    seq = _REPEAT_PATTERN.sub(repl, seq)

    # If anything like "( ) 123" remains, remove (defensive)
    seq = re.sub(r"[\(\)\d]", "", seq)

    # Replace remaining IUPAC ambiguity codes
    seq = "".join(IUPAC_MAPPING.get(b, b) for b in seq)
    return seq


def clean_fasta_to_df(input_fasta: Path) -> pd.DataFrame:
    """
    Read FASTA, expand repeats like (N20), replace IUPAC bases, return DataFrame:
    ['id', 'original_seq', 'clean_seq'].
    """
    records: List[Dict[str, str]] = []
    for record in SeqIO.parse(str(input_fasta), "fasta"):
        seq_str = str(record.seq).upper()
        cleaned = _expand_repeats_and_iupac(seq_str)
        records.append({"id": record.id, "original_seq": seq_str, "clean_seq": cleaned})

    df = pd.DataFrame(records)
    print(f"✅ Cleaned and expanded {len(df)} sequences from {input_fasta}")
    return df


def save_clean_fasta(df: pd.DataFrame, output_path: Path, seq_col: str = "clean_seq", id_col: str = "id") -> None:
    """
    Save sequences from DataFrame to FASTA.
    """
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq

    out_records = [
        SeqRecord(Seq(row[seq_col]), id=row[id_col], description="")
        for _, row in df.iterrows()
    ]
    SeqIO.write(out_records, str(output_path), "fasta")
    print(f"✅ Saved {len(out_records)} sequences to {output_path}")


# =========================
# ORF screening (ATG...TAA)
# NOTE:
# Kudla et al. (Science 2009) oligonucleotides are synthetic fragments
# used for PCR assembly of full-length GFP genes.
# They are NOT complete ORFs and therefore should not be screened
# using start/stop codon criteria.

# =========================
def quick_orf_diagnosis(df, col="clean_seq"):
    seqs = df[col].astype(str)

    has_ATG_anywhere = seqs.str.contains("ATG").mean()
    has_TAA = seqs.str.contains("TAA").mean()
    has_TAG = seqs.str.contains("TAG").mean()
    has_TGA = seqs.str.contains("TGA").mean()

    print("Fraction containing ATG anywhere:", has_ATG_anywhere)
    print("TAA:", has_TAA, "TAG:", has_TAG, "TGA:", has_TGA)
    print("Any stop:",
          (seqs.str.contains("TAA|TAG|TGA")).mean())
    
def validate_orf_oligos(seq: str, slide_step: int = 1) -> Dict:
    """
    Validate ORF:
    - Start codon: ATG
    - Stop codon: TAA (first in-frame after start)
    - Coding region length between start and stop must be multiple of 3 (in-frame stop search enforces this)

    Returns a dict with keys: valid, reason, start_index, stop_index, coding_length, start_codon, stop_codon
    """
    seq = seq.strip().upper()
    START = {"ATG"}
    STOP = {"TAA"}

    start_index = None
    for i in range(0, len(seq) - 2, slide_step):
        if seq[i:i+3] in START:
            start_index = i
            break
    if start_index is None:
        return {"valid": False, "reason": "No start codon found.", "start_index": None, "stop_index": None}

    stop_index = None
    for j in range(start_index + 3, len(seq) - 2, 3):
        if seq[j:j+3] in STOP:
            stop_index = j
            break
    if stop_index is None:
        return {
            "valid": False,
            "reason": "No in-frame stop codon found after start.",
            "start_index": start_index,
            "stop_index": None,
            "start_codon": seq[start_index:start_index+3],
        }

    coding_len = stop_index - start_index - 3
    return {
        "valid": True,
        "reason": "Valid ORF found.",
        "start_index": start_index,
        "stop_index": stop_index,
        "coding_length": coding_len,
        "start_codon": seq[start_index:start_index+3],
        "stop_codon": seq[stop_index:stop_index+3],
    }


def screen_df_for_orf(df: pd.DataFrame, seq_col: str = "clean_seq", slide_step: int = 3) -> pd.DataFrame:
    results = []
    for _, row in df.iterrows():
        res = validate_orf_oligos(row[seq_col], slide_step=slide_step)
        res["id"] = row["id"]
        res["seq_len"] = len(row[seq_col])
        results.append(res)

    out = pd.DataFrame(results)
    print(f"✅ ORF screening complete: {len(out)} sequences processed")
    if "valid" in out.columns:
        print(out["valid"].value_counts())
    return out


# =========================
# CodonBERT embeddings
# =========================

@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "lhallee/CodonBERT"
    max_len: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def diagnose_tokenization(df_clean, tokenizer, seq_col="clean_seq", n=5):

    print("\n=== Tokenization diagnosis ===")
    unk_id = tokenizer.unk_token_id
    print("Tokenizer:", tokenizer.__class__.__name__)
    print("unk_token:", tokenizer.unk_token, "unk_token_id:", unk_id)
    print("model_max_length:", getattr(tokenizer, "model_max_length", None))

    # pick a few sequences
    n = min(n, len(df_clean))
    idxs = list(range(n))
    seqs = [df_clean[seq_col].iloc[i] for i in idxs]

    prefixes = []
    unk_ratios = []
    lengths = []

    for i, s in enumerate(seqs):
        enc = tokenizer(to_codon_string(s), truncation=True, padding=False)
        ids = enc["input_ids"]
        lengths.append(len(ids))
        prefixes.append(tuple(ids[:60]))

        if unk_id is not None and len(ids) > 0:
            ur = sum(t == unk_id for t in ids) / len(ids)
        else:
            ur = float("nan")
        unk_ratios.append(ur)

        print(f"\n--- sample {i} ---")
        print("seq length (nt):", len(s))
        print("input_ids length:", len(ids))
        print("first 30 input_ids:", ids[:30])
        print("UNK ratio:", ur)

    print("\nUnique prefixes among samples (first 60 ids):", len(set(prefixes)))
    print("Token length stats min/median/max:", min(lengths), np.median(lengths), max(lengths))
    if unk_id is not None and len(unk_ratios) > 0:
        print("UNK ratio stats min/median/max:",
              np.nanmin(unk_ratios), np.nanmedian(unk_ratios), np.nanmax(unk_ratios))
    print("=== End diagnosis ===\n")

def to_codon_string(seq: str) -> str:
    """
    Convert DNA sequence into codon tokens with whitespace.
    Example: ATGGAACTT -> 'ATG GAA CTT'
    """
    seq = seq.upper().strip()
    L = len(seq) - (len(seq) % 3)
    return " ".join(seq[i:i+3] for i in range(0, L, 3))

def get_sequence_embedding_meanpool(seq: str, tokenizer, model, max_len: int, device: str) -> torch.Tensor:
    """
    Return one vector per sequence: mask-aware mean pooling over last hidden state.
    Output shape: [hidden_dim] on CPU.
    """
    inputs = tokenizer(
        seq,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1]  # [1, L, H]
    mask = attention_mask.unsqueeze(-1).float()  # [1, L, 1]

    summed = (last_hidden * mask).sum(dim=1)      # [1, H]
    counts = mask.sum(dim=1).clamp(min=1.0)       # [1, 1]
    mean_pooled = (summed / counts).squeeze(0)    # [H]

    return mean_pooled.detach().cpu()

def get_sequence_embedding_probe(
    seq: str,
    tokenizer,
    model,
    max_len: int,
    device: str,
    mid_frac: float = 0.6,   # 取中间 60%
) -> torch.Tensor:
    """
    Return one vector per sequence, designed for short/overlapping oligos:
    - Get last_hidden_state: [1, L, H]
    - Remove special tokens (CLS/SEP) if present
    - Take middle window tokens (by fraction)
    - Compute mean + std over tokens, then concatenate: [2H]
    Output shape: [2*hidden_dim] on CPU
    """
    inputs = tokenizer(
        seq,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    h = outputs.last_hidden_state.squeeze(0)          # [L, H]
    m = attention_mask.squeeze(0).bool()              # [L]

    # keep only real tokens (exclude PAD)
    h = h[m]                                          # [L_real, H]

    # drop special tokens if present (many BERTs have them at ends)
    # this is safe even if the model doesn't use special tokens heavily
    if h.shape[0] >= 4:
        h = h[1:-1]                                   # drop first/last

    L = h.shape[0]
    if L == 0:
        raise RuntimeError("Empty token sequence after masking.")

    # take middle window (more robust for overlap-based oligos)
    frac = max(0.2, min(0.9, mid_frac))
    w = int(L * frac)
    w = max(4, min(L, w))
    start = (L - w) // 2
    end = start + w
    hw = h[start:end]                                 # [w, H]

    # mean and std pooling (std captures local variability)
    mean_vec = hw.mean(dim=0)
    std_vec = hw.std(dim=0, unbiased=False)

    vec = torch.cat([mean_vec, std_vec], dim=0)       # [2H]
    return vec.detach().cpu()

def compute_embeddings_for_df_meanpool(
    df: pd.DataFrame,
    seq_col: str = "clean_seq",
    cfg: EmbeddingConfig = EmbeddingConfig(),
) -> pd.DataFrame:
    """
    Compute ONE embedding vector per sequence. Great for PCA/clustering.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForPreTraining.from_pretrained(cfg.model_name).to(cfg.device)
    model.eval()

    seq_vecs = []
    for seq in tqdm(df[seq_col], desc=f"Embedding sequences ({cfg.device})"):
        v = get_sequence_embedding_meanpool(seq, tokenizer, model, max_len=cfg.max_len, device=cfg.device)
        seq_vecs.append(v)

    out = df[["id", seq_col]].copy()
    out.rename(columns={seq_col: "clean_seq"}, inplace=True)
    out["seq_embedding"] = seq_vecs  # each is [hidden_dim]
    return out

def compute_embeddings_for_df(
    df: pd.DataFrame,
    seq_col: str = "clean_seq",
    cfg: EmbeddingConfig = EmbeddingConfig(),
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModel.from_pretrained(cfg.model_name).to(cfg.device)
    model.eval()

    seq_vecs = []
    for seq in tqdm(df[seq_col], desc=f"Embedding sequences ({cfg.device})"):
        v = get_sequence_embedding_probe(
            seq, tokenizer, model, max_len=cfg.max_len, device=cfg.device, mid_frac=0.6
        )
        seq_vecs.append(v)

    out = df[["id", seq_col]].copy()
    out.rename(columns={seq_col: "clean_seq"}, inplace=True)
    out["seq_embedding"] = seq_vecs
    return out


# =========================
# Main pipeline
# =========================

def main() -> None:
    assert_inputs()

    fasta_path = DATA_RAW / "oligos.fasta"
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    # Quick peek (optional)
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        print(rec.id)
        print(repr(rec.seq))
        print(len(rec))
        break

    # 1) Clean
    df_clean = clean_fasta_to_df(fasta_path)
    quick_orf_diagnosis(df_clean, col="clean_seq")

    # 2) Save cleaned fasta to cache
    cleaned_fasta = DATA_CACHE / "oligos_clean.fasta"
    save_clean_fasta(df_clean, cleaned_fasta)

    #）  Diagnose tokenization
    cfg = EmbeddingConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    diagnose_tokenization(df_clean, tokenizer, seq_col="clean_seq", n=5)
    # 3) Embeddings (optional, can be slow on CPU)
    
    df_emb = compute_embeddings_for_df(df_clean, seq_col="clean_seq", cfg=cfg)
    print(df_emb["seq_embedding"].iloc[0].shape)  # e.g. torch.Size([768])

    X = np.stack([v.numpy() if hasattr(v, "numpy") else np.array(v) for v in df_emb["seq_embedding"]])
    print("X shape:", X.shape)   # (n_sequences, hidden_dim)


    X = np.stack(df_emb["seq_embedding"].values)

    # 看前两条 embedding 的差异
    print("L2 distance between seq 0 and 1:",
        np.linalg.norm(X[0] - X[1]))

    print("Std across samples (mean over dims):",
        X.std(axis=0).mean())

    # PCA visualization 2D
    from sklearn.decomposition import PCA

    Z = PCA(n_components=2, random_state=0).fit_transform(X)
    df_emb["pca1"] = Z[:, 0]
    df_emb["pca2"] = Z[:, 1]
    
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(df_emb["pca1"], df_emb["pca2"])
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("CodonBERT embeddings (PCA)")
    plt.show()
    from sklearn.cluster import KMeans

    #Kmeans clustering on embeddings
    k = 3  # 先随便试 3 或 4
    labels = KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(X)
    df_emb["cluster"] = labels
    df_emb["cluster"].value_counts()
    plt.figure()
    
    plt.scatter(df_emb["pca1"], df_emb["pca2"], c=df_emb["cluster"])
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("PCA + KMeans clusters")
    plt.show()

    # Cluster stats
    def gc_content(s: str) -> float:
        s = s.upper()
        return (s.count("G") + s.count("C")) / max(len(s), 1)

    df_emb["len"] = df_emb["clean_seq"].str.len()
    df_emb["gc"] = df_emb["clean_seq"].apply(gc_content)
    df_emb.groupby("cluster")[["len", "gc"]].agg(["count", "mean", "std"])

if __name__ == "__main__":
    main()