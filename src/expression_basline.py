import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# =====================
# Config
# =====================
MODEL_NAME = "lhallee/CodonBERT"

DATA_PATH = Path(__file__).parent.parent / "data" / "Supplementary_Data_15_Integrated_Phenotypic_Measurements.csv"

SEQ_COL = "gs.sequence"
TARGET_COL = "clean.lin.prot.mean"
STRUCTURE_TARGET_COL = "gs.utrCdsStructureMFE"
MAX_LEN = 512
MODEL_BATCH_SIZE = 8
RANDOM_STATE = 42


# =====================
# Sequence cleaning
# =====================
IUPAC_MAPPING = {
    "R": "A", "Y": "C", "D": "A", "H": "A", "N": "A",
    "B": "A", "V": "A", "W": "A", "S": "A", "M": "A", "K": "A",
}


def clean_sequence(seq):
    seq = str(seq).upper().strip()

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
# Embedding
# =====================
def mean_pool_embedding(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def compute_codonbert_embeddings(sequences, tokenizer, model, device):
    all_embeddings = []

    for start in tqdm(range(0, len(sequences), MODEL_BATCH_SIZE), desc="Embedding"):
        batch_seqs = sequences[start:start + MODEL_BATCH_SIZE]

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        #pooled = mean_pool_embedding(
        #    outputs.last_hidden_state,
        #    inputs["attention_mask"]
        #).detach().cpu().numpy()

        # CLS embedding: one 768-d vector per sequence
        pooled = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        all_embeddings.append(pooled)

    return np.vstack(all_embeddings)

def compute_codonbert_embeddings_MFE(sequences, tokenizer, model, device, pooling="mean", first_k_tokens=None):
    all_embeddings = []

    for start in tqdm(range(0, len(sequences), MODEL_BATCH_SIZE), desc=f"Embedding-{pooling}-{first_k_tokens}"):
        batch_seqs = sequences[start:start + MODEL_BATCH_SIZE]

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        if pooling == "cls":
            pooled = hidden[:, 0, :]

        elif pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        elif pooling == "first_k_mean":
            # token 0 is [CLS], so real codon tokens start at 1
            # take tokens 1 to first_k_tokens
            k = first_k_tokens
            local_hidden = hidden[:, 1:1 + k, :]
            local_mask = attention_mask[:, 1:1 + k].unsqueeze(-1)
            pooled = (local_hidden * local_mask).sum(dim=1) / local_mask.sum(dim=1).clamp(min=1)

        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        all_embeddings.append(pooled.detach().cpu().numpy())

    return np.vstack(all_embeddings)
# =====================
# Evaluation
# =====================
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    spearman = spearmanr(y_test, pred).correlation

    print(f"\n{name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Spearman: {spearman:.4f}")

    return {
        "model": name,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
    }


def main():
    # Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv(DATA_PATH)
    df["protein_mean"] = df[
    [
        "log.prot.repOne",
        "log.prot.repTwo",
        "log.prot.repThree",
        "log.prot.repFour",
    ]
    ].mean(axis=1)  
    
    df = df[[SEQ_COL, TARGET_COL, STRUCTURE_TARGET_COL]].copy()
    df = df.dropna(subset=[SEQ_COL, TARGET_COL, STRUCTURE_TARGET_COL])

    df["clean_seq"] = df[SEQ_COL].apply(format_sequence_for_codonbert)
    df = df.dropna(subset=["clean_seq"])

    print("Filtered shape:", df.shape)
    print(df[[SEQ_COL, "clean_seq", TARGET_COL]].head())

    # Optional first run: small sample
    df = df.sample(n=10000, random_state=RANDOM_STATE).reset_index(drop=True)

    sequences = df["clean_seq"].tolist()
    y_expr = df[TARGET_COL].astype(float).values
    y_mfe = df[STRUCTURE_TARGET_COL].astype(float).values
    # Load CodonBERT
    print("\nLoading CodonBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    pooling_experiments = [
    ("mean_all", "mean", None),
    ("first_10_codons", "first_k_mean", 10),
    ("first_20_codons", "first_k_mean", 20),
]

    for exp_name, pooling, first_k in pooling_experiments:
        print("\n" + "=" * 60)
        print(f"Experiment: {exp_name}")
        print("=" * 60)

        X = compute_codonbert_embeddings_MFE(
            sequences,
            tokenizer,
            model,
            device,
            pooling=pooling,
            first_k_tokens=first_k,
        )

        # 先预测 protein_mean
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_expr,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

        evaluate_model(
            f"Ridge protein_mean - {exp_name}",
            make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            X_train,
            X_test,
            y_train,
            y_test,
        )

        # 再预测 MFE
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_mfe,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

        evaluate_model(
            f"Ridge MFE - {exp_name}",
            make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            X_train,
            X_test,
            y_train,
            y_test,
        )
        

    # Embedding
    X = compute_codonbert_embeddings(sequences, tokenizer, model, device)
    print("Embedding shape:", X.shape)
    print(np.mean(X))
    print(np.std(X))
    print(np.min(X))
    print(np.max(X))
    # Train/test split
    print("\n==============================")
    print("Predicting UTR-CDS Structure MFE")
    print("==============================")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_mfe,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    evaluate_model(
        "Ridge - MFE",
        make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
        X_train,
        X_test,
        y_train,
        y_test,
    )
    print("\n==============================")
    print("Predicting Protein Expression")
    print("==============================")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_expr,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    results = []

    results.append(
        evaluate_model(
            "Linear Regression",
            LinearRegression(),
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    results.append(
        evaluate_model(
            "Ridge",
            Ridge(alpha=1.0),
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    #results.append(
    #    evaluate_model(
    #        "Random Forest",
    #        RandomForestRegressor(
    #            n_estimators=200,
    #            random_state=RANDOM_STATE,
    #            n_jobs=-1,
    #        ),
    #        X_train,
    #        X_test,
    #        y_train,
    #        y_test,
    #    )
    #)

    results_df = pd.DataFrame(results)
    output_path = Path(__file__).parent.parent / "data" / "expression_baseline_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nSaved results to:", output_path)
    print(results_df)


if __name__ == "__main__":
    
    main()