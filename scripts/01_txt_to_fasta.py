from pathlib import Path
from typing import Optional

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

def txt_to_fasta(txt_file, fasta_file):
    with open(txt_file, "r") as infile, open(fasta_file, "w") as outfile:
        for line in infile:
            # Skip headers or empty lines
            if not line.strip():
                continue
            if line.lower().startswith("oligo name"):
                continue
            
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                name, seq = parts[0], parts[1]
                outfile.write(f">{name}\n{seq}\n")

# Example usage
input_txt = REPO_ROOT / "data" / "meta" / "kudla.som.oligos.txt"
output_fasta = DATA_RAW / "oligos.fasta"
txt_to_fasta(input_txt, output_fasta)
print(f"✅ FASTA file written to {output_fasta}")
