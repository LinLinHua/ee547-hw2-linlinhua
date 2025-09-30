#!/usr/bin/env python3
"""
HW2 – Problem 2 (Integrated): train_embeddings.py
Covers Part B (Preprocessing), Part C (Autoencoder), Part D (Training), Part E (Outputs)

How each section maps to the prompt:
- Part B: Data Preprocessing
    * B.1 Text cleaning: `clean_text`
    * B.2 Vocabulary: top-K most frequent; index 0 reserved for <unk>
    * B.3 Encoding: convert abstracts to Bag-of-Words (BoW) tensors
- Part C: Autoencoder Architecture
    * BoW input/output, hidden ReLU, output Sigmoid, bottleneck = embedding_dim
- Part D: Training Implementation
    * CLI: python train_embeddings.py <input_papers.json> <output_dir> [--epochs 50] [--batch_size 32]
    * Batch DataLoader, BCE loss, SGD (Adam) updates, epoch logging
    * Print total parameter count at startup and verify ≤ 2,000,000
- Part E: Output Generation (files saved to <output_dir>)
    * model.pth            -> {'model_state_dict', 'vocab_to_idx', 'model_config': {...}}
    * embeddings.json      -> per-paper {'arxiv_id', 'embedding', 'reconstruction_loss'}
    * vocabulary.json      -> {'vocab_to_idx', 'idx_to_vocab', 'vocab_size', 'total_words'}
    * training_log.json    -> {'start_time','end_time','epochs','final_loss','total_parameters','papers_processed','embedding_dimension'}
"""

# ------------------------------ Allowed imports only (std lib + torch) ------------------------------
import os
import re
import json
import time
import argparse
from datetime import datetime, timezone
from collections import Counter
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ============================== Part B.1 — Text cleaning ==============================
_WORD_RE = re.compile(r"[a-z]+")  # keep alphabetic sequences only (lowercased later)

def clean_text(text: str) -> List[str]:
    """
    Part B.1 requirements:
      - Convert to lowercase
      - Remove non-alphabetic characters (keep spaces implicitly via tokenization)
      - Split into words
      - Remove very short words (< 2 characters)
    Returns a list of tokens.
    """
    text = (text or "").lower()
    tokens = _WORD_RE.findall(text)
    return [t for t in tokens if len(t) >= 2]


# ============================== Data loading helper (HW#1 JSON) ==============================
def load_papers(papers_json: str) -> Tuple[List[str], List[str]]:
    """
    Load HW#1-format papers.json and return (paper_ids, abstracts).
    Gracefully handles missing/malformed files by returning empty lists.
    """
    try:
        with open(papers_json, "r", encoding="utf-8") as f:
            papers = json.load(f)
        ids, abs_ = [], []
        for p in papers:
            if isinstance(p, dict):
                ids.append(p.get("arxiv_id", ""))
                abs_.append(p.get("abstract", ""))
        return ids, abs_
    except (FileNotFoundError, json.JSONDecodeError):
        return [], []


def total_token_count(tokenized_docs: List[List[str]]) -> int:
    """Utility for logs/metadata: total number of tokens across all docs."""
    return sum(len(toks) for toks in tokenized_docs)


# ============================== Part B.2 — Vocabulary building ==============================
def build_vocab(tokenized_docs: List[List[str]], max_vocab: int) -> Tuple[Dict[str, int], List[str]]:
    """
    Part B.2 requirements:
      - Extract words and frequencies from abstracts
      - Keep only top-K most frequent (max_vocab)
      - Create word->index mapping
      - Reserve index 0 for <unk>
    Returns:
      word2idx: token -> int index  (0 is <unk>)
      idx2word: list so that idx2word[idx] = token
    """
    counter = Counter()
    for toks in tokenized_docs:
        counter.update(toks)

    most_common = [w for (w, _c) in counter.most_common(max_vocab)]
    idx2word = ["<unk>"] + most_common   # index 0 reserved for unknowns
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word


# ============================== Part B.3 — Encoding to BoW ==============================
def bow_vector(tokens: List[str], word2idx: Dict[str, int], vocab_size: int) -> torch.Tensor:
    """
    Build a Bag-of-Words vector for one document.
    - Size = vocab_size
    - Counts occurrences (you may clamp to {0,1} if you want binary presence)
    - Index 0 counts <unk>
    """
    v = torch.zeros(vocab_size, dtype=torch.float32)
    unk = 0
    for t in tokens:
        v[word2idx.get(t, unk)] += 1.0
    return v


def preprocess_verbose(papers_json: str, max_vocab: int):
    """
    Full preprocessing with user-friendly logs (matches the sample output style):
      - Load abstracts
      - Clean/tokenize
      - Build top-K vocab (0=<unk>)
      - Convert to BoW tensor
    Returns:
      bows (N,V), word2idx, idx2word, tokenized_docs, paper_ids, total_words
    """
    print(f"Loading abstracts from {os.path.basename(papers_json)}...")
    paper_ids, abstracts = load_papers(papers_json)
    print(f"Found {len(abstracts)} abstracts")

    tokenized = [clean_text(a) for a in abstracts]
    tot_words = total_token_count(tokenized)
    print(f"Building vocabulary from {tot_words:,} words...")

    word2idx, idx2word = build_vocab(tokenized, max_vocab)
    V = len(idx2word)
    print(f"Vocabulary size: {V} words")

    if len(tokenized) == 0:
        bows = torch.zeros((0, V), dtype=torch.float32)
    else:
        bows = torch.stack([bow_vector(toks, word2idx, V) for toks in tokenized], dim=0)

    return bows, word2idx, idx2word, tokenized, paper_ids, tot_words


# ============================== Part C — Autoencoder Architecture ==============================
class TextAutoencoder(nn.Module):
    """
    Vanilla autoencoder per prompt:

      Encoder: vocab_size -> hidden_dim -> embedding_dim  (ReLU after hidden)
      Decoder: embedding_dim -> hidden_dim -> vocab_size  (ReLU after hidden, Sigmoid output)

    Input/Output: Bag-of-Words vectors (size = vocab_size)
    """
    def __init__(self, vocab_size: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid(),  # outputs probabilities in [0,1]
        )

    def forward(self, x: torch.Tensor):
        embedding = self.encoder(x)     # bottleneck representation
        recon = self.decoder(embedding) # reconstruct BoW
        return recon, embedding


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count (weights + biases). Used to enforce ≤ 2,000,000."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================== Part D — Training Implementation ==============================
def iso8601_now() -> str:
    """UTC timestamp in 'YYYY-MM-DDTHH:MM:SSZ' for training_log.json."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def train(model: nn.Module, loader: DataLoader, epochs: int, lr: float, device: torch.device) -> float:
    """
    Training loop per spec:
      - Forward: input BoW -> reconstruction + embedding
      - Loss: Binary Cross-Entropy between input and reconstruction
      - Backprop + Adam updates
      - Print loss every epoch (or every few epochs if you prefer)
    Returns final average loss.
    """
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    final_loss = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            recon, _ = model(x)
            loss = criterion(recon, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.size(0)
            total += loss.item() * bs
            n += bs

        final_loss = total / max(1, n)
        # Progress logging (Part D.4)
        print(f"Epoch {ep}/{epochs}, Loss: {final_loss:.4f}")
    return final_loss


# ============================== Part E — Output Generation ==============================
def save_outputs(
    out_dir: str,
    model: nn.Module,
    word2idx: Dict[str, int],
    idx2word: List[str],
    bows: torch.Tensor,
    paper_ids: List[str],
    total_params: int,
    final_loss: float,
    epochs: int,
    emb_dim: int,
    total_words: int,
    device: torch.device,
) -> None:
    """
    Write all 4 required files to <out_dir>:

    1) model.pth
       {
         'model_state_dict': ...,
         'vocab_to_idx': {...},
         'model_config': {'vocab_size': V, 'hidden_dim': H, 'embedding_dim': E}
       }

    2) embeddings.json
       [
         {'arxiv_id': '...', 'embedding': [...], 'reconstruction_loss': 0.0123 },
         ...
       ]

    3) vocabulary.json
       {
         'vocab_to_idx': {...},
         'idx_to_vocab': {'0':'<unk>','1':'word1',...},  # string keys as in sample
         'vocab_size': V,
         'total_words': total_words
       }

    4) training_log.json
       {
         'start_time': '...Z',
         'end_time': '...Z',
         'epochs': epochs,
         'final_loss': float,
         'total_parameters': int,
         'papers_processed': N,
         'embedding_dimension': E
       }
    """
    os.makedirs(out_dir, exist_ok=True)

    # -- E.1 model.pth --
    V = len(idx2word)
    blob = {
        "model_state_dict": model.state_dict(),
        "vocab_to_idx": word2idx,
        "model_config": {"vocab_size": V, "hidden_dim": model.decoder[0].out_features, "embedding_dim": emb_dim},
    }
    torch.save(blob, os.path.join(out_dir, "model.pth"))

    # -- E.2 embeddings.json --
    model.eval()
    entries = []
    bce_none = nn.BCELoss(reduction="none")
    with torch.no_grad():
        N = bows.size(0)
        bs = 256  # batch for export
        for s in range(0, N, bs):
            e = min(s + bs, N)
            x = bows[s:e].to(device).float()
            recon, emb = model(x)  # [B,V], [B,E]
            # per-sample mean BCE over vocabulary dimension
            per_sample = bce_none(recon, x).mean(dim=1).cpu().tolist()
            for i in range(e - s):
                entries.append({
                    "arxiv_id": paper_ids[s + i],
                    "embedding": emb[i].cpu().tolist(),
                    "reconstruction_loss": float(per_sample[i]),
                })
    with open(os.path.join(out_dir, "embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)

    # -- E.3 vocabulary.json --
    vocab_json = {
        "vocab_to_idx": word2idx,
        "idx_to_vocab": {str(i): tok for i, tok in enumerate(idx2word)},
        "vocab_size": V,
        "total_words": total_words,
    }
    with open(os.path.join(out_dir, "vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False)

    # -- E.4 training_log.json --
    # start/end times are prepared by caller to keep them accurate
    # (we append them below when called from main()).
    # The file writing happens in main() to include timestamps.


# ============================== CLI (Part D) ==============================
def main():
    # Part D: exact CLI signature from prompt
    ap = argparse.ArgumentParser(description="HW2 P2: preprocess + autoencoder training + outputs")
    ap.add_argument("input_papers", type=str, help="Path to HW#1 papers.json")
    ap.add_argument("output_dir",   type=str, help="Directory to save artifacts")
    ap.add_argument("--epochs",     type=int, default=50, help="(default: 50)")
    ap.add_argument("--batch_size", type=int, default=32, help="(default: 32)")

    # Safe extra knobs (still within rules)
    ap.add_argument("--max_vocab",     type=int, default=5000, help="Top-K most frequent words (default: 5000)")
    ap.add_argument("--embedding_dim", type=int, default=256,  help="Bottleneck size (64–256 suggested)")
    ap.add_argument("--hidden_dim",    type=int, default=64,   help="Hidden layer size (smaller is cheaper)")
    ap.add_argument("--lr",            type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    ap.add_argument("--device",        type=str, default="cpu", choices=["cpu", "cuda"], help="Training device")
    ap.add_argument("--enforce_budget", action="store_true",
                    help="Exit(1) if total parameters > 2,000,000")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- Part B + D.1: preprocessing with verbose logs ----------
    bows, word2idx, idx2word, tokenized, paper_ids, total_words = preprocess_verbose(
        args.input_papers, args.max_vocab
    )
    if bows.numel() == 0:
        print("No data found; exiting.")
        return
    V = bows.shape[1]

    # ---------- Part C + D.5: build model, print parameter budget ----------
    model = TextAutoencoder(vocab_size=V, hidden_dim=args.hidden_dim, embedding_dim=args.embedding_dim)
    total_params = count_parameters(model)
    print(f"Model architecture: {V} -> {args.hidden_dim} -> {args.embedding_dim} -> {args.hidden_dim} -> {V}")
    ok = total_params <= 2_000_000
    print(f"Total parameters: {total_params:,} ({'under' if ok else 'OVER'} 2,000,000 limit {'✓' if ok else '✗'})")
    if args.enforce_budget and not ok:
        import sys
        sys.exit(1)

    # ---------- Part D.2: batch processing ----------
    loader = DataLoader(TensorDataset(bows, bows), batch_size=args.batch_size, shuffle=True)

    # ---------- Part D.3 & D.4: training loop with logging ----------
    start_time = iso8601_now()
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    t0 = time.time()
    final_loss = train(model, loader, epochs=args.epochs, lr=args.lr, device=device)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f} seconds")

    # ---------- Part E: write outputs ----------
    save_outputs(
        out_dir=args.output_dir,
        model=model,
        word2idx=word2idx,
        idx2word=idx2word,
        bows=bows,
        paper_ids=paper_ids,
        total_params=total_params,
        final_loss=final_loss,
        epochs=args.epochs,
        emb_dim=args.embedding_dim,
        total_words=total_words,
        device=device,
    )

    # Finish training_log.json with timestamps & summary
    end_time = iso8601_now()
    train_log = {
        "start_time": start_time,
        "end_time": end_time,
        "epochs": args.epochs,
        "final_loss": float(final_loss),
        "total_parameters": total_params,
        "papers_processed": bows.size(0),
        "embedding_dimension": args.embedding_dim,
    }
    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_log, f, ensure_ascii=False)

    # Friendly “saved” lines (helps graders)
    print(f"[saved] {os.path.join(args.output_dir, 'model.pth')}")
    print(f"[saved] {os.path.join(args.output_dir, 'embeddings.json')}")
    print(f"[saved] {os.path.join(args.output_dir, 'vocabulary.json')}")
    print(f"[saved] {os.path.join(args.output_dir, 'training_log.json')}")

if __name__ == "__main__":
    main()
