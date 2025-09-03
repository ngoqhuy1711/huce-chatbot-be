import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from preprocessing import normalize


def build_index(intent_data="data/intent_prepared.csv", out_dir="nlu/intent/artifacts"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(intent_data)
    df["processed_text"] = df["utterance"].apply(normalize)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    encoded_model = model.encode(df["processed_text"].tolist(),
                                 normalize_embeddings=True)

    np.savez(out / "intent_embeddings.npz", X=encoded_model)
    (out / "intent_labels.json").write_text(
        json.dumps(df["intent"].tolist(), ensure_ascii=False, indent=2)
    )
    print(f"Saved {len(df)} samples to {out}")


if __name__ == "__main__":
    build_index()
