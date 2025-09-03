from pathlib import Path

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import normalize


def find_best_threshold(y_true, y_pred_labels, y_scores):
    best_threshold = 0.0
    best_f1 = 0.0
    for threshold in np.arange(0.0, 1.01, 0.01):
        y_pred = [label if score >= threshold else "unknown" for label, score in zip(y_pred_labels, y_scores)]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp and yt != "unknown")
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != yp and yp != "unknown")
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != yp and yt != "unknown")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


class IntentDetector:
    def __init__(self, artifacts_dir="nlu/intent/artifacts", threshold=0.55):
        artifacts_path = Path(artifacts_dir)
        self.embeddings = np.load(artifacts_path / "intent_embeddings.npz")["X"]
        self.labels = json.loads((artifacts_path / "intent_labels.json").read_text())
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.threshold = threshold

    def predict(self, text: str) -> str:
        text = normalize(text)
        if not text:
            return "unknown"
        text_emb = self.model.encode([text], normalize_embeddings=True)
        sims = cosine_similarity(text_emb, self.embeddings)[0]
        best_idx = np.argmax(sims)
        if sims[best_idx] >= self.threshold:
            return self.labels[best_idx]
        return "unknown"

    def predict_with_score(self, text: str):
        text = normalize(text)
        if not text:
            return "unknown", 0.0
        text_emb = self.model.encode([text], normalize_embeddings=True)
        sims = cosine_similarity(text_emb, self.embeddings)[0]
        best_idx = np.argmax(sims)
        best_score = float(sims[best_idx])
        best_label = self.labels[best_idx]
        return best_label, best_score


if __name__ == "__main__":
    detector = IntentDetector()
    while True:
        text = input("Enter your utterance: ")
        intent = detector.predict(text)
        print(f"Predicted intent: {intent}")
