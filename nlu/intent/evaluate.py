from datetime import datetime

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from detector import IntentDetector, find_best_threshold


def evaluate(eval_csv="data/intent_eval.csv", output_file="evaluation_report.txt"):
    df = pd.read_csv(eval_csv).dropna(subset=["text", "intent_true"])
    df["text"] = df["text"].astype(str)

    det = IntentDetector(artifacts_dir="nlu/intent/artifacts")

    # Get predictions with scores
    preds_scores = []
    for t in df["text"]:
        intent, score = det.predict_with_score(t)
        preds_scores.append((intent, score))

    y_true = df["intent_true"].astype(str).tolist()
    y_pred_labels = [p for p, _ in preds_scores]
    y_scores = [s for _, s in preds_scores]

    # Find best threshold
    best_threshold = find_best_threshold(y_true, y_pred_labels, y_scores)

    # Apply best threshold for final predictions
    y_pred_final = []
    for label, score in preds_scores:
        if score >= best_threshold:
            y_pred_final.append(label)
        else:
            y_pred_final.append("unknown")

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_final)

    # Get unique labels for analysis
    unique_labels = sorted(set(y_true + y_pred_final))

    # Calculate per-class statistics
    class_stats = {}
    for label in unique_labels:
        if label != "unknown":
            tp = sum(1 for yt, yp in zip(y_true, y_pred_final) if yt == label and yp == label)
            fp = sum(1 for yt, yp in zip(y_true, y_pred_final) if yt != label and yp == label)
            fn = sum(1 for yt, yp in zip(y_true, y_pred_final) if yt == label and yp != label)
            tn = sum(1 for yt, yp in zip(y_true, y_pred_final) if yt != label and yp != label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            class_stats[label] = {
                'count': y_true.count(label),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }

    # Write comprehensive report to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("INTENT DETECTION EVALUATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Evaluation dataset: {eval_csv}\n")
        f.write(f"Model artifacts: nlu/intent/artifacts\n")
        f.write("=" * 80 + "\n\n")

        # Dataset Overview
        f.write("1. DATASET OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples: {len(y_true)}\n")
        f.write(f"Unique intents: {len(set(y_true))}\n")
        f.write(f"Intent distribution:\n")
        intent_counts = df["intent_true"].value_counts()
        for intent, count in intent_counts.items():
            f.write(f"  - {intent}: {count} ({count / len(y_true) * 100:.1f}%)\n")
        f.write("\n")

        # Threshold Analysis
        f.write("2. THRESHOLD ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best threshold: {best_threshold:.3f}\n")
        f.write(f"Current default threshold: 0.55\n")
        f.write(f"Threshold improvement: {best_threshold - 0.55:.3f}\n\n")

        # Overall Performance
        f.write("3. OVERALL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Unknown predictions: {y_pred_final.count('unknown')}\n")
        f.write(f"Unknown rate: {y_pred_final.count('unknown') / len(y_true) * 100:.1f}%\n")
        f.write(f"Confidence score statistics:\n")
        f.write(f"  - Mean: {sum(y_scores) / len(y_scores):.3f}\n")
        f.write(f"  - Median: {sorted(y_scores)[len(y_scores) // 2]:.3f}\n")
        f.write(f"  - Min: {min(y_scores):.3f}\n")
        f.write(f"  - Max: {max(y_scores):.3f}\n")
        f.write(f"  - Std: {pd.Series(y_scores).std():.3f}\n\n")

        # Classification Report
        f.write("4. CLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_true, y_pred_final, digits=3))
        f.write("\n")

        # Per-Class Performance
        f.write("5. PER-CLASS PERFORMANCE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"{'Intent':<20} {'Count':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<4} {'FP':<4} {'FN':<4}\n")
        f.write("-" * 80 + "\n")
        for label in sorted(class_stats.keys()):
            stats = class_stats[label]
            f.write(
                f"{label:<20} {stats['count']:<8} {stats['precision']:<10.3f} {stats['recall']:<10.3f} {stats['f1']:<10.3f} {stats['tp']:<4} {stats['fp']:<4} {stats['fn']:<4}\n")
        f.write("\n")

        # Confusion Matrix
        f.write("6. CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        f.write("Predicted vs Actual:\n")
        f.write(str(cm))
        f.write("\n\n")

        # Error Analysis
        f.write("7. ERROR ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("Top misclassified examples:\n")
        errors = []
        for i, (text, true_label, pred_label, score) in enumerate(zip(df["text"], y_true, y_pred_final, y_scores)):
            if true_label != pred_label:
                errors.append((text, true_label, pred_label, score))

        # Sort by confidence score (most confident errors first)
        errors.sort(key=lambda x: x[3], reverse=True)

        for i, (text, true_label, pred_label, score) in enumerate(errors[:10]):
            f.write(f"{i + 1}. Text: '{text[:60]}...'\n")
            f.write(f"   True: {true_label} | Predicted: {pred_label} | Score: {score:.3f}\n\n")

        # High Confidence Examples
        f.write("8. HIGH CONFIDENCE EXAMPLES\n")
        f.write("-" * 40 + "\n")
        high_conf_examples = [(t, l, s) for (t, l, s) in zip(df["text"], y_pred_labels, y_scores) if s >= 0.8]
        for i, (text, label, score) in enumerate(high_conf_examples[:10]):
            f.write(f"{i + 1}. Text: '{text[:60]}...'\n")
            f.write(f"   Predicted: {label} | Score: {score:.3f}\n\n")

        # Recommendations
        f.write("9. RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        f.write(f"1. Use threshold {best_threshold:.3f} for optimal F1-score\n")
        f.write(f"2. Consider adding more training data for intents with low recall\n")
        f.write(f"3. Review high-confidence errors for potential data quality issues\n")
        f.write(f"4. Monitor unknown rate: {y_pred_final.count('unknown') / len(y_true) * 100:.1f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    # Also print summary to console
    print(f"=== EVALUATION COMPLETE ===")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Report saved to: {output_file}")
    print(f"Total samples: {len(y_true)}")
    print(f"Unknown rate: {y_pred_final.count('unknown') / len(y_true) * 100:.1f}%")

    # Print top 3 worst performing classes
    worst_classes = sorted(class_stats.items(), key=lambda x: x[1]['f1'])[:3]
    print(f"\nTop 3 worst performing intents:")
    for intent, stats in worst_classes:
        print(f"  - {intent}: F1={stats['f1']:.3f}, Recall={stats['recall']:.3f}")


if __name__ == "__main__":
    evaluate()
