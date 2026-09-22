import numpy as np


def compute_seqeval(p, label_list, metric):

    predictions, labels = p

    predictions = np.argmax(predictions, axis=2)

    true_predictions = [

        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]

        for prediction, label in zip(predictions, labels)

    ]

    true_labels = [

        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]

        for prediction, label in zip(predictions, labels)

    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {

        "precision": results["overall_precision"],

        "recall": results["overall_recall"],

        "f1": results["overall_f1"],

        "accuracy": results["overall_accuracy"],

    }


def compute_f1(p, label_list, metric):

    predictions, labels = p

    predictions = np.argmax(predictions, axis=2)

    true_predictions = [

        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]

        for prediction, label in zip(predictions, labels)

    ]

    true_labels = [

        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]

        for prediction, label in zip(predictions, labels)

    ]

    return metric.compute(predictions=[i for j in true_predictions for i in j], references=[i for j in true_labels for i in j])

def compute_f1_flat(p, label_list, metric):

    predictions, labels = p

    predictions = np.argmax(predictions, axis=1)

    true_predictions = [

        [label_list[p] for (p, l) in zip(predictions, labels) if l != -100]

    ]

    true_labels = [

        [label_list[l] for (p, l) in zip(predictions, labels) if l != -100]

    ]

    return metric.compute(predictions=[i for j in true_predictions for i in j], references=[i for j in true_labels for i in j])


def binary_prf(predictions, labels, positive: int = 1) -> dict:
    """Precision/recall/F1 of the positive class over positions whose label != -100."""
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    keep = labels != -100
    predictions, labels = predictions[keep], labels[keep]
    tp = int(np.sum((predictions == positive) & (labels == positive)))
    fp = int(np.sum((predictions == positive) & (labels != positive)))
    fn = int(np.sum((predictions != positive) & (labels == positive)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": float(np.mean(predictions == labels)) if len(labels) else 0.0,
            "support": tp + fn}


def compute_prf(p) -> dict:
    """Trainer metric for padded (batch, seq, labels) logits; also works for flat (words, labels) logits."""
    logits, labels = p
    return binary_prf(np.argmax(logits, axis=-1), labels)


def aggregate_runs(runs: list) -> dict:
    """Mean/std/min/max over runs (e.g. seeds) for every numeric metric of a flat metric dict."""
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    summary = {}
    for key in keys:
        values = np.array([run[key] for run in runs if key in run], dtype=float)
        summary[key] = {"mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "n": int(len(values))}
    return summary
