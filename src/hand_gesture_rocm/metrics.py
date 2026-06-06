"""Classification metrics without adding training-time framework coupling."""

from __future__ import annotations

from .constants import CLASS_NAMES, NUM_CLASSES


def empty_confusion_matrix(num_classes: int = NUM_CLASSES) -> list[list[int]]:
    """Create a zero-filled confusion matrix."""
    return [[0 for _ in range(num_classes)] for _ in range(num_classes)]


def update_confusion_matrix(
    matrix: list[list[int]],
    targets: list[int] | tuple[int, ...],
    predictions: list[int] | tuple[int, ...],
) -> None:
    """Update a confusion matrix in place."""
    for target, prediction in zip(targets, predictions):
        matrix[int(target)][int(prediction)] += 1


def classification_summary(
    matrix: list[list[int]],
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict:
    """Compute accuracy, macro metrics and per-class metrics."""
    num_classes = len(class_names)
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(num_classes))

    per_class = []
    precisions = []
    recalls = []
    f1_scores = []

    for index, name in enumerate(class_names):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(num_classes) if row != index)
        false_negative = sum(matrix[index][column] for column in range(num_classes) if column != index)
        support = sum(matrix[index])

        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1_score = _safe_divide(2 * precision * recall, precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        per_class.append(
            {
                "class_id": index,
                "class_name": name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": support,
            }
        )

    return {
        "accuracy": _safe_divide(correct, total),
        "precision_macro": _mean(precisions),
        "recall_macro": _mean(recalls),
        "f1_macro": _mean(f1_scores),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "support": total,
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
