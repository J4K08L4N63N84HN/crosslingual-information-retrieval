""" Functions for custom metrics.
"""

import datasets
from sklearn.metrics import recall_score, f1_score, log_loss, precision_score

_CITATION = """
"""

_DESCRIPTION = """\
"""

_KWARGS_DESCRIPTION = """
"""


def simple_accuracy(preds, labels):
    """calculate Accuracy

    Args:
        preds: Predictions
        labels: Lables

    Returns:
        int: Accuracy

    """
    return (preds == labels).mean()


def prob2label(prod):
    """Transforms Probability to 0/1 Labels

    Args:
        prod: Probability of prediction (confidence)

    Returns:
        int: 0/1 Labels

    """
    return (prod > 0.5)


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class custom_metric(datasets.Metric):
    """Create Custom Metric for huggingface. Computes F1, Accuracy, Recall, Precision and Log Loss.

    """

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float32"),
                    "references": datasets.Value("float32"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _compute(self, predictions, references):
        """Compute F1, Accuracy, Recall, Precision and Log Loss.

        Args:
            predictions: Predictions of Model (in probability)
            references: Correct Labels

        Returns:
            dict: F1, Accuracy, Recall, Precision and Log Loss

        """
        references = [int(single_labels) for single_labels in references]
        predictions_label = prob2label(predictions)

        return {"accuracy": simple_accuracy(predictions_label, references),
                "recall": recall_score(references, predictions_label),
                "precision": precision_score(references, predictions_label),
                "f1": f1_score(references, predictions_label),
                "log_loss": log_loss(references, predictions, labels=[0, 1])}
