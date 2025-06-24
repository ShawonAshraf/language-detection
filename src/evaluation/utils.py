from typing import Callable

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm

from classifier import DocumentClassifier, NgramProfiler, calculate_distance


def infer(clf: DocumentClassifier, docs: list[str]) -> list[str]:
    """Infer the language of the docs"""

    predictions = []
    for _, doc in tqdm(enumerate(docs), total=len(docs)):
        predictions.append(clf.predict(doc))

    return predictions


def evaluate(
    predictions: list[str], labels: list[str]
) -> tuple[float, float, float, float, str]:
    """Evaluate the predictions"""
    accuracy = accuracy_score(predictions, labels)
    precision = precision_score(predictions, labels, average="macro")
    recall = recall_score(predictions, labels, average="macro")
    f1 = f1_score(predictions, labels, average="macro")

    # also get the classification report
    report = classification_report(predictions, labels)

    return accuracy, precision, recall, f1, report


profile_sizes = [50, 100, 200, 300, 400]


def explore_profile_size(
    test_texts: list[str],
    test_languages: list[str],
    train_texts: list[str],
    train_languages: list[str],
    profile_sizes: list[int] = profile_sizes,
    distance_fn: Callable = calculate_distance,
) -> None:
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []

    for profile_size in profile_sizes:
        classifier = DocumentClassifier(
            profiler=NgramProfiler(),
            profile_size=profile_size,
            distance_fn=calculate_distance,
        )

        classifier.fit(train_texts, train_languages)

        predictions = infer(classifier, test_texts)
        accuracy, precision, recall, f1, _ = evaluate(predictions, test_languages)

        f1_scores.append(f1)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

    plt.plot(profile_sizes, f1_scores, label="F1 Score")
    plt.plot(profile_sizes, accuracies, label="Accuracy")
    plt.plot(profile_sizes, precisions, label="Precision")
    plt.plot(profile_sizes, recalls, label="Recall")
    plt.legend()
    plt.show()
