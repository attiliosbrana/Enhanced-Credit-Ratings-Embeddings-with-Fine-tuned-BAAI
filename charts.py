import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(ground_truth, predictions, pos_label):
    return {
        "Accuracy": accuracy_score(ground_truth, predictions),
        "Precision": precision_score(ground_truth, predictions, pos_label=pos_label),
        "Recall": recall_score(ground_truth, predictions, pos_label=pos_label),
        "F1 Score": f1_score(ground_truth, predictions, pos_label=pos_label),
    }


def plot_performance_bar(metrics_values, model_name):
    metrics = list(metrics_values.keys())
    values = list(metrics_values.values())
    plt.figure(figsize=(10, 6))
    sns.barplot(x=metrics, y=values, palette="viridis")
    plt.ylim(0, 1)
    plt.title(f"Performance Metrics for {model_name}")
    plt.show()


def plot_confusion_matrix(ground_truth, predictions, model_name, category1, category2):
    conf_matrix = confusion_matrix(
        ground_truth, predictions, labels=[category1, category2]
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=[category1, category2],
        yticklabels=[category1, category2],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()


def plot_all_models_comparison(models_metrics):
    metrics = list(next(iter(models_metrics.values())).keys())
    model_names = list(models_metrics.keys())

    plt.figure(figsize=(12, 7))

    bar_width = 0.2
    for idx, metric in enumerate(metrics):
        values = [models_metrics[model_name][metric] for model_name in model_names]
        positions = [i + bar_width * idx for i in range(len(model_names))]
        plt.bar(positions, values, width=bar_width, label=metric)

    plt.xticks([i + bar_width for i in range(len(model_names))], model_names)
    plt.legend(loc="best")
    plt.ylim(0, 1)
    plt.title("Model Comparison based on Metrics")
    plt.show()
