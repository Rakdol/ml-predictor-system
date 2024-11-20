import matplotlib.pyplot as plt


def plot_evaluated_model_result(y_test, y_pred):
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))

    step = 168
    ax_i = 0
    for i in range(0, y_test.shape[0], step):
        if ax_i > 8:
            break
        axes[ax_i].plot(y_test[i : i + step].to_numpy(), label="Ground Truth")
        axes[ax_i].plot(y_pred[i : i + step], label="Predictions")
        ax_i += 1
    plt.legend()
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_metric_results(eval_metrics: dict):
    # Extract keys and values
    labels = list(eval_metrics.keys())
    values = list(eval_metrics.values())

    # Plotting
    fig = plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color="skyblue", edgecolor="black")
    plt.title("Comparison of Evaluation Metrics", fontsize=16)
    plt.ylabel("Metric Value", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale("log")  # Use logarithmic scale for better comparison
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.legend()
    plt.tight_layout()
    plt.close(fig)
    return fig
