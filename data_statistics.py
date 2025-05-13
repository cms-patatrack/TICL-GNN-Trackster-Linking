import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score


def print_dataset_statistics(trainDataset, epsilon=0.1):

    num_events = len(trainDataset)
    print(f"Number of events in training dataset: {num_events}")

    num_nodes, num_edges, num_neg, num_pos = 0, 0, 0, 0
    for ev in trainDataset:
        num_nodes += ev.num_nodes
        num_edges += len(ev.y)
        num_pos += (ev.y > epsilon).sum()
        num_neg += (ev.y <= epsilon).sum()

    print(f"Number of nodes: {num_nodes}")
    print(f"Mean Number of nodes: {num_nodes/num_events}")
    print(f"Number of edges: {num_edges}")
    print(f"Mean Number of edges: {num_edges/num_events}")
    print(f"Number of positive edges: {num_pos}")
    print(f"Mean Number of positive edges: {num_pos/num_events}")
    print(f"Number of negative edges: {num_neg}")
    print(f"Mean Number of negative edges: {num_neg/num_events}")


def classification_threshold_scores(scores, ground_truth, ax, threshold_step=0.05, plot=True, save=False, output_folder=None, filename=None):
    """
    Plots and saves the figure of the dependancy of th eaccuracy, True Positive rate (TPR) and 
    True Negative rate (TNR) on the value of the classification threshold.
    """
    y = (ground_truth > 0).astype(int)
    thresholds = np.arange(0, 1 + threshold_step, threshold_step)
    ACC, TNR, TPR, F1 = [], [], [], []
    for threshold in thresholds:
        prediction = scores > threshold

        TN, FP, FN, TP = confusion_matrix(y, prediction).ravel()
        ACC.append((TP+TN)/(TN + FP + FN + TP))
        TNR.append(TN/(TN+FP))
        TPR.append(TP/(TP+FN))
        F1.append(f1_score(y, prediction))

    # Saving the figure of the classification threshold plot
    if plot or save:
        ax.plot(thresholds, ACC, 'go-', label='ACC', linewidth=2)
        ax.plot(thresholds, TNR, 'bo-', label='TNR', linewidth=2)
        ax.plot(thresholds, TPR, 'ro-', label='TPR', linewidth=2)
        ax.plot(thresholds, F1, 'mo-', label='F1', linewidth=2)
        ax.set_xlabel("Threshold", fontsize=15)
        ax.title(
            "Accuracy / TPR / TNR / F1 based on the classification threshold value", fontsize=16)
        ax.legend()

        if save and output_folder is not None and filename is not None:
            ax.savefig(f"{output_folder}/{filename}", dpi=300,
                       bbox_inches='tight', transparent=True)

    return ACC, TNR, TPR, F1, thresholds
