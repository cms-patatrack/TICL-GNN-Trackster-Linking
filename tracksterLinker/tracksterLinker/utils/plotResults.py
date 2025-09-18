from tqdm import tqdm
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, balanced_accuracy_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import class_likelihood_ratios, precision_recall_fscore_support, accuracy_score
from tracksterLinker.utils.dataStatistics import weighted_precision_recall_f1, weighted_precision_recall_f1_from_precalc

import seaborn as sn
"""Testing of the trained models."""


# statistical analysis of prediction results
def classification_threshold_scores(scores, ground_truth, ax=None, threshold_step=0.05, plot=True, save=False, output_folder=None, filename=None, weight=None):
    """
    Plots and saves the figure of the dependancy of th eaccuracy, True Positive rate (precision) and 
    True Negative rate (recall) on the value of the classification threshold.
    """
    y = (ground_truth > 0).astype(int)
    thresholds = np.arange(0, 1 + threshold_step, threshold_step)
    accuracy, recall, precision, F1 = [], [], [], []
    for threshold in thresholds:
        prediction = (scores > threshold).astype(float)
        prec, rec, f1, support = precision_recall_fscore_support(y, prediction, sample_weight=weight, average='binary', pos_label=1, zero_division=0.0)

        accuracy.append(accuracy_score(y, prediction, sample_weight=weight))
        recall.append(rec)
        precision.append(prec)
        F1.append(f1)

    # Saving and plotting the figure of the classification threshold plot
    if plot or save:
        ax.plot(thresholds, accuracy, 'go-', label='accuracy', linewidth=2)
        ax.plot(thresholds, recall, 'bo-', label='recall', linewidth=2)
        ax.plot(thresholds, precision, 'ro-', label='precision', linewidth=2)
        ax.plot(thresholds, F1, 'mo-', label='F1', linewidth=2)
        ax.set_xlabel("Threshold", fontsize=15)
        ax.set_title("Accuracy / precision / recall / F1 based on the classification threshold value", fontsize=16)
        ax.legend()

        # Save Data not Image
        # if save and output_folder is not None and filename is not None:
        #     ax.savefig(f"{output_folder}/{filename}_class_threshold.png", dpi=300,
        #                bbox_inches='tight', transparent=True)

    return accuracy, recall, precision, F1, thresholds

def plot_binned_validation_results(pred, y, weights, thres=0.65, output_folder=None, file_suffix=None):
    pred_discrete = (pred > thres).float()
    y_discrete = (y > 0).float()
    bin_edges = [weights.min(), 5, 10, 20, 50, 100, 300, weights.max()]
    
    bin_vals = []
    for i in range(len(bin_edges)-1):
        in_bin = (weights >= bin_edges[i]) & (weights < bin_edges[i+1])
        if in_bin.sum() > 0:  # avoid empty bins
            acc = accuracy_score(y_discrete[in_bin].numpy(), pred_discrete[in_bin].numpy())
            prec = precision_score(y_discrete[in_bin].numpy(), pred_discrete[in_bin].numpy())
            rec = recall_score(y_discrete[in_bin].numpy(), pred_discrete[in_bin].numpy())
            bin_vals.append((float(bin_edges[i]), float(bin_edges[i+1]), acc, prec, rec))
            plot_validation_results(pred[in_bin], y[in_bin], thres, weight=weights[in_bin], file_suffix=f"{file_suffix}_bin_{float(bin_edges[i])}_{float(bin_edges[i+1])}", output_folder=output_folder)

    for low, high, acc, prec, rec in bin_vals:
        if acc != None:
            print(f"Bin {low:.2f} - {high:.2f}: Accuracy {acc:.4f}, Precision {prec:.4f}, Recall {rec:.4f}")

def plot_validation_results(pred, y, save=True, thres=0.6, output_folder=None, file_suffix=None, ax=None, weight=None):
    save = save and output_folder is not None and file_suffix is not None

    if ax is None:
        print("Create Plot for Validation Results")
        fig, ax = plt.subplots(5, 2)
        fig.set_figheight(30)
        fig.set_figwidth(40)


    pred = pred.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    
    # Plots without predictiohn threshold
    _, recall, precision, _, thresholds = classification_threshold_scores(pred, y, ax[0, 0], threshold_step=0.05, weight=weight)
    plot_roc_curve(pred, y, ax[0, 1], weight=weight)
    plot_edge_distribution(pred, y, ax[1, :])
    
    # Plots with fixed threshold
    best_threshold = thres

    pred_discrete = (pred > thres).astype(int)
    y_discrete = (y > 0).astype(int)

    plot_confusion_matrix(pred_discrete, y_discrete, ax[2, 0], thres=best_threshold)
    plot_confusion_matrix(pred_discrete, y_discrete, ax[2, 1], thres=best_threshold, weight=weight)
    plot_prediction_distribution(pred, y, ax[3:5, :], thres=best_threshold)

    # TODO: Save data, even with ax provided
    if save and fig is not None and output_folder is not None and file_suffix is not None:
        fig.savefig(f"{output_folder}/{file_suffix}_validation_results.png", dpi=300,
                   bbox_inches='tight', transparent=True)

    return best_threshold

def get_model_prediction(model, testLoader, prepare_network_input_data=None,
                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Gets model predictions on test edges.
    model: the trained network.
    testLoader: DataLoader of already pre-processed data.
    """

    model.to(device)
    predictions, truth = [], []

    for sample in tqdm(testLoader, desc="Getting model predictions"):
        sample = sample.to(device)

        if prepare_network_input_data is not None:
            inputs = prepare_network_input_data(sample.x, sample.edge_index)
        else:
            inputs = (sample.x, sample.edge_index)

        link_pred, emb = model(*inputs)
        predictions.append(link_pred.cpu().detach().numpy())
        truth.append(sample.edge_label.cpu().detach().numpy())
    return truth, predictions


def get_best_threshold(y, pred, weight, threshold_step=0.05, epsilon=0.02, default=0.65):
    # Find the threshold for which recall and precision intersect
    thresholds = np.arange(0, 1, threshold_step)
    _, _, _, f1 = weighted_precision_recall_f1(y, pred, weight)
    y_discrete = (y > 0).int()

    pred_discrete = (pred >= thresholds[0]).int()
    _, _, _, f1 = weighted_precision_recall_f1(y_discrete, pred_discrete, weight)

    pred_discrete = (pred >= thresholds[1]).int()
    _, _, _, next_f1 = weighted_precision_recall_f1(y_discrete, pred_discrete, weight)

    for i in range(len(thresholds)-1):
        old_f1 = f1
        f1 = next_f1

        pred_discrete = (pred >= thresholds[i+1]).int()
        _, _, _, next_f1 = weighted_precision_recall_f1(y_discrete, pred_discrete, weight)
        if (old_f1 < f1 and f1 > next_f1):
            return thresholds[i]

    print("Choose a default threshold...")
    return default


def plot_edge_distribution(pred, y, axes):
    y_discrete = (y > 0).astype(float)
    true_pred = pred[y_discrete == 1]
    false_pred = pred[y_discrete != 1]

    bins = 100
    axes[0].hist(false_pred, bins=bins, density=1, label="False Edges", histtype='step')
    axes[0].hist(true_pred, bins=bins, density=1, label="True Edges", histtype='step')
    axes[0].legend(loc="upper center")  # loc="upper left")
    axes[0].set_title("True and False Edge Prediction Distribution", fontsize=14)
    axes[0].set_xlabel("Predicted score", fontsize=14)
    axes[0].set_ylabel('Probability [%]', fontsize=14)

    axes[1].hist(pred, bins=bins, label="All predictions")
    axes[1].legend()
    axes[1].set_title("Edge Prediction Distribution", fontsize=14)
    axes[1].set_xlabel("Predicted score", fontsize=14)
    axes[1].set_ylabel('Counts', fontsize=14)


def print_acc_scores_from_precalc(tp, fp, fn, tn):
    prec, rec, spec, f1 = weighted_precision_recall_f1_from_precalc(tp, fp, fn, tn)
    print(f"Percentage of edges classified to merge: {(tp + fp)/(tp + fp + fn + tn)}")
    print(f"Percentage of true edges which should be merged: {(tp + fn)/(tp + fp + fn + tn)}")

    print(f"F1 score: {f1:.3f}")
    print(f"Accuracy: {tp / (tp + fp + fn + tn):.3f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Specificity: {spec:.4f}")

    print(f"Positive Likelihood Ratio: {rec / (1 - spec + 1e-8)}; x in [1.0, inf]; Higher is better; 1.0 means meaningless ML;")
    print(f"Negative Likelihood Ratio: {(1 - rec) / (spec + 1e-8)}; x in [0.0, 1.0]; Lower is better; 1.0 means menaingless ML;")

def print_binned_acc_scores(pred, y, weights, thres=0.65):
    pred_discrete = (pred > thres).float()
    y_discrete = (y > 0).float()
    bin_edges = [weights.min(), 5, 10, 20, 50, 100, 300, weights.max()]
    
    bin_vals = []
    for i in range(len(bin_edges)-1):
        in_bin = (weights >= bin_edges[i]) & (weights < bin_edges[i+1])
        if in_bin.sum() > 0:  # avoid empty bins
            acc = accuracy_score(y_discrete[in_bin].numpy(), pred_discrete[in_bin].numpy())
            prec = precision_score(y_discrete[in_bin].numpy(), pred_discrete[in_bin].numpy())
            rec = recall_score(y_discrete[in_bin].numpy(), pred_discrete[in_bin].numpy())
            print(f"Bin {float(bin_edges[i]):.2f} - {float(bin_edges[i+1]):.2f}: Accuracy {acc:.4f}, Precision {prec:.4f}, Recall {rec:.4f}")
        else:
            print(f"Bin {float(bin_edges[i]):.2f} - {float(bin_edges[i+1]):.2f}: Empty")


def plot_roc_curve(pred, y, ax, weight=None):
    y_discrete = (y > 0).astype(int)
    fpr, tpr, _ = roc_curve(y_discrete, pred, sample_weight=weight)

    fpr = np.round(fpr, decimals=8)
    _, unique_indices = np.unique(fpr, return_index=True)
    fpr = fpr[unique_indices]
    tpr = tpr[unique_indices]

    auc_val = auc(fpr, tpr)

    ax.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % auc_val)
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC", fontsize=14)
    ax.legend(loc="lower right")


def plot_confusion_matrix(pred_discrete, y_discrete, ax, thres=0.65, weight=None):
    cf_matrix = confusion_matrix(y_discrete, pred_discrete, sample_weight=weight, normalize='all')
    # Normal Confusion Matrix
    sn.heatmap(cf_matrix, annot=True, cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if (weight is not None):
        ax.set_title(f"Weighted. Threshold: {thres}", fontsize=14)
    else:
        ax.set_title(f"Threshold: {thres}", fontsize=14)


def plot_prediction_distribution(pred, y, axes, thres=0.65):
    # Predictions in linear and log scales
    axes[0, 0].set_title('Prediction distribution', fontsize=14)
    axes[0, 0].hist(pred, bins=30)

    axes[0, 1].set_title('Prediction distribution Log', fontsize=14)
    axes[0, 1].hist(pred, bins=30)
    axes[0, 1].set_yscale('log')
    # ------------------------

    # Truth labels in linear and log scales
    axes[1, 0].hist(y, bins=30)
    axes[1, 0].set_title('True Edge Labels', fontsize=14)

    axes[1, 1].hist(y, bins=30)
    axes[1, 1].set_title('True Edge Labels Log', fontsize=14)
    axes[1, 1].set_yscale('log')
    # ------------------------
