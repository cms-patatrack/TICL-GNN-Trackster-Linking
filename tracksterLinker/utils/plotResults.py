from tqdm import tqdm
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import class_likelihood_ratios, precision_recall_fscore_support

import seaborn as sn
import mplhep as hep


plt.style.use(hep.style.CMS)
"""Testing of the trained models."""


# statistical analysis of prediction results
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

    # Saving and plotting the figure of the classification threshold plot
    if plot or save:
        ax.plot(thresholds, ACC, 'go-', label='ACC', linewidth=2)
        ax.plot(thresholds, TNR, 'bo-', label='TNR', linewidth=2)
        ax.plot(thresholds, TPR, 'ro-', label='TPR', linewidth=2)
        ax.plot(thresholds, F1, 'mo-', label='F1', linewidth=2)
        ax.set_xlabel("Threshold", fontsize=15)
        ax.set_title("Accuracy / TPR / TNR / F1 based on the classification threshold value", fontsize=16)
        ax.legend()

        if save and output_folder is not None and filename is not None:
            ax.savefig(f"{output_folder}/{filename}", dpi=300,
                       bbox_inches='tight', transparent=True)

    return ACC, TNR, TPR, F1, thresholds


def plot_validation_results(pred, y, save=True, output_folder=None, file_suffix=None, ax=None):
    save = save and output_folder is not None and file_suffix is not None

    if ax is None:
        print("crease")
        fig, ax = plt.subplots(6, 2)
        fig.set_figheight(30)
        fig.set_figwidth(40)

    _, TNR, TPR, _, thresholds = classification_threshold_scores(pred, y, ax[0, 0], threshold_step=0.05)
    plot_confusion_matrix(pred, y, ax[2, :])

    best_threshold = get_best_threshold(TNR, TPR, thresholds)
    plot_prediction_distribution(pred, y, ax[3:6, :], thres=best_threshold)
    plot_edge_distribution(pred, y, ax[1, :])
    plot_roc_curve(pred, y, ax[0, 1], best_threshold)

    print_acc_scores(pred, y, best_threshold)

    # TODO: Save plots and data
    # if save and output_folder is not None and file_suffix is not None:
    #     ax.savefig(f"{output_folder}/{file_suffix}.png", dpi=300,
    #                bbox_inches='tight', transparent=True)


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


def get_best_threshold(TNR, TPR, thresholds, epsilon=0.02, default=0.65):
    # Find the threshold for which TNR and TPR intersect

    for i in range(len(thresholds)-1):
        if abs(TNR[i] - TPR[i]) < epsilon:
            return round(thresholds[i], 3)

        if TNR[i] - TPR[i] < 0 and TNR[i+1] - TPR[i+1] >= epsilon:
            return round(0.5*(thresholds[i] + thresholds[i+1]), 3)

    print("Choose a default threshold...")
    return default


def plot_edge_distribution(pred, y, axes):
    y_discrete = (y > 0).astype(int)
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


def print_acc_scores(pred, y, thres=0.65):
    pred_discrete = (pred > thres).astype(int)
    y_discrete = (y > 0).astype(int)

    TN, FP, FN, TP = confusion_matrix(y_discrete, pred_discrete).ravel()
    tot = TN + FP + FN + TP

    print(f"Scores for Classification with Threshold: {thres}.")
    print(f"F1 score: {f1_score(y_discrete, pred_discrete):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_discrete, pred_discrete):.3f}")
    print(f"Accuracy: {(TP+TN)/tot:.3f}")
    print(f"Precision: {TP/(TP+FP)*100:.4f}")
    print(f"Recall: {TP/(TP+FN)*100:.4f}")
    print(f"Negative predictive value: {TN/(TN+FN)*100:.4f}")
    print(f"True negative rate: {TN/(TN+FP)*100:.4f}")

    sample_weight = compute_sample_weight(class_weight='balanced', y=y_discrete)
    prec_w, rec_w, fscore_w, _ = precision_recall_fscore_support(y_discrete, pred_discrete, sample_weight=sample_weight)

    print(f"Precision Weighted: {prec_w}")
    print(f"Recall Weighted: {rec_w}")
    print(f"F1 score Weighted: {fscore_w}")

    pos_lr, neg_lr = class_likelihood_ratios(y_discrete, pred_discrete, raise_warning=False)
    print(f"Positive Likelihood Ratio: {pos_lr}")
    print(f"Negative Likelihood Ratio: {neg_lr}")


def plot_roc_curve(pred, y, ax, thres=0.65):
    y_discrete = (y > 0).astype(int)
    fpr, tpr, _ = roc_curve(y_discrete, pred)

    ax.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % auc(fpr, tpr))
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC. Threshold: {thres}", fontsize=14)
    ax.legend(loc="lower right")


def plot_confusion_matrix(pred, y, axes, thres=0.65):

    pred_discrete = (pred > thres).astype(int)
    y_discrete = (y > 0).astype(int)
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_discrete)

    cf_matrix_w_norm = confusion_matrix(y_discrete, pred_discrete, sample_weight=sample_weight, normalize='all')
    cf_matrix = confusion_matrix(y_discrete, pred_discrete, normalize='all')

    # Normal Confusion Matrix
    sn.heatmap(cf_matrix, annot=True, cbar=False, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Threshold: {thres}", fontsize=14)

    # Weighted Cofusion Matrix
    sn.heatmap(cf_matrix_w_norm, annot=True, cbar=False, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"Weighted. Threshold: {thres}", fontsize=14)


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

    # Thresholded labels in linear and log scales
    thresholded = (pred > thres).astype(int)
    axes[2, 0].set_title(f'Predicted Edge Labels for thr: {thres}', fontsize=14)
    axes[2, 0].hist(thresholded, bins=30)

    axes[2, 1].set_title(f'Predicted Edge Labels Log for thr: {thres}', fontsize=14)
    axes[2, 1].hist(thresholded, bins=30)
    axes[2, 1].set_yscale('log')
    # ------------------------
