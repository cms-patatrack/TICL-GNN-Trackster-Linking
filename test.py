import argparse
from tqdm import tqdm
import importlib.util

import pickle
import matplotlib.pyplot as plt
import numpy as np

import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import class_likelihood_ratios, precision_recall_fscore_support

import mplhep as hep

# plt.style.use(hep.style.CMS)
"""Testing of the trained models."""


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


def evaluate_clusters(dataset, predictions=None, thr=None, geometric=False, numevents=1000, PU=False):
    """
    Calculates homogenity, completeness, v-measure, adjusted random index and mutual info of the clustering.
    Predictions are the list of lists containing network scores for edges.
    """
    hom, compl, vmeas, randind, mutinfo = [], [], [], [], []

    for i in tqdm(range(numevents)):
        event = dataset[i]
        if not geometric:
            components, predicted_cluster_ids = find_connected_components(
                event, predictions[i], edge_index=None, thr=thr, PU=PU)
        else:
            predicted_cluster_ids = event.candidate_match.cpu().detach().numpy()

        t_match = event.best_simTs_match.cpu().detach().numpy()

        hom.append(metrics.homogeneity_score(t_match, predicted_cluster_ids))
        compl.append(metrics.completeness_score(
            t_match, predicted_cluster_ids))
        vmeas.append(metrics.v_measure_score(t_match, predicted_cluster_ids))
        randind.append(metrics.adjusted_rand_score(
            t_match, predicted_cluster_ids))
        mutinfo.append(metrics.adjusted_mutual_info_score(
            t_match, predicted_cluster_ids))

    return hom, compl, vmeas, randind, mutinfo


def classification_thresholds_plot(scores, ground_truth, threshold_step=0.05, output_folder=None, epoch=None):
    """
    Plots and saves the figure of the dependancy of th eaccuracy, True Positive rate (TPR) and 
    True Negative rate (TNR) on the value of the classification threshold.
    """
    thresholds = np.arange(0, 1 + threshold_step, threshold_step)
    ACC, TNR, TPR, F1 = [], [], [], []
    for threshold in thresholds:

        prediction = scores > threshold

        TN, FP, FN, TP = confusion_matrix(ground_truth, prediction).ravel()
        ACC.append((TP+TN)/(TN + FP + FN + TP))
        TNR.append(TN/(TN+FP))
        TPR.append(TP/(TP+FN))
        F1.append(f1_score(ground_truth, prediction))

    # Saving the figure of the classification threshold plot
    fig = plt.figure(figsize=(9, 6))
    plt.plot(thresholds, ACC, 'go-', label='ACC', linewidth=2)
    plt.plot(thresholds, TNR, 'bo-', label='TNR', linewidth=2)
    plt.plot(thresholds, TPR, 'ro-', label='TPR', linewidth=2)
    plt.plot(thresholds, F1, 'mo-', label='F1', linewidth=2)
    plt.xlabel("Threshold", fontsize=15)
    plt.title(
        "Accuracy / TPR / TNR / F1 based on the classification threshold value", fontsize=16)
    plt.legend()

    if output_folder is not None:
        f_name = f"classification_thresholds_epoch_{epoch}.png" if epoch is not None else "classification_thresholds.png"
        plt.savefig(f"{output_folder}/{f_name}", dpi=300,
                    bbox_inches='tight', transparent=True)
        plt.show()

    return TNR, TPR, thresholds


def get_best_threshold(TNR, TPR, thresholds, epsilon=0.02, default=0.65):
    # Find the threshold for which TNR and TPR intersect

    for i in range(len(thresholds)-1):
        if abs(TNR[i] - TPR[i]) < epsilon:
            return round(thresholds[i], 3)

        if TNR[i] - TPR[i] < 0 and TNR[i+1] - TPR[i+1] >= epsilon:
            return round(0.5*(thresholds[i] + thresholds[i+1]), 3)
    print("Chosen a default threshold...")
    return default


def save_pred(pred_flat, lab_flat, epoch=0, out_folder=None):

    true_pred = pred_flat[lab_flat == 1]
    false_pred = pred_flat[lab_flat != 1]

    bins = 100
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_subplot(121)
    ax1.hist(false_pred, bins=bins, density=1,
             label="False predictions", histtype='step')
    ax1.hist(true_pred, bins=bins, density=1,
             label="True predictions", histtype='step')
    ax1.legend(loc="upper center")  # loc="upper left")
    # ax1.set_yscale('log')

    ax2 = fig.add_subplot(122)
    ax2.hist(pred_flat, bins=bins, label="All predictions")
    ax2.legend()

    ax1.set_title("True and False edge prediction distribtion", fontsize=15)
    ax1.set_xlabel("Predicted score", fontsize=14)
    ax1.set_ylabel('Probability [%]', fontsize=14)
    ax2.set_title("Edge prediction distribtion", fontsize=15)
    ax2.set_xlabel("Predicted score", fontsize=14)
    ax2.set_ylabel('Counts', fontsize=14)
    plt.show()

    if out_folder is not None:
        fig.savefig(f'{out_folder}/edge-pred-distributions-epoch-{epoch+1}.pdf',
                    dpi=300, bbox_inches='tight', transparent=True)
        fig.savefig(f'{out_folder}/edge-pred-distributions-eppoch-{epoch+1}.png',
                    dpi=300, bbox_inches='tight', transparent=True)


def test(truth, scores, classification_threshold=0.7, output_folder=None, epoch=None):

    results = {}
    print(scores)
    predictions = scores > classification_threshold
    sample_weight = compute_sample_weight(class_weight='balanced', y=truth)

    cf_matrix = confusion_matrix(truth, predictions)
    print(f"Confusion matrix:\n{cf_matrix}\n")

    cf_matrix_w_norm = confusion_matrix(
        truth, predictions, sample_weight=sample_weight, normalize='all')
    print(f"Confusion matrix weighted:\n{cf_matrix_w_norm}\n")

    TN, FP, FN, TP = cf_matrix.ravel()
    print(f"TN: {TN} \t FN: {FN} \t TP: {TP} \t FP: {FP}")
    results["F1"] = f1_score(truth, predictions)
    results["BA"] = balanced_accuracy_score(truth, predictions)

    # Sensitivity, hit rate, recall, or true positive rate
    results["TPR"] = TP/(TP+FN)
    # Specificity or true negative rate
    results["TNR"] = TN/(TN+FP)
    # Precision or positive predictive value
    results["PPV"] = TP/(TP+FP)
    # Negative predictive value
    results["NPV"] = TN/(TN+FN)
    # Fall out or false positive rate
    results["FPR"] = FP/(FP+TN)
    # False negative rate
    results["FNR"] = FN/(TP+FN)
    # False discovery rate
    results["FDR"] = FP/(TP+FP)

    tot = TN + FP + FN + TP
    ACC = (TP+TN)/tot
    # normalized to total edges in test dataset
    print(f"Confusion matrix scaled:\n{cf_matrix/tot}\n")
    print(f"Accuracy: {ACC:.4f}")
    print(f"Precision: {results['PPV']:.4f}")
    print(f"Negative predictive value: {results['NPV']:.4f}")
    print(
        f"Recall: Correctly classifying {results['TPR']*100:.4f} % of positive edges")
    print(
        f"True negative rate: Correctly classifying {results['TNR']*100:.4f} % of all negative edges")
    print(f"F1 score: {results['F1']:.4f}")

    prec_w, rec_w, fscore_w, _ = precision_recall_fscore_support(
        truth, predictions, sample_weight=sample_weight)
    print(prec_w, rec_w, fscore_w)
    print(f"Balanced accuracy: {results['BA']:.4f}")
    print(f"Precision weighted: {prec_w}")
    print(f"Recall weighted: {rec_w}")
    print(f"F1 score weighted: {fscore_w}")

    # computes the positive and negative likelihood ratios (LR+, LR-) to assess the predictive
    # power of a binary classifier. As we will see, these metrics are independent of the proportion between
    # classes in the test set, which makes
    # them very useful when the available data for a study has a different class proportion than the target application.
    pos_lr, neg_lr = class_likelihood_ratios(
        truth, predictions, raise_warning=False)
    print(
        f"positive_likelihood_ratio: {pos_lr}, negative_likelihood_ratio: {neg_lr}")

    max_el = max(np.amax(cf_matrix/tot), np.amax(cf_matrix_w_norm))

    # plot confusion matrix
    fig, px = plt.subplots(1, 2, figsize=(8, 4))
    plt.set_cmap("viridis")
    px[0].set_xlabel("Predicted")
    px[0].set_ylabel("True")
    cax = px[0].matshow(cf_matrix/tot)

    px[0].set_title(f"(ACC: {ACC:.4f}, TPR: {results['TPR']:.4f}, TNR: {results['TNR']:.4f})\nThreshold: {classification_threshold}",
                    fontsize=14)

    px[1].set_xlabel("Predicted")
    px[1].set_ylabel("True")
    cax = px[1].matshow(cf_matrix_w_norm)
    px[1].set_title(f"(BA: {results['BA']:.4f}, TPR: {results['TPR']:.4f}, TNR: {results['TNR']:.4f})\n Threshold: {classification_threshold}",
                    fontsize=14)
    # fig.colorbar(cax)

    results["TN"], results["FP"], results["FN"], results["TP"] = TN, FP, FN, TP
    results["ACC"] = ACC
    results["cf_matrix"] = cf_matrix
    # Scores output by the neural network without classification
    results["scores"] = scores
    results["prediction"] = predictions
    results["ground_truth"] = truth
    results["classification_threshold"] = classification_threshold

    if output_folder is not None:
        f_name = f"confusion_matrix_epoch_{epoch}.png" if epoch is not None else "confusion_matrix_epoch.png"
        plt.savefig(f"{output_folder}/{f_name}", dpi=300,
                    bbox_inches='tight', transparent=True)

        pkl_name = f"test_results_{epoch}.pkl" if epoch is not None else "test_results.pkl"
        with open(f'{output_folder}/{pkl_name}', 'wb') as f:
            pickle.dump(results, f)
    plt.show()
    # ----------------------------------

    fpr, tpr, _ = roc_curve(truth, scores)
    results["ROC_AUC"] = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label="ROC curve (area = %0.2f)" % results["ROC_AUC"])
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.title("Receiver operating characteristic", fontsize=24)
    plt.legend(loc="lower right")

    if output_folder is not None:
        roc_name = f"roc_{epoch}.png" if epoch is not None else "roc.png"
        plt.savefig(f"{output_folder}/{roc_name}", dpi=300,
                    bbox_inches='tight', transparent=True)
    plt.show()

    return results


def getAccuracy(y_true, y_prob, classification_threshold):
    # TODO: make this a vector function
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > classification_threshold
    return (y_true == y_prob).sum().item() / y_true.size(0)


def load_model_for_testing(modelFolder):

    modelPath = f"{modelFolder}/model.pt"
    model_architecture_file = f"{modelFolder}/architecture.py"

    # Loading the model file
    print(">>> Loading model from the provided path...")
    spec = importlib.util.spec_from_file_location(
        "model", model_architecture_file)
    model_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_lib)
    model = model_lib.TracksterLinkingNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(modelPath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model epoch: {checkpoint['epoch']}")
    print(">>> Model Loaded.")
    prepare_network_input_data = None
    try:
        prepare_network_input_data = model_lib.prepare_network_input_data
    except Exception as ex:
        print(ex)
    return model, prepare_network_input_data


def plotLoss(model_base_folder):
    plt.style.use(hep.style.CMS)
    # plt.style.use('seaborn-whitegrid')
    train_loss_path = model_base_folder + "/loss/train_loss_history.pkl"
    val_loss_path = model_base_folder + "/loss/val_loss_history.pkl"

    with open(train_loss_path, 'rb') as f:
        train_loss = pickle.load(f)
    with open(val_loss_path, 'rb') as f:
        val_loss = pickle.load(f)

    fig = plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-whitegrid')
    plt.plot(range(1, len(train_loss)+1),
             train_loss, label='train', linewidth=2)
    plt.plot(range(1, len(val_loss)+1), val_loss, label='val', linewidth=2)
    plt.ylabel("Loss", fontsize=22)
    plt.xlabel("Epochs", fontsize=22)
    plt.title("Training and Validation Loss", fontsize=24)
    plt.legend()
    plt.show()
    # plt.savefig(f"{model_base_folder}/loss/losses.png")


def truth_pairs(data_list, ev, thr):
    data_ev = prepare_test_data(data_list, ev)
    truth_edge_index = data_ev.edge_index
    truth_edge_label = data_ev.edge_label > thr
    truth_nodes_features = data_ev.x

    src_edge_index_true = truth_edge_index[0][truth_edge_label]
    dest_edge_index_true = truth_edge_index[1][truth_edge_label]
    edge_scores_Ereco = data_ev.edge_score_Ereco[truth_edge_label]

    index_tuple = []
    for i in range(len(src_edge_index_true)):
        index_tuple.append([src_edge_index_true[i].item(),
                           dest_edge_index_true[i].item()])
    return truth_nodes_features, index_tuple, edge_scores_Ereco


def prediction_pairs(model, data_list, ev, thr, prepare_network_input_data=None, return_net_out=False, device=False, edge_features=False):
    data_ev = prepare_test_data(data_list, ev)
    if prepare_network_input_data is not None:
        if edge_features:
            if data_ev.edge_index.shape[1] != data_ev.edge_features.shape[0]:
                print("ERROR: edge index shape is different from edge features shape")
                return 0
            inputs = prepare_network_input_data(data_ev.x, data_ev.edge_index, data_ev.edge_features,
                                                device='cuda:0' if next(model.parameters()).is_cuda else 'cpu')

        else:
            inputs = prepare_network_input_data(data_ev.x, data_ev.edge_index,
                                                device='cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    else:
        inputs = (data_ev.x, data_ev.edge_index)

    if device:
        out, emb = model(
            *inputs, device='cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    else:
        out, emb = model(*inputs)
    edge_index = data_ev.edge_index
    truth_edge_label = out > thr
    node_features = data_ev.x

    src_edge_index_true = edge_index[0][truth_edge_label]
    dest_edge_index_true = edge_index[1][truth_edge_label]
    edge_scores_Ereco = data_ev.edge_score_Ereco[truth_edge_label]

    index_tuple = []
    for i in range(len(src_edge_index_true)):
        index_tuple.append([src_edge_index_true[i].item(),
                           dest_edge_index_true[i].item()])

    if return_net_out:
        return node_features, index_tuple, edge_scores_Ereco, out

    return node_features, index_tuple, edge_scores_Ereco


def connectivity_matrix(model, data_list, ev, thr, similarity=True, prepare_network_input_data=None):
    data_ev = prepare_test_data(data_list, ev)

    if prepare_network_input_data is not None:
        inputs = prepare_network_input_data(data_ev.x, data_ev.edge_index,
                                            device='cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    else:
        inputs = data_ev.x, data_ev.edge_index
    out, emb = model(*inputs)
    N = data_ev.num_nodes
    mat = np.zeros([N, N])
    truth_mat = np.zeros([N, N])
    for indx, src in enumerate(data_ev.edge_index[0]):
        dest = data_ev.edge_index[1][indx]
        mat[src][dest] = out[indx]
        mat[dest][src] = out[indx]
        truth_mat[src][dest] = data_ev.edge_label[indx]
        truth_mat[dest][src] = data_ev.edge_label[indx]

    if similarity == False:
        mat = mat > thr
    return mat, truth_mat


def plot_prediction_distribution(model, test_dl, threshold=0.7):
    for sample in test_dl:
        sample = sample.to(device)
        pred, emb = model(sample.x, sample.edge_index)
        targets = sample.edge_label
        break

    pred = pred.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    fig, axs = plt.subplots(3, figsize=(5, 8))
    fig.suptitle('Distributions of the labels')
    axs[0].set_title('Prediction distribution')
    axs[0].hist(pred, bins=20)
    axs[1].hist(targets, bins=20)
    axs[1].set_title('True Edge Labels')
    thresholded = (np.array(pred) > thr).astype(int)

    axs[2].set_title(f'Predicted Edge Labels for thr: {thr}')
    axs[2].hist(thresholded, bins=20)

    print(f"Edge labels: number of positive: {targets.sum()}")
    print(f"Predictions: number of positive: {thresholded.sum()}")

    plt.tight_layout()
    # hep.cms.text('Preliminary')
    plt.show()


def get_truth_labels(data_list, ev):
    """list of indices of best matched simts to all ts in an event"""
    x_best_simts = data_list[ev][5]
    return x_best_simts


def get_cand_labels(data_list, ev):
    """candidates containing the trackster"""
    cand_match = data_list[ev][6]
    return cand_match
