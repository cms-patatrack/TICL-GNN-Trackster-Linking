import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch
import torch.nn.functional as F

from test import *
from GNN_TrackLinkingNet import prepare_network_input_data


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.4):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions, targets):        
        """Binary focal loss, mean.

        Per https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5 with
        improvements for alpha.
        :param bce_loss: Binary Cross Entropy loss, a torch tensor.
        :param targets: a torch tensor containing the ground truth, 0s and 1s.
        :param gamma: focal loss power parameter, a float scalar.
        :param alpha: weight of the class indicated by 1, a float scalar.
        """
        bce_loss = F.binary_cross_entropy(predictions, targets)
        p_t = torch.exp(-bce_loss)
        alpha_tensor = (1 - self.alpha) + targets * (2 * self.alpha - 1)  
        # alpha if target = 1 and 1 - alpha if target = 0
        f_loss = alpha_tensor * (1 - p_t) ** self.gamma * bce_loss
        return f_loss.mean()
    
def get_unique_run(models_path):
    """
    Prepare the output folder run id.
    """
    previous_runs = os.listdir(models_path)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    return run_number

def train_val(model, trainLoader, valLoader, optimizer, loss_function, epochs, outputModelPath,
              scheduler=None, with_scores=False, update_every=1, save_checkpoint_every=1,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Performs training and validation of the network for a required number of epochs.
    """    
    outputModelCheckpointPath = outputModelPath + "/checkpoints/"
    outputLossFunctionPath = outputModelPath + "/loss/"
    outputTrainingPlotsPath = outputModelPath + "/trainingPlots/"
    
    # Create directories for saving the models/checkpoints if not exist
    os.mkdir(outputModelPath)
    os.mkdir(outputModelCheckpointPath)
    os.mkdir(outputLossFunctionPath)
    os.mkdir(outputTrainingPlotsPath)
    
    train_loss_history = []
    val_loss_history = []

    print(f"Model output directory: {outputModelPath}")
    print(f"Saving checkpoints every {save_checkpoint_every} epochs.")
    
    print(">>> Model training started.")
    model.to(device)
    
    for epoch in range(epochs):
        batchloss = []
        train_true_seg, train_pred_seg, scores_seg = [], [], []
        b = 0
        num_samples = len(trainLoader)
        optimizer.zero_grad()
                
        # Training ----------------------------------------------------
        model.train()
        for sample in tqdm(trainLoader, desc=f'Training epoch {epoch+1}'):
            sample.to(device)
            
            num_tracksters, feat_dim = sample.x.shape
            if num_tracksters <= 1:
                continue
        
            inputs = prepare_network_input_data(sample.x, sample.edge_index)
            out, emb = model(*inputs)
            if not with_scores:
                loss = loss_function(out, sample.edge_label.to(torch.float32))
            else:
                loss = loss_function(out, sample.edge_score_Ereco)
            batchloss.append(loss.item())
            loss.backward()
            
            if (b+1) % update_every == 0 or (b+1) == num_samples:
                optimizer.step()
                optimizer.zero_grad()
            
            b += 1
            
            seg_np = sample.edge_label.cpu().numpy()
            scores = sample.edge_score_Ereco.cpu().numpy()
            pred_np = out.detach().cpu().numpy()

            train_true_seg.append(seg_np.reshape(-1))
            train_pred_seg.append(pred_np.reshape(-1))
            scores_seg.append(scores.reshape(-1))

        train_true_cls = np.concatenate(train_true_seg)
        train_pred_cls = np.concatenate(train_pred_seg)
        scores_cls = np.concatenate(scores_seg)
        
        plot_prediction_distribution_standard_and_log(train_pred_cls, train_true_cls,
                                                      epoch=epoch+1, thr = 0.65, scores=scores_cls,
                                                      folder=outputTrainingPlotsPath)

        train_loss = np.mean(batchloss)
        
        if hasattr(model, 'writer'):
            model.writer.add_scalar("Loss/train", train_loss, epoch)
            model.writer.flush()
            
        train_loss_history.append(train_loss)
        # End Training ----------------------------------------------------
            
        # Validation ------------------------------------------------------
        val_true_seg, val_pred_seg = [], []
        
        with torch.set_grad_enabled(False):
            batchloss = []
            model.eval()
            for sample in tqdm(valLoader, desc=f'Validation epoch {epoch+1}'):
                sample.to(device)
                
                num_tracksters, feat_dim = sample.x.shape
                if num_tracksters <= 1:
                    continue
                
                inputs = prepare_network_input_data(sample.x, sample.edge_index)
                out, emb = model(*inputs)
                if not with_scores:
                    val_loss = loss_function(out, sample.edge_label.to(torch.float32))
                else:
                    val_loss = loss_function(out, sample.edge_score_Ereco)
                
                batchloss.append(val_loss.item())
                
                seg_np = sample.edge_label.cpu().numpy()
                pred_np = out.detach().cpu().numpy()

                val_true_seg.append(seg_np.reshape(-1))
                val_pred_seg.append(pred_np.reshape(-1))

        val_true_cls = np.concatenate(val_true_seg)
        val_pred_cls = np.concatenate(val_pred_seg)        
                
        print("Testing model on validation data...")
        TNR, TPR, thresholds = classification_thresholds_plot(val_pred_cls, val_true_cls, threshold_step=0.05,
                                       output_folder=outputTrainingPlotsPath, epoch=epoch+1)
        classification_threshold = get_best_threshold(TNR, TPR, thresholds)
        print(f"Chosen classification threshold is: {classification_threshold}")
        
        plot_prediction_distribution_standard_and_log(val_pred_cls, val_true_cls,
                                                      epoch=epoch+1, thr = classification_threshold,
                                                     folder=outputTrainingPlotsPath, val=True)
        
        test_results = test(val_true_cls, val_pred_cls, classification_threshold=classification_threshold,
                            output_folder=outputTrainingPlotsPath, epoch=epoch+1)

        val_loss = np.mean(batchloss)
        
        if hasattr(model, 'writer'):
            model.writer.add_scalar("Loss/val", val_loss, epoch)
            model.writer.flush()
        val_loss_history.append(val_loss)
        # End Validation ----------------------------------------------------
        
        # Save checkpoint every 'save_checkpoint_every' epochs
        if(epoch != 0 and epoch != epochs-1 and (epoch+1) % save_checkpoint_every == 0):
            print("Saving a model checkpoint.")
            torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss_history,
                        }, outputModelCheckpointPath + f"/epoch_{epoch+1}_val_loss{val_loss:.6f}.pt")
        
        if scheduler is not None:
            scheduler.step(val_loss_history[-1])
            for param_group in optimizer.param_groups:
                print(f"lr: {param_group['lr']}")
            
        print(f"epoch {epoch+1}: Train loss: {train_loss_history[-1]} \t Val loss: {val_loss_history[-1]}")
        
        # Save the updated picture and pkl files of the losses 
        save_loss(train_loss_history, val_loss_history, outputLossFunctionPath)

    # Save the final model
    print(f">>> Training finished. Saving model to {outputModelPath + '/model.pt'}")
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_history,
                }, outputModelPath + "/model.pt")
    
    if hasattr(model, 'writer'):
        model.writer.close()
        
    return train_loss_history, val_loss_history

def save_loss(train_loss_history, val_loss_history, outputLossFunctionPath):
    # Saving the figure of the training and validation loss
    fig = plt.figure(figsize=(9, 5))
    #plt.style.use('seaborn-whitegrid')
    epochs = len(train_loss_history)
    plt.plot(range(1, epochs+1), train_loss_history, label='train', linewidth=2)
    plt.plot(range(1, epochs+1), val_loss_history, label='val', linewidth=2)
    plt.ylabel("Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    plt.title("Training and Validation Loss", fontsize=18)
    plt.legend()
    plt.savefig(f"{outputLossFunctionPath}/losses.png")
    plt.show()
    
    # Save the train and validation loss histories to pkl files
    with open(outputLossFunctionPath + 'train_loss_history.pkl','wb') as f:
        pickle.dump(train_loss_history, f)

    with open(outputLossFunctionPath + 'val_loss_history.pkl','wb') as f:
        pickle.dump(val_loss_history, f)

def print_training_dataset_statistics(trainDataset):
    print(f"Number of events in training dataset: {len(trainDataset)}")
    num_nodes, num_edges, num_neg, num_pos = 0, 0, 0, 0
    for ev in trainDataset:
        num_nodes += ev.num_nodes
        num_edges += len(ev.edge_label)
        num_pos += (ev.edge_label == 1).sum()
        num_neg += (ev.edge_label == 0).sum()
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Number of positive edges: {num_pos}")
    print(f"Number of negative edges: {num_neg}")


def plot_prediction_distribution_standard_and_log(pred, targets, epoch=None, thr = 0.65, scores=None, folder=None, val=False):
    
    fig = plt.figure(figsize=(10,9))
    #plt.style.use('seaborn-whitegrid')

    fig.suptitle(f'Distributions of the labels (Epoch: {epoch})')
    
    num_plots = 4 if scores is not None else 3
    
    # Predictions in linear and log scales
    ax1 = fig.add_subplot(num_plots,2,1)
    ax1.set_title('Prediction distribution')
    ax1.hist(pred, bins=30)
    
    ax2 = fig.add_subplot(num_plots,2,2)
    ax2.set_title('Prediction distribution Log')
    ax2.hist(pred, bins=30)
    ax2.set_yscale('log')
    #------------------------
    
    # Truth labels in linear and log scales
    ax3 = fig.add_subplot(num_plots,2,3)
    ax3.hist(targets, bins=30)
    ax3.set_title('True Edge Labels')
    thresholded = (np.array(pred) > thr).astype(int)
    
    ax4 = fig.add_subplot(num_plots,2,4)
    ax4.hist(targets, bins=30)
    ax4.set_title('True Edge Labels Log')
    ax4.set_yscale('log')
    #------------------------

    # Thresholded labels in linear and log scales
    ax5 = fig.add_subplot(num_plots,2,5)
    ax5.set_title(f'Predicted Edge Labels for thr: {thr}')
    ax5.hist(thresholded, bins=30)
    
    ax6 = fig.add_subplot(num_plots,2,6)
    ax6.set_title(f'Predicted Edge Labels Log for thr: {thr}')
    ax6.hist(thresholded, bins=30)
    ax6.set_yscale('log')
    #------------------------
    
    if scores is not None:
        ax7 = fig.add_subplot(num_plots,2,7)
        ax7.set_title('Edge Scores')
        ax7.hist(scores, bins=30)
        
        ax8 = fig.add_subplot(num_plots,2,8)
        ax8.set_title('Edge Scores Log')
        ax8.hist(scores, bins=30)
        ax8.set_yscale('log')

    print(f"Edge labels: number of positive: {targets.sum()}")
    print(f"Predictions: number of positive: {thresholded.sum()}")

    plt.xlim([0, 1])
    plt.tight_layout()
    if folder is not None:
        name = f"plot_train_prediction_distribution_epoch_{epoch}.png" if val else f"plot_val_prediction_distribution_epoch_{epoch}.png"
        plt.savefig(f'{folder}/{name}', bbox_inches='tight')
    plt.show()