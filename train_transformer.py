from tqdm import tqdm
import numpy as np

import torch

from LossFunction import Loss


def train(model, optimizer, loader, epoch, loss_obj, vocab_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    epoch_loss = 0
    model.train()
    for sample in tqdm(loader, desc=f"Training Epoch {epoch}"):
        # reset optimizer and enable training mode
        optimizer.zero_grad()

        # move data to the device
        X = sample[0]
        Y = sample[1]
        ys = sample[2]

        z = model(X, Y)

        # compute the loss
        loss = loss_obj(z.contiguous().view(-1, vocab_size), ys)

        # back-propagate and update the weight
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return float(epoch_loss)/len(loader)


def test(model, loader, epoch, loss_obj, vocab_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # vocab_size = converter.n_words

    with torch.set_grad_enabled(False):
        model.eval()
        # pred, y = [], []
        val_loss = 0

        for sample in tqdm(loader, desc=f"Validation Epoch {epoch}"):
            X = sample[0]
            Y = sample[1]
            ys = sample[2]

            z = model(X, Y)
            # predicted_index = z.argmax(-1)
            # predicted_number = converter.index2word[int(predicted_index[0, -1].item())]

            # pred.append(predicted_number)
            # y.append(ys[0, -1])
            loss = loss_obj.forward(z.contiguous().view(-1, vocab_size), ys).item()
            val_loss += loss

        val_loss /= len(loader)
    return val_loss
