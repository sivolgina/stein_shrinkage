import torch
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Training loop
def cross_entropy2d(input, target, weight=None, reduction='mean'):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, reduction='mean', ignore_index=250
    )
    return loss

def train(train_loader, model, optimizer, epoch_i, epoch_total):
        count = 0
        loss_list = []
        for (images, labels) in train_loader:
            count += 1
            optimizer.zero_grad()
            model.train()

            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = cross_entropy2d(pred, labels)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        return(np.array(loss_list).mean())

