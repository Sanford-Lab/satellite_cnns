# Copy of utility module from @milesial's implementation of PyTorch U-Net model
# https://github.com/milesial/Pytorch-UNet/blob/2f62e6b1c8e98022a6418d31a76f6abd800e5ae7/utils/dice_score.py

from torch import (
    Tensor, 
    float32, 
    long, 
    channels_last, 
    autocast, 
    inference_mode,
    where
)
import torch.functional as F

from benin.model import MoveDim


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

# from evaluate.py:

@inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in dataloader:
            
            # Put inputs and labels in correct shape order
            to_channels_first = MoveDim(-1, 1)
            inputs, labels = to_channels_first(batch['inputs']),to_channels_first(batch['labels'])

            # move inputs and labels to correct device and type
            inputs = inputs.to(device=device, dtype=float32, memory_format=channels_last)
            labels = labels.to(device=device, dtype=long)

            # predict the mask
            labels_pred = net(inputs)

            if net.n_classes == 1:
                assert labels.min() >= 0 and labels.max() <= 1, 'True mask indices should be in [0, 1]'
                labels_pred = (F.sigmoid(labels_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(labels_pred, labels, reduce_batch_first=False)
            else:
                assert labels.min() >= 0 and labels.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                labels = F.one_hot(labels, net.n_classes).permute(0, 3, 1, 2).float()
                labels_pred = F.one_hot(labels_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(labels_pred[:, 1:], labels[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)