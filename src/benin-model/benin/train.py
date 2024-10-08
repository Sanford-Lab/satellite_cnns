# This package includes work covered by the following copyright and permission notices:
#
#  GitHub repo milesial/Pytorch-UNet, commit f2625b
#     train.py
#
#  Licensed under GNU General Public License v3.0
# ______________________________________________________________________________________
#
# This script is set up to create and train a model based on U-Net architecture
# (from the @milesial implementation). Refactoring still required
#
#   - Uses npz_dataset.py to read npz files and create torch dataset
#   - trains based on parameters
#   * CURRENTLY ONLY SET UP FOR SINGLE CLASS CLASSIFCATION *
#   * NOT CHECKED FOR NON-BENIN PROJECT SUPPORT *
#
#   SPIRES Lab, 2023
# ======================================================================================



import os
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from benin.model import UNet, MoveDim
from benin.dice_score import dice_loss, evaluate
from benin.npz_dataset import DatasetFromPath, train_test_split, TRAIN_TEST_RATIO


# Default values
EPOCHS : int= 10
BATCH_SIZE : int = 512 # 2^n best
LEARNING_RATE = 1e-5
WEIGHT_DECAY= 1e-8
MOMENTUM : float= 0.999
GRADIENT_CLIPPING : float = 1.0
OPTIMIZER = 'RMS'

dir_checkpoint = Path('./checkpoints/')

def run_torch(
    data_path: str,
    model_path: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    train_test_ratio: float = TRAIN_TEST_RATIO,
    checkpointing: bool = False,
    optimizer : str= OPTIMIZER,
    mixed_precision = False,
    gradient_clipping = GRADIENT_CLIPPING
) -> None:
    """Trains a new TestModel based on arguments"""
    
    print(f"data_path: {data_path}")
    print(f"model_path: {model_path}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"train_test_ratio: {train_test_ratio}")
    print("-" * 40)
    
    # Create dataset
    dataset = DatasetFromPath(data_path)
    print(str(dataset))
    
    # Split into train/test
    train_set, test_set = train_test_split(dataset, train_test_ratio)
    n_train = len(train_set)
    
    # Loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
    
    # Make Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    num_input_bands = train_set['inputs'].shape[3]
    # TODO: Adjust n_classes correctly
    num_labels_bands = 1 #train_set['labels'].shape[3]
    model = UNet(n_channels=num_input_bands, n_classes=num_labels_bands)
    logging.info(f'U-Net instantiated with {num_input_bands} input bands and {num_labels_bands} classes')
        # Put the model in GPU memory for faster referencing
    model = model.to(memory_format=torch.channels_last)
    model.to(device)
    if checkpointing: model.use_checkpointing()
    
    # Optimizer
    if optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(model.parameters(),
                        lr=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        momentum=MOMENTUM,
                        foreach=True)
    else:
        raise NotImplementedError
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        # Loss criterion
    criterion = torch.nn.CrossEntropyLoss() if model.n_classes > 1 else torch.nn.BCEWithLogitsLoss()
    global_step = 0
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {LEARNING_RATE}
        Training size:   {n_train}
        Testing size: {len(test_set)}
        Checkpoints?     {checkpointing}
        Device:          {device.type}
        Mixed Precision: {mixed_precision}
    ''')
    
    # Run training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        logging.info(f'Starting training epoch {epoch}')
        for batch in train_loader:

            # Put inputs and labels in correct shape order (TODO: move this to model as first layer)
            to_channels_first = MoveDim(-1, 1)
            inputs, labels = to_channels_first(batch['inputs']),to_channels_first(batch['labels'])

            
            assert inputs.shape[1] == model.n_channels, \
                f'Network has been defined with {model.n_channels} input channels, ' \
                f'but loaded inputs have {inputs.shape[1]} channels. Please check that ' \
                'the inputs are loaded correctly.'

            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=mixed_precision):
                masks_pred = model(inputs)
                if model.n_classes == 1:
                    loss = criterion(masks_pred.squeeze(1), labels.float().squeeze(1))
                    loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), labels.float().squeeze(1), multiclass=False)
                else:
                    loss = criterion(masks_pred, labels)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(labels, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        
            global_step += 1
            epoch_loss += loss.item()
            
            # Evaluation round
            division_step = (n_train // (5 * batch_size))
            if division_step > 0:
                if global_step % division_step == 0:

                    val_score = evaluate(model, val_loader, device, mixed_precision)
                    scheduler.step(val_score)

                    logging.info('Validation Dice score: {}'.format(val_score))

        
        if checkpointing:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
    
    # Save model
    torch.save(model.state_dict(), f'{model_path}_UNET')

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        help="Directory path to read data files from.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Directory path to write the trained model to.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of times to go through the training dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of training examples to learn from at once.",
    )
    parser.add_argument(
        "--train-test-ratio",
        type=float,
        default=TRAIN_TEST_RATIO,
        help="Ratio of examples to use for training and for testing.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=OPTIMIZER,
        help="Optimizer used for training (Adam or SGD)",
    )
    parser.add_argument(
        "--mixed-precision",
        type=bool,
        default=False,
        help="Whether or not you should youse mixed-precision",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
     
    try: 
        run_torch(data_path=args.data_path,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_test_ratio=args.train_test_ratio,
            optimizer=args.optimizer,
            mixed_precision=args.mixed_precision,
        )  
        
    # If we run out of memory, use checkpointing
    except torch.cuda.OutOfMemoryError:
        run_torch(data_path=args.data_path,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_test_ratio=args.train_test_ratio,
            optimizer=args.optimizer,
            mixed_precision=args.mixed_precision,
            checkpointing=True
        )

if __name__ == "__main__":
    main()
