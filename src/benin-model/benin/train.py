import os

from torch import optim, cuda, channels_last
from torch import device as get_device
from torch.utils.data import DataLoader

from model import UNet
from npz_dataset import DatasetFromPath, train_test_split, TRAIN_TEST_RATIO


LOSSES = {
    'MSE': optim.MSELoss(),
    'CrossEntropy' : optim.CrossEntropyLoss(),
}

OPTIMIZERS = {
    'Adam' : optim.Adam,
    'SGD' : optim.SGD
}

# Default values
EPOCHS = 10
BATCH_SIZE = 512 # 2^n best


def run(
    data_path: str,
    model_path: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    train_test_ratio: float = TRAIN_TEST_RATIO,
    checkpointing: bool = False,
    optimizer = 'Adam'
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
    
    # Loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
    
    # Optimizer
    given_opt = OPTIMIZERS[optimizer]
    optimizer = given_opt(model.parameters(),
                    lr=1e-5,
                    weight_decay=1e-8,
                    momentum=0.999,
                    foreach=True)
    
    # Make Model
    device = get_device('cuda' if cuda.is_available() else 'cpu')
        # 4 channels for RGB NVDI TODO: Change to whatever dataset shape is:
    model = UNet(n_channels=4, n_classes=1)
        # Put the model in GPU memory for faster referencing
    model = model.to(memory_format=channels_last)
    model.to(device)
    if checkpointing: model.use_checkpointing()
    
    # Run training
    for batch in train_loader:
        pass


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
        default='Adam',
        help="Optimizer used for training (Adam or SGD)",
    )
    args = parser.parse_args()

    try: 
        run(data_path=args.data_path,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_test_ratio=args.train_test_ratio,
            optimizer=args.optimizer)
        
        
    except cuda.OutOfMemoryError:
        # If we run out of memory, use checkpointing
        run(data_path=args.data_path,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_test_ratio=args.train_test_ratio,
            optimizer=args.optimizer,
            checkpointing=True)

if __name__ == "__main__":
    main()