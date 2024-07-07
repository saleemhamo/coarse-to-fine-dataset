import sys

sys.path.append('/app')

from data.datasets import get_dataset
from utils.constants import *


def main():
    # Print the current directory
    current_directory = os.getcwd()
    print(f"Current directory: {current_directory}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Train Annotation path: {CHARADES_ANNOTATIONS_TRAIN}")

    # Load Charades dataset
    dataset = get_dataset(CHARADES_STA)
    print("Loaded Charades dataset")

    # Print some train and test data
    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()
    print("First 5 training annotations:")
    for annotation in train_data[:5]:
        print(annotation)
    print("First 5 test annotations:")
    for annotation in test_data[:5]:
        print(annotation)


if __name__ == "__main__":
    main()
