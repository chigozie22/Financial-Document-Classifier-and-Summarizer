import os
from datasets import load_dataset

def load_finqa_dataset_and_save(split="test", save_dir="data/finqa"):
    """
    Load FinQA dataset from Hugging Face and save it to a local file.
    The dataset will be saved as a CSV file.
    
    :param split: Dataset split (train, test, validation)
    :param save_dir: Directory to save the dataset
    """
    # Load the FinQA dataset
    dataset = load_dataset("ibm-research/finqa")
    dataset_split = dataset[split]

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define the save path (CSV file for the split)
    save_path = os.path.join(save_dir, f"finqa_{split}.csv")

    # Convert the dataset to a Pandas DataFrame and save as CSV
    dataset_split.to_pandas().to_csv(save_path, index=False)
    
    print(f"✅ {split} dataset saved to {save_path}")
    return dataset_split  # Returning the dataset object if needed

# Example usage
load_finqa_dataset_and_save(split="test", save_dir="data/finqa")
