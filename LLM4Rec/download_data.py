"""
Script to download datasets from Hugging Face and save them as JSONL files.
"""
import json
from datasets import load_dataset
from pathlib import Path

def download_and_save_dataset(dataset_name, output_filename, data_dir="data"):
    """
    Download a dataset from Hugging Face and save it as JSONL format.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        output_filename: Name of the output JSONL file
        data_dir: Directory to save the data
    """
    print(f"Downloading dataset: {dataset_name}...")
    
    # Create data directory if it doesn't exist
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name)
    
    # Get the split (usually 'train', but could be different)
    # Try common split names
    if 'train' in dataset:
        split_data = dataset['train']
    elif 'test' in dataset:
        split_data = dataset['test']
    else:
        # Use the first available split
        split_name = list(dataset.keys())[0]
        split_data = dataset[split_name]
    
    # Save as JSONL
    output_path = data_path / output_filename
    print(f"Saving to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in split_data:
            # Each item is a dictionary, write it as a JSON line
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Successfully saved {len(split_data)} items to {output_path}")
    return len(split_data)

def main():
    """Main function to download both datasets."""
    print("Starting dataset download...")
    print("=" * 60)
    
    # Download user profiles dataset
    user_count = download_and_save_dataset(
        dataset_name="zetianli/CS329H_Project_user_profiles",
        output_filename="user_profiles.jsonl"
    )
    
    print("\n" + "=" * 60)
    
    # Download business dataset
    business_count = download_and_save_dataset(
        dataset_name="zetianli/CS329H_Project_business",
        output_filename="business.jsonl"
    )
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Total user profiles: {user_count}")
    print(f"Total businesses: {business_count}")

if __name__ == "__main__":
    main()

