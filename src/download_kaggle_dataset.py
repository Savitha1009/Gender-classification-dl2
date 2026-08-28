import os
import sys
import argparse
import zipfile

KAGGLE_DATASETS = {
    "1": ("competitions", "facial-gender-recognition", "Kaggle Facial Gender Recognition Competition"),
    "2": ("datasets", "aadyasingh55/face-age-gender-dataset", "Face Age Gender Dataset (UTKFace format)"),
    "3": ("datasets", "graphical27/gender-detection", "Gender Detection Dataset"),
    "4": ("datasets", "mustafahabeeb90/gender-classification", "Gender Classification Dataset")
}

def download_dataset(choice, target_dir):
    if choice not in KAGGLE_DATASETS:
        print(f"Invalid choice '{choice}'. Select 1, 2, 3, or 4.")
        return
        
    ds_type, ds_name, description = KAGGLE_DATASETS[choice]
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"\n[Kaggle Downloader] Target: {description} ({ds_name})")
    
    if ds_type == "competitions":
        cmd = f"kaggle competitions download -c {ds_name} -p \"{target_dir}\""
    else:
        cmd = f"kaggle datasets download -d {ds_name} -p \"{target_dir}\""
        
    print(f"Executing: {cmd}")
    ret = os.system(cmd)
    
    if ret != 0:
        print("\n[Notice] Automatic Kaggle CLI download requires kaggle API credentials (~/.kaggle/kaggle.json).")
        print("Manual Download Instructions:")
        print(f"1. Visit: https://www.kaggle.com/{'competitions' if ds_type=='competitions' else 'datasets'}/{ds_name}")
        print(f"2. Download zip and extract contents directly into: {target_dir}")
        return

    # Extract any downloaded zip files in target_dir
    zip_files = [f for f in os.listdir(target_dir) if f.endswith('.zip')]
    for z in zip_files:
        zip_path = os.path.join(target_dir, z)
        print(f"Extracting {z}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(zip_path)
        
    print(f"[Kaggle Downloader] Dataset successfully prepared in '{target_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kaggle Dataset Downloader Helper")
    parser.add_argument('--dataset_id', type=str, default="1", choices=["1", "2", "3", "4"], help="Dataset index (1-4)")
    parser.add_argument('--target_dir', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'dataset'), help="Output directory")
    args = parser.parse_args()
    
    download_dataset(args.dataset_id, args.target_dir)
