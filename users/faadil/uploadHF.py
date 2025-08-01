import os
import argparse
import huggingface_hub
from token_hub import TOKEN, nameaccount
from tools import _upload_batch
import sys

def upload_specific_folder(folder_path):
    """
    Uploads a specific folder to Hugging Face's dataset repository.
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder {folder_path} does not exist!")
    
    print(f"Uploading folder: {folder_path}")
    
    huggingface_hub.login(token=TOKEN)
    repo_id = "dolphinteam/DolphinChat-Validation"
    _upload_batch(repo_id, folder_path, chksize=100, repo_type="dataset")
    
    print("Finished uploading the dataset")
    return True

def main(args):
    parser = argparse.ArgumentParser(description='Upload a specific folder to Hugging Face')
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder to upload')
    args_class = parser.parse_args(args)

    folder_path = args_class.folder
    upload_specific_folder(folder_path)

if __name__ == "__main__":
    main(sys.argv[1:])
