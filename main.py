import os

from datasets import Dataset
from tqdm import tqdm
from PIL import Image

REPO_OWNER = "" # Username or organization name
REPO_NAME = "" # Repository name
DATA_DIR_PATH = r"" # Path to the directory containing the images
FILE_EXTENSION = "" # File extension of the images (e.g. .JPEG, .jpg, .png)

records = []
for dir_name in os.listdir(DATA_DIR_PATH):
    dir_path = os.path.join(DATA_DIR_PATH, dir_name)
    if os.path.isdir(dir_path):
        for file_name in tqdm(os.listdir(dir_path)):
            if file_name.endswith(FILE_EXTENSION):
                file_path = os.path.join(dir_path, file_name)
                with Image.open(file_path) as im:
                    records.append({
                        "image": im,
                        "label": dir_name
                    })

dataset = Dataset.from_dict({"image": [r["image"] for r in records], "label": [r["label"] for r in records]})
dataset.push_to_hub(f"{REPO_OWNER}/{REPO_NAME}")
