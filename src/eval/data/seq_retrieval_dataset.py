import os
from torch.utils.data import Dataset
import json
from PIL import Image

class RetrievalDataset(Dataset):
    def __init__(
        self,
        json_dir_path,
        annotation_json_file,
        is_train,
        dataset_name,
    ):
        self.json_dir_path = json_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name
        print("Loading retrieval from", annotation_json_file)
        full_annotations = json.load(open(annotation_json_file))

        for i in range(len(full_annotations)):
            self.annotations.append(full_annotations[i])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.dataset_name == "obelics":
            # load json file
            with open(os.path.join(self.json_dir_path, self.annotations[idx])) as f:
                data = json.load(f)
            image_1 = Image.open(data["image_info"][0])
            image_1.load()
            text_1 = data["texts"][1]
            image_2 = Image.open(data["image_info"][2])
            image_2.load()
            text_2 = data["texts"][3]
        return {
            "image_1": image_1,
            "text_1": text_1,
            "image_2": image_2,
            "text_2": text_2,
            "image_id": idx
        }
