"""
The seed bench looks like 

"questions": [
{
    "answer": "A",
    "choice_a": "One",
    "choice_b": "Two",
    "choice_c": "Three",
    "choice_d": "Four",
    "data_id": "1454426_2591111986",
    "data_type": "image",
    "question": "How many towels are in the image?",
    "question_id": "101669",
    "question_type_id": 5
},
"""
from torch.utils.data import Dataset
import json
import os
from PIL import Image


def is_integer_string(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def filter_questions(data, task='all'):
    if task == "image":
        return [q for q in data if 1 <= q["question_type_id"] <= 9]
    elif task == "video":
        return [q for q in data if 10 <= q["question_type_id"] <= 12]
    elif task == "all":
        return data
    elif is_integer_string(task):
        return [q for q in data if q["question_type_id"] == int(task)]
    else:
        raise ValueError(f"Invalid task: {task}")
        

class SEEDBenchDataset(Dataset):
    def __init__(
        self,
        image_dir_path,
        question_path,
        is_train,
        dataset_name="seed_bench",
    ):
        print("Loading SEED Bench dataset...")
        print("Loading questions from", question_path)
        self.questions = json.load(open(question_path, "r"))["questions"]
        self.questions = filter_questions(self.questions, task="image")
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name
  

    def __len__(self):
        return len(self.questions)

        
    def get_img_path(self, question):
        if self.dataset_name == "seed_bench":
            return os.path.join(self.image_dir_path, f"{question['data_id']}.jpg")
        else:
            raise Exception(f"Unknown instruction tuning dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        return {
            "image": image,
            "question": question["question"],
            "candidates": [question["choice_a"], question["choice_b"], question["choice_c"], question["choice_d"]],
            "answers": [question["answer"]],
            "question_id": question["question_id"],
        }