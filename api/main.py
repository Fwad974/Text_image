import json
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from api.model import Input
from app import app, logger

model_api = APIRouter()

import os
import sys
import json
import zipfile
import natsort
import glob
import numpy as np
from PIL import Image as PilImage

os.environ['TOKENIZERS_PARALLELISM'] = "false"

import torchvision
from torchvision import transforms
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode
from tqdm.notebook import tqdm
import torch
import clip
from PIL import Image

from multilingual_clip import pt_multilingual_clip
import transformers

sys.path.append('.')


def download_image_data():
    os.system("git clone https://github.com/crux82/mscoco-it/")
    os.system("wget -N -q --show-progress http://images.cocodataset.org/zips/val2014.zip")
    os.system("gunzip /content/mscoco-it/mscoco-it/captions_ita_devset_validated.json.gz")
    os.system("gunzip /content/mscoco-it/mscoco-it/captions_ita_devset_unvalidated.json.gz")


def get_path_coco(image_id):
    image_id = int(image_id)
    return f"photos/val2014/COCO_val2014_{image_id:012d}.jpg"


def image_database_preparing():
    img_folder = 'photos/'
    if not os.path.exists(img_folder) or len(os.listdir(img_folder)) == 0:
        os.makedirs(img_folder, exist_ok=True)

    with zipfile.ZipFile("val2014.zip", 'r') as zf:
        for member in tqdm(zf.infolist(), desc='Extracting'):
            zf.extract(member, img_folder)

    with open("/content/mscoco-it/mscoco-it/captions_ita_devset_unvalidated.json") as filino:
        data = json.load(filino)["annotations"]
    return data


class CLIP:
    def __init__(self, text_model_name='M-CLIP/XLM-Roberta-Large-Vit-B-32', image_model_name="ViT-B/32",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.image_model, self.image_preprocess = clip.load(image_model_name, device=device)
        self.lang_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(text_model_name)
        self.lang_tokenizer = transformers.AutoTokenizer.from_pretrained(text_model_name)

    def get_imge_embedding(self, image):
        return self.image_model.encode_image(image).cpu().detach().numpy()

    def get_text_embedding(self, text):
        embedding = self.lang_model.forward(text, self.lang_tokenizer).cpu().detach().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


download_image_data()
data = image_database_preparing()
good_images = [get_path_coco(d["image_id"]) for d in data]
img_names = list(glob.glob('photos/val2014/*.jpg'))
destroy_images = set(img_names).difference(set(good_images))
for img in destroy_images:
    os.remove(img)
model = CLIP()


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def get_image_name(self, idx):
        return self.total_imgs[idx]

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = PilImage.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


class SimpleTextDataset(torch.utils.data.Dataset):

    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class Data:
    def __init__(self, path="photos/val2014", batch_size=32):
        self.image_size = 224
        self.val_preprocess = transforms.Compose([
            Resize([self.image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(self.image_size),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.dataset = CustomDataSet(path, transform=self.val_preprocess)
        # text_dataset = SimpleTextDataset([elem["caption"] for elem in data])

        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            drop_last=False)
        self.batch_size = batch_size
        self.image_features = self.image_loader()

    #  text_loader = torch.utils.data.DataLoader(
    #     text_dataset,
    #    batch_size=32,
    #   shuffle=False)
    def image_loader(self):
        image_features = []
        for i, images in enumerate(tqdm(self.loader)):
            # images = images.permute(0, 2, 3, 1).numpy()
            # features = model.get_image_features(images,)
            features = model.get_imge_embedding(images)
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            image_features.extend(features)
        return image_features

    def find_image(self, text_query, n=1):
        zeroshot_weights = model.get_text_embedding(text_query)
        distances = np.dot(self.image_features, zeroshot_weights.reshape(-1, 1))
        file_paths = []
        for i in range(1, n + 1):
            idx = np.argsort(distances, axis=0)[-i, 0]
            file_paths.append('photos/val2014/' + self.dataset.get_image_name(idx))
        return file_paths


# model=CLIP()

process = Data()


@model_api.post("/predict")
def predict(data: Input, process=process) -> JSONResponse:
    res = process(data.description)
    return JSONResponse(
        content={"output": res}, status_code=200
    )
