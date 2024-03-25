'''
This script is used for obtaining image features using openclip model
The images source could be downloaded from https://github.com/xheon/panoptic-reconstruction (front3d-2d.zip)
'''
import torch
from PIL import Image
import open_clip
import os
import glob
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()
model.to(device=device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

img_path = '../front3d_2d_images'# path should be set
list_json_files = os.listdir(img_path)

path_image_features = '../open_clip_features/images/'
os.makedirs(path_image_features, exist_ok=True)

counter = 0
for jf in tqdm(list_json_files):
    existing_file = os.listdir(img_path + os.sep + jf)
    png_files = glob.glob(img_path + os.sep + jf + '/*.png')
    png_files.sort()
    images = torch.empty(len(png_files), 3, 224, 224)
    for idx, img in enumerate(png_files):
        images[idx, :, :, :] = preprocess(Image.open(img))
        counter += 1
    images = images.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(images)
        # print(image_features.shape)
        image_features = image_features.cpu()
    torch.save(image_features, f'{path_image_features}{jf}.pt')

print(counter)

