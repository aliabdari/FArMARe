'''
This script is developed to extract the resnet 152 model features of the images
The images source could be downloaded from https://github.com/xheon/panoptic-reconstruction (front3d-2d.zip)
'''
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm


def load_model():
    model = models.resnet152(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    return model


def process_imagery_data(pretrained_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = pretrained_model.to(device)
    # Image preprocessing transformation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # general path of images - The path should be set after the aforementioned dataset download
    path_images = '../front3d_2d_images/'

    path_image_features = '../resnet_bert_features/images/'
    os.makedirs(path_image_features, exist_ok=True)

    overall_folders = os.listdir(path_images)
    for folder in tqdm(overall_folders):
        images_names = []
        seps_path = path_images + folder
        existing_files = os.listdir(seps_path)
        # print(len(existing_files))
        for fi in existing_files:
            if fi.endswith(".png"):
                # print(seps_path+os.sep+fi)
                images_names.append(seps_path+os.sep+fi)
        images_names.sort()
        input_tensor_overall = torch.empty(len(images_names), 3, 224, 224)
        for idx, img in enumerate(images_names):
                image = Image.open(img).convert("RGB")
                input_tensor_overall[idx, :, :, :] = preprocess(image)

        input_batch = input_tensor_overall.to(device)

        # Extract features
        with torch.no_grad():
            features = pretrained_model(input_batch)

        # Flatten the features tensor
        features = torch.flatten(features, start_dim=1)

        features = features.cpu()
        torch.save(features, f'{path_image_features}{folder}.pt')

        # Print the shape of the features
        if features.shape[0] == 0:
            print('size feature', features.shape)
            exit(0)
        # print(type(features))


if __name__ == '__main__':
    pretrained_model = load_model()
    process_imagery_data(pretrained_model=pretrained_model)



