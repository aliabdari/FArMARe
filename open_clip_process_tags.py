'''
This script is developed to extract the open clip features of the obtained tags using yolo for each of the available images in each house
'''
import torch
import open_clip
from tqdm import tqdm
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()
model.to(device=device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

version_yolo = 5
pickle_file = open(f'yolo/images_tags_info_customdataset_v{version_yolo}.pkl', 'rb')
pickle_file = pickle.load(pickle_file)

for jf in tqdm(pickle_file):
    tmp_tensor = torch.empty(len(jf['image_tags']), 512)
    list_tags = []
    list_img_names = []
    for tag in jf['image_tags']:
        list_img_names.append(tag['name'])
        tmp_tag = tag['tag']
        if tmp_tag is None:
            tmp_tag = 'house'
        list_tags.append(tmp_tag)
    sorted_indexes = [i for i, _ in sorted(enumerate(list_img_names), key=lambda x: x[1])]
    sorted_tags_list = [list_tags[i] for i in sorted_indexes]
    token_level_tokenized = tokenizer(sorted_tags_list)
    token_level_tokenized = token_level_tokenized.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        features_tokens = model.encode_text(token_level_tokenized)
        print(features_tokens.shape)

        file_name = jf['json_file']
        features_tokens = features_tokens.cpu()
        torch.save(features_tokens, f'./yolo/images_tags_info_customdataset_v{version_yolo}/{file_name}.pt')


print('Finished')
