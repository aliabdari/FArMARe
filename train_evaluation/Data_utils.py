import os
import torch
from torch.utils.data import Dataset
import pickle
import Constants


class DescriptionSceneDataset(Dataset):
    def __init__(self, data_description_path, data_scene_path, type_model_scene):
        self.description_path = data_description_path
        self.data_scene = data_scene_path
        self.type_model_scene = type_model_scene
        with open('../houses_data/houses_data.pkl', 'rb') as f:
            self.pickle_file = pickle.load(f)

    def __len__(self):
        return len(self.data_scene)

    def __getitem__(self, index):
        try:
            data_description = torch.load(self.description_path + os.sep + self.pickle_file[index]['json_file'] + '.pt')
            data_scene = torch.load(self.data_scene + os.sep + self.pickle_file[index]['json_file'] + '.pt')
            if self.type_model_scene == Constants.model_scene_mean:
                data_scene = torch.mean(data_scene, 0)
        except:
            print('index:', index)
            exit(0)
        return data_description, data_scene


class DescriptionSceneDatasetMem(Dataset):
    def __init__(self, data_description_path, data_scene_path, type_model_scene, is_synopsis=False,
                 data_synopsis_path=None):
        self.description_path = data_description_path
        self.data_scene = []
        self.data_description = []
        self.type_model_scene = type_model_scene
        self.synopsis = is_synopsis
        self.data_synopsis_path = data_synopsis_path
        with open(
                '../houses_data/houses_data.pkl',
                'rb') as f:
            self.pickle_file = pickle.load(f)
        for jf in self.pickle_file:
            if self.synopsis:
                with open(
                        self.data_synopsis_path + os.sep + jf['json_file'] + '.pkl',
                        'rb') as f:
                    existing_indexes = pickle.load(f)
                    existing_indexes.sort()
                    tmp_tensor_list = []
                    tmp_tensor = torch.load(data_scene_path + os.sep + jf['json_file'] + '.pt')
                    for ex_idx in existing_indexes:
                        tmp_tensor_list.append(tmp_tensor[ex_idx])
                    final_tensor = torch.stack(tmp_tensor_list)
                    self.data_scene.append(final_tensor)
            else:
                self.data_scene.append(torch.load(data_scene_path + os.sep + jf['json_file'] + '.pt'))
            self.data_description.append(torch.load(data_description_path + os.sep + jf['json_file'] + '.pt'))

    def __len__(self):
        return len(self.data_scene)

    def __getitem__(self, index):
        try:
            file_path = './processed_index.txt'
            with open(file_path, 'a') as file:
                new_content = str(index)
                file.write("\n" + new_content)

            data_description = self.data_description[index]
            data_scene = self.data_scene[index]
            if self.type_model_scene == Constants.model_scene_mean:
                data_scene = torch.mean(data_scene, 0)
            return data_description, data_scene
        except:
            print('index', index)
            exit(0)


class DescriptionSceneDatasetMemClassifier(Dataset):
    def __init__(self, data_description_path, data_scene_path, type_model_scene):
        self.description_path = data_description_path
        self.data_scene = []
        self.data_description = []
        self.data_tag = []
        self.type_model_scene = type_model_scene
        with open(
                '../houses_data/houses_data.pkl',
                'rb') as f:
            self.pickle_file = pickle.load(f)
        with open('../yolo/images_tags_info_customdataset_v8.pkl', 'rb') as f:
            self.tag_pickle_file = pickle.load(f)
        names = ['Bed', 'Cabinet', 'Carpet', 'Ceramic floor', 'Chair', 'Closet', 'Cupboard', 'Curtains', 'Dining Table',
                 'Door', 'Frame', 'Futec frame', 'Futech tiles', 'Gypsum Board', 'Lamp', 'Nightstand', 'Shelf',
                 'Sideboard', 'Sofa', 'TV stand', 'Table', 'Transparent Closet', 'Wall Panel', 'Window', 'Wooden floor']
        for jf in self.pickle_file:
            self.data_scene.append(torch.load(data_scene_path + os.sep + jf['json_file'] + '.pt'))
            self.data_description.append(torch.load(data_description_path + os.sep + jf['json_file'] + '.pt'))
            images_tag_list = []
            for item in self.tag_pickle_file:
                if item['json_file'] == jf['json_file']:
                    for imgtag in item['image_tags']:
                        if imgtag['tag'] is None:
                            images_tag_list.append(25)
                        else:
                            images_tag_list.append(names.index(imgtag['tag']))
                    self.data_tag.append(images_tag_list)
                    break

    def __len__(self):
        return len(self.data_scene)

    def __getitem__(self, index):
        try:
            file_path = './processed_index.txt'
            with open(file_path, 'a') as file:
                new_content = str(index)
                file.write("\n" + new_content)

            data_description = self.data_description[index]
            data_scene = self.data_scene[index]
            data_tag = self.data_tag[index]
            if self.type_model_scene == Constants.model_scene_mean:
                data_scene = torch.mean(data_scene, 0)
            return data_description, data_scene, data_tag
        except:
            print('index', index)
            exit(0)
