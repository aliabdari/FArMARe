import os
import random
import torch
from torch.utils.data import Dataset
import pickle
import Constants


# Scene and Description dataset
class DatasetMean(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        return x1, x2


class DescriptionSceneDataset(Dataset):
    def __init__(self, data_description_path, data_scene_path, type_model_scene):
        self.description_path = data_description_path
        self.data_scene = data_scene_path
        self.type_model_scene = type_model_scene
        with open(
                '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/houses_data/houses_data.pkl',
                'rb') as f:
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
    def __init__(self, data_description_path, data_scene_path, type_model_scene, is_synopsis=False, data_synopsis_path=None):
        self.description_path = data_description_path
        self.data_scene = []
        self.data_description = []
        self.type_model_scene = type_model_scene
        self.synopsis = is_synopsis
        self.data_synopsis_path = data_synopsis_path
        with open(
                '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/houses_data/houses_data.pkl',
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
    def __init__(self, data_description_path, data_scene_path, type_model_scene, is_synopsis=False, data_synopsis_path=None):
        self.description_path = data_description_path
        self.data_scene = []
        self.data_description = []
        self.data_tag = []
        self.type_model_scene = type_model_scene
        self.synopsis = is_synopsis
        self.data_synopsis_path = data_synopsis_path
        with open(
                '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/houses_data/houses_data.pkl',
                'rb') as f:
            self.pickle_file = pickle.load(f)
        with open(
                '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/yolo/images_tags_info_customdataset_v8.pkl',
                'rb') as f:
            self.tag_pickle_file = pickle.load(f)
        names = ['Bed', 'Cabinet', 'Carpet', 'Ceramic floor', 'Chair', 'Closet', 'Cupboard', 'Curtains', 'Dining Table',
                 'Door', 'Frame', 'Futec frame', 'Futech tiles', 'Gypsum Board', 'Lamp', 'Nightstand', 'Shelf',
                 'Sideboard', 'Sofa', 'TV stand', 'Table', 'Transparent Closet', 'Wall Panel', 'Window', 'Wooden floor']
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


class DescriptionSceneDatasetMemTags(Dataset):
    def __init__(self, data_description_path, data_scene_path, data_tag_path, type_model_scene, is_late_fusion=False):
        self.data_scene = []
        self.data_description = []
        self.data_tag = []
        self.type_model_scene = type_model_scene
        self.is_late_fusion = is_late_fusion
        with open(
                '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/houses_data/houses_data.pkl',
                'rb') as f:
            self.pickle_file = pickle.load(f)
        for jf in self.pickle_file:
            self.data_scene.append(torch.load(data_scene_path + os.sep + jf['json_file'] + '.pt'))
            self.data_description.append(torch.load(data_description_path + os.sep + jf['json_file'] + '.pt'))
            self.data_tag.append(torch.load(data_tag_path + os.sep + jf['json_file'] + '.pt'))

    def __len__(self):
        return len(self.data_scene)

    def __getitem__(self, index):
        try:
            data_description = self.data_description[index]
            data_scene = self.data_scene[index]
            data_tag = self.data_tag[index]
            if not data_scene.shape == data_tag.shape:
                print('ERROR')
                exit(0)
            if self.is_late_fusion:
                return data_description, data_scene, data_tag
            entire_data_scene = torch.cat((data_scene, data_tag), 1)
            if self.type_model_scene == Constants.model_scene_mean:
                entire_data_scene = torch.mean(entire_data_scene, 0)
            return data_description, entire_data_scene
        except:
            print(index)
            exit(0)


class DescriptionSceneDatasetHouseMean(Dataset):
    def __init__(self, data_description_path, data_scene_living, data_scene_bedroom, no_house, type_model_scene):
        self.description_path = data_description_path
        self.data_scene_living = data_scene_living
        self.data_scene_bedroom = data_scene_bedroom
        self.no_houses = no_house
        self.type_model_scene = type_model_scene

        root_path = '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/dataset_3dfront/outputs_house'
        houses_data = open(root_path + f'/created_house_dataset_indexes_{self.no_houses}.pkl', 'rb')
        self.houses_data = pickle.load(houses_data)

    def __len__(self):
        return len(self.houses_data)

    def __getitem__(self, index):
        data_description = torch.load(self.description_path + os.sep + "desc_" + str(index.item()) + ".pt")

        retrieve_orig = self.houses_data[index]
        livingroom_idx = retrieve_orig['living_room']
        bedrooms_idx = retrieve_orig['bedroom']

        data_scene_ = torch.zeros(1 + len(bedrooms_idx), self.data_scene_bedroom.size()[1])
        data_scene_[0, :] = self.data_scene_living[livingroom_idx]
        for i, j in enumerate(bedrooms_idx):
            data_scene_[i + 1, :] = self.data_scene_bedroom[j]
        if self.type_model_scene == 'mean':
            data_scene_ = torch.mean(data_scene_, 0)

        return data_description, data_scene_


class DatasetsRecVersionRandom(Dataset):
    def __init__(self, data_description_path, data_scene):
        self.description_path = data_description_path
        self.data_scene = data_scene

    def __len__(self):
        return len(self.data_scene)

    def __getitem__(self, index):
        x1 = torch.load(self.description_path + os.sep + "desc_" + str(index.item()) + ".pt")
        no_of_sentences = random.randrange(1, x1.size()[0] - 1)
        selected_list = random.sample(range(0, x1.size()[0] - 1), no_of_sentences)
        x1_new = x1[selected_list]
        x2 = self.data_scene[index]
        return x1_new, x2


class DatasetsRecVersionV2(Dataset):
    def __init__(self, data_scene):
        self.data_scene = data_scene

    def __len__(self):
        return len(self.data_scene)

    def __getitem__(self, index):
        print(index.item())
        x = self.data_scene[index]
        return x


class DatasetsRecIncludingVideo(Dataset):
    def __init__(self, data_description_path, data_scene, representation_video):
        self.description_path = data_description_path
        self.data_scene = data_scene
        self.representation_video = representation_video

    def __len__(self):
        return len(self.data_scene)

    def __getitem__(self, index):
        x1 = torch.load(self.description_path + os.sep + "desc_" + str(index.item()) + ".pt")
        x2_1 = self.data_scene[index]
        x2_2 = self.representation_video[index % 500][:]
        x2 = torch.cat((x2_1, x2_2), dim=0)
        return x1, x2
