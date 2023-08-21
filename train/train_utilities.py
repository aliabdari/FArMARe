import os
import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from transformers import BertTokenizer, BertModel
import spacy
from textblob import Word
import matplotlib.pyplot as plt
import Constants


def get_entire_data(clip_ones, token_level, is_tag_need, is_synopsis):
    roots = {}
    if clip_ones:
        roots[Constants.root_scene_path] = '../open_clip_features/images'
        if not token_level:
            roots[Constants.root_description_path] = '../open_clip_features/descriptions/sentences'
        else:
            roots[Constants.root_description_path] = '../open_clip_features/descriptions/tokens'
        if is_tag_need:
            roots[Constants.root_tag_path] = '../yolo/images_tags_info_customdataset_v8'
        if is_synopsis:
            roots[Constants.root_synopsis_path] = '../open_clip_features/images_synopsis_index'
    else:
        if not token_level:
            roots[Constants.root_description_path] = '../bert_features/sentences'
        else:
            roots[Constants.root_description_path] = '../bert_features/tokens'
        roots[Constants.root_scene_path] = '../imagery_features'

    return roots


def get_scene_data(data_scene, indices, type_model_scene):
    root_path = data_scene
    with open('/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/houses_data/houses_data.pkl',
            'rb') as f:
        pickle_file = pickle.load(f)

    data_scene_list = []
    list_length = []
    if indices is not None:
        if type_model_scene == Constants.model_scene_mean:
            tmp_scene = torch.load(root_path + os.sep + pickle_file[0]['json_file'] + '.pt')
            data_scene = torch.empty(len(indices), tmp_scene.shape[1])
        index = 0
        for idx in indices:
            data_scene_ = torch.load(root_path + os.sep + pickle_file[idx]['json_file'] + '.pt')
            if type_model_scene == 'mean':
                data_scene_ = torch.mean(data_scene_, 0)
                data_scene[index, :] = data_scene_
                index += 1
            elif type_model_scene == Constants.model_scene_onedimensional:
                # getting the padding scenes
                data_scene_list.append(data_scene_)
    else:
        tmp_scene = torch.load(root_path + os.sep + pickle_file[0]['json_file'] + '.pt')
        data_scene = torch.empty(len(pickle_file), tmp_scene.shape[1])
        index = 0
        for item in pickle_file:
            data_scene_ = torch.load(root_path + os.sep + item['json_file'] + '.pt')
            if type_model_scene == Constants.model_scene_mean:
                data_scene_ = torch.mean(data_scene_, 0)
                data_scene[index, :] = data_scene_
                index += 1
            elif type_model_scene == Constants.model_scene_onedimensional:
                data_scene_list.append(data_scene_)
    if type_model_scene == Constants.model_scene_onedimensional:
        data_scene = pad_sequence(data_scene_list, batch_first=True)
        list_length = [len(x) for x in data_scene_list]
        data_scene = torch.transpose(data_scene, 1, 2)
    return data_scene, list_length


def get_scene_data_with_tag(data_scene, data_tag, indices, type_model_scene):
    root_path_scene = data_scene
    root_path_tag = data_tag
    with open('/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/houses_data/houses_data.pkl',
            'rb') as f:
        pickle_file = pickle.load(f)

    entire_data_scene_list = []
    list_length = []
    if indices is not None:
        if type_model_scene == Constants.model_scene_mean:
            tmp_scene = torch.load(root_path_scene + os.sep + pickle_file[0]['json_file'] + '.pt')
            tmp_tag = torch.load(root_path_tag + os.sep + pickle_file[0]['json_file'] + '.pt')
            tmp_entire_data_scene = torch.cat((tmp_scene, tmp_tag), 1)
            entire_data_scene = torch.empty(len(indices), tmp_entire_data_scene.shape[1])
        index = 0
        for idx in indices:
            data_scene_ = torch.load(root_path_scene + os.sep + pickle_file[idx]['json_file'] + '.pt')
            data_tag_ = torch.load(root_path_scene + os.sep + pickle_file[idx]['json_file'] + '.pt')
            entire_data_scene_ = torch.cat((data_scene_, data_tag_), 1)
            if type_model_scene == Constants.model_scene_mean:
                entire_data_scene_ = torch.mean(entire_data_scene_, 0)
                entire_data_scene[index, :] = entire_data_scene_
                index += 1
            elif type_model_scene == Constants.model_scene_onedimensional:
                # getting the padding scenes
                entire_data_scene_list.append(entire_data_scene_)
    if type_model_scene == Constants.model_scene_onedimensional:
        entire_data_scene = pad_sequence(entire_data_scene_list, batch_first=True)
        list_length = [len(x) for x in entire_data_scene_list]
        entire_data_scene = torch.transpose(entire_data_scene, 1, 2)
    return entire_data_scene, list_length


# def get_scene_data_with_tag_late_fusion(data_scene, data_tag, indices, type_model_scene):
#     root_path_scene = data_scene
#     root_path_tag = data_tag
#     with open('/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/houses_data/houses_data.pkl',
#             'rb') as f:
#         pickle_file = pickle.load(f)
#
#     entire_data_scene_list = []
#     data_scene_list = []
#     data_tag_list = []
#     list_length = []
#     if indices is not None:
#         if type_model_scene == Constants.model_scene_mean:
#             tmp_scene = torch.load(root_path_scene + os.sep + pickle_file[0]['json_file'] + '.pt')
#             tmp_tag = torch.load(root_path_tag + os.sep + pickle_file[0]['json_file'] + '.pt')
#             # tmp_entire_data_scene = torch.cat((tmp_scene, tmp_tag), 1)
#             data_scene = torch.empty(len(indices), tmp_scene.shape[1])
#             data_tag = torch.empty(len(indices), tmp_tag.shape[1])
#         index = 0
#         for idx in indices:
#             data_scene_ = torch.load(root_path_scene + os.sep + pickle_file[idx]['json_file'] + '.pt')
#             data_tag_ = torch.load(root_path_scene + os.sep + pickle_file[idx]['json_file'] + '.pt')
#             entire_data_scene_ = torch.cat((data_scene_, data_tag_), 1)
#             if type_model_scene == Constants.model_scene_mean:
#                 entire_data_scene_ = torch.mean(entire_data_scene_, 0)
#                 entire_data_scene[index, :] = entire_data_scene_
#                 index += 1
#             elif type_model_scene == Constants.model_scene_onedimensional:
#                 # getting the padding scenes
#                 entire_data_scene_list.append(entire_data_scene_)
#                 data_scene_list.append(data_scene_)
#                 data_tag_list.append(data_tag_)
#     else:
#         tmp_scene = torch.load(root_path + os.sep + pickle_file[0]['json_file'] + '.pt')
#         data_scene = torch.empty(len(pickle_file), tmp_scene.shape[1])
#         index = 0
#         for item in pickle_file:
#             data_scene_ = torch.load(root_path + os.sep + item['json_file'] + '.pt')
#             if type_model_scene == Constants.model_scene_mean:
#                 data_scene_ = torch.mean(data_scene_, 0)
#                 data_scene[index, :] = data_scene_
#                 index += 1
#             elif type_model_scene == Constants.model_scene_onedimensional:
#                 data_scene_list.append(data_scene_)
#     if type_model_scene == Constants.model_scene_onedimensional:
#         data_scene_list = pad_sequence(data_scene_list, batch_first=True)
#         data_tag_list = pad_sequence(data_tag_list, batch_first=True)
#         list_length = [len(x) for x in entire_data_scene_list]
#     return data_scene_list, data_tag_list, list_length


def retrieve_indices():
    if os.path.isfile('indices/indices.pkl'):
        indices_pickle = open('indices/indices.pkl', "rb")
        indices_pickle = pickle.load(indices_pickle)
        train_indices = indices_pickle["train"]
        val_indices = indices_pickle["val"]
        test_indices = indices_pickle["test"]
    else:
        with open('../houses_data/houses_data.pkl','rb') as f:
            pickle_file = pickle.load(f)
        data_size = len(pickle_file)
        train_ratio = .7
        val_ratio = .15
        perm = torch.randperm(data_size)
        train_indices = perm[:int(data_size * train_ratio)]
        val_indices = perm[int(data_size * train_ratio):int(data_size * (val_ratio + train_ratio))]
        test_indices = perm[int(data_size * (val_ratio + train_ratio)):]
        indices_pickle = {"train": train_indices, "val": val_indices, "test": test_indices}
        with open('indices/indices.pkl', 'wb') as f:
            pickle.dump(indices_pickle, f)
    return train_indices, val_indices, test_indices


def create_percent_queries(scene_result, entire_descriptor, desired_output_indexes):
    similarity = torch.nn.functional.cosine_similarity(entire_descriptor, scene_result, dim=1)
    similarity = similarity.squeeze()
    sorted_indices = torch.argsort(similarity, descending=True)
    sorted_indices = sorted_indices.tolist()
    r_50 = 0
    r_10 = 0
    r_5 = 0
    r_1 = 0

    for j in sorted_indices[:50]:
        if j in desired_output_indexes:
            r_50 = 1

    for j in sorted_indices[:10]:
        if j in desired_output_indexes:
            r_10 = 1

    for j in sorted_indices[:5]:
        if j in desired_output_indexes:
            r_5 = 1

    for j in sorted_indices[:1]:
        if j in desired_output_indexes:
            r_1 = 1

    return r_50, r_10, r_5, r_1


# def dcg_idcg_calculator(n, desired_output_indexes, sorted_indices):
#     dcg = 0
#     idcg = 0
#     if n == "entire":
#         length_desired_list = len(desired_output_indexes)
#         for i, j in enumerate(sorted_indices[:length_desired_list]):
#             idcg += 1 / math.log2(i + 2)
#             if j in desired_output_indexes:
#                 dcg += 1 / math.log2(i + 2)
#     else:
#         length_desired_list = min(len(desired_output_indexes), n)
#         for i, j in enumerate(sorted_indices[:length_desired_list]):
#             idcg += 1 / math.log2(i + 2)
#             if j in desired_output_indexes:
#                 dcg += 1 / math.log2(i + 2)
#     return dcg / idcg


# def dcg_idcg_calculator_v2(n, desired_output_indexes, sorted_indices):
#     dcg = 0
#     idcg = 0
#     length_desired_list = len(desired_output_indexes)
#     if length_desired_list < 5:
#         return None
#     if n == "entire":
#         for i, j in enumerate(sorted_indices[:length_desired_list]):
#             idcg += 1 / math.log2(i + 2)
#             if j in desired_output_indexes:
#                 dcg += 1 / math.log2(i + 2)
#     else:
#         if length_desired_list < n:
#             return None
#         else:
#             for i, j in enumerate(sorted_indices[:n]):
#                 idcg += 1 / math.log2(i + 2)
#                 if j in desired_output_indexes:
#                     dcg += 1 / math.log2(i + 2)
#     return dcg / idcg


# def get_desired_relevant(selected_desc_index, no_of_videos, section, type_room):
#     with open('/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/network/relevances/relevance_'
#               + str(no_of_videos)
#               + '_' + section
#               + '_' + type_room + '.pkl', 'rb') as f:
#         info = pickle.load(f)
#     return info[selected_desc_index]


# def get_desired_relevant_house(selected_desc_index, section, coef, thresh):
#     if no_houses in [2000, 6000, 20000]:
#         with open('/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/network/relevances/relevance_house'
#                   + '_' + section
#                   + '_' + str(no_houses) + '.pkl', 'rb') as f:
#             info = pickle.load(f)
#             info = info[selected_desc_index]
#     elif no_houses == 50000:
#         with open('/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/network/relevance_house_50000/house'
#                   + '_' + section
#                   + '_' + no_houses
#                   + '_' + selected_desc_index + '.pkl', 'rb') as f:
#             info = pickle.load(f)
#
#     desired_list = []
#     for idx in info:
#         relevance_measure = idx['r_c'] + coef * idx['r_b']
#         relevance_measure_final = (relevance_measure - (idx['min_q'] * coef)) / ((idx['max_q'] + coef) - (idx['min_q'] * coef))
#         # print('relevance_measure:', relevance_measure_final)
#         if relevance_measure_final > 1:
#             print('idx', idx)
#             exit(0)
#         if relevance_measure_final >= thresh:
#             desired_list.append(idx['index'])
#     print('len desired list', len(desired_list))
#
#     return desired_list


def save_best_model(best_model_state_dict_scene, best_model_state_dict_description, model_name):
    model_path = "models"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path = model_path + os.sep + model_name
    torch.save({'model_state_dict_scene': best_model_state_dict_scene,
                'model_state_dict_description': best_model_state_dict_description},
               model_path)


def load_best_model(model_name):
    model_path = "models"
    model_path = model_path + os.sep + model_name
    check_point = torch.load(model_path)
    best_model_state_dict_scene = check_point['model_state_dict_scene']
    best_model_state_dict_description = check_point['model_state_dict_description']
    return best_model_state_dict_scene, best_model_state_dict_description


def write_train_history_to_file(data, file_name):
    path = './results'
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + os.sep + file_name
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'w') as f:
        for d in data:
            f.write(str(d))
            f.write('\n')


def write_models_evaluation_to_file(data, file_name):
    path = './results'
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + os.sep + file_name
    tags = ["ds1", "ds5", "ds10", "sd1", "sd5", "sd10", "ndcg_10", "ndcg", "median_rank_ds", "median_rank_sd"]
    with open(path, 'a') as f:
        for i, d in enumerate(data):
            f.write(tags[i] + " : " + str(d))
            f.write('\n')
        f.write('-'*30)
        f.write('\n')


def write_queries_eval_to_file(data, file_name):
    path = './results'
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + os.sep + file_name
    if os.path.exists(path):
        os.remove(path)
    tags = ["r1", "r5", "r10", "ndcg_10", "ndcg"]
    with open(path, 'w') as f:
        for i, d in enumerate(data):
            f.write(tags[i] + " : " + str(d))
            f.write('\n')


def create_rank(result, entire_descriptor, desired_output_index):
    similarity = torch.nn.functional.cosine_similarity(entire_descriptor, result, dim=1)
    similarity = similarity.squeeze()
    sorted_indices = torch.argsort(similarity, descending=True)
    position = torch.where(sorted_indices == desired_output_index)
    return position[0].item(), sorted_indices


def get_related_descriptions(data_description_path, indices, type_model_desc):
    tensor_lists = []
    if indices is not None:
        if type_model_desc == 'mean':
            data_description = torch.empty(len(indices), 768)
            index = 0
            for idx in indices:
                data_description[index, :] = torch.mean((torch.load(data_description_path + os.sep + "desc_" + str(idx.item()) + ".pt")), 0)
                index += 1
            return data_description
        else:
            for idx in indices:
                tensor_lists.append(torch.load(data_description_path + os.sep + "desc_" + str(idx.item()) + ".pt"))
    else:
        number_of_files = 0
        for file in os.listdir(data_description_path):
            if file.endswith(".pt"):
                number_of_files += 1
        if type_model_desc == 'mean':
            data_description = torch.empty(number_of_files, 768)
            index = 0
            for idx in range(number_of_files):
                data_description[index, :] = torch.mean((torch.load(data_description_path + os.sep + "desc_" + str(idx) + ".pt")), 0)
                index += 1
            return data_description
        else:
            for idx in range(number_of_files):
                tensor_lists.append(torch.load(data_description_path + os.sep + "desc_" + str(idx) + ".pt"))
    return tensor_lists


# def get_related_descriptions_combined(data_description_path_living, data_description_path_bedroom, indices, type_model_desc):
#     tensor_lists = []
#
#     with open('indices/indices_combined_guidance.pkl', 'rb') as f:
#         indices_guidance = pickle.load(f)
#
#     if indices is not None:
#         if type_model_desc == 'mean':
#             data_description = torch.empty(len(indices), 768)
#             index = 0
#             for idx in indices:
#                 retrieve_orig = indices_guidance[idx]
#                 if retrieve_orig['type'] == 'living':
#                     data_description[index, :] = torch.mean(torch.load(data_description_path_living + os.sep + "desc_" + str(retrieve_orig['index']) + ".pt"), 0)
#                     index += 1
#                 elif retrieve_orig['type'] == 'bedroom':
#                     data_description[index, :] = torch.mean(torch.load(data_description_path_bedroom + os.sep + "desc_" + str(retrieve_orig['index']) + ".pt"), 0)
#                     index += 1
#             return data_description
#         else:
#             for idx in indices:
#                 retrieve_orig = indices_guidance[idx]
#                 if retrieve_orig['type'] == 'living':
#                     tensor_lists.append(torch.load(data_description_path_living + os.sep + "desc_" + str(retrieve_orig['index']) + ".pt"))
#                 elif retrieve_orig['type'] == 'bedroom':
#                     tensor_lists.append(torch.load(data_description_path_bedroom + os.sep + "desc_" + str(retrieve_orig['index']) + ".pt"))
#             return tensor_lists
#     else:
#         number_of_files_living = 0
#         number_of_files_bedroom = 0
#         for file in os.listdir(data_description_path_living):
#             if file.endswith(".pt"):
#                 number_of_files_living += 1
#         for file in os.listdir(data_description_path_bedroom):
#             if file.endswith(".pt"):
#                 number_of_files_bedroom += 1
#
#         if type_model_desc == 'mean':
#             data_description = torch.empty((number_of_files_living + number_of_files_bedroom), 768)
#             index = 0
#             for idx in range(number_of_files_living):
#                 data_description[index, :] = torch.mean(torch.load(data_description_path_living + os.sep + "desc_" + str(idx) + ".pt"), 0)
#                 index += 1
#             for idx in range(number_of_files_bedroom):
#                 data_description[index, :] = torch.mean(torch.load(data_description_path_bedroom + os.sep + "desc_" + str(idx) + ".pt"), 0)
#                 index += 1
#             return data_description
#         else:
#             for idx in range(number_of_files_living):
#                 tensor_lists.append(torch.load(data_description_path_living + os.sep + "desc_" + str(idx) + ".pt"))
#             for idx in range(number_of_files_bedroom):
#                 tensor_lists.append(torch.load(data_description_path_bedroom + os.sep + "desc_" + str(idx) + ".pt"))
#             return tensor_lists


# def get_related_descriptions_house(data_description_path, indices, type_model_desc):
#     with open('/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/houses_data/houses_data.pkl',
#             'rb') as f:
#         pickle_file = pickle.load(f)
#
#     tensor_lists = []
#
#     if indices is not None:
#         if type_model_desc == 'mean':
#             data_description = torch.empty(len(indices), 768)
#             index = 0
#             for idx in indices:
#                 retrieve_orig = indices_guidance[idx]
#                 if retrieve_orig['type'] == 'living':
#                     data_description[index, :] = torch.mean(torch.load(data_description_path_living + os.sep + "desc_" + str(retrieve_orig['index']) + ".pt"), 0)
#                     index += 1
#                 elif retrieve_orig['type'] == 'bedroom':
#                     data_description[index, :] = torch.mean(torch.load(data_description_path_bedroom + os.sep + "desc_" + str(retrieve_orig['index']) + ".pt"), 0)
#                     index += 1
#             return data_description
#         else:
#             for idx in indices:
#                 tensor_lists.append(torch.load(data_description_path + os.sep + pickle_file[idx]['json_file'] + ".pt"))
#             return tensor_lists
#     else:
#         number_of_files = 0
#         for file in os.listdir(data_description_path):
#             if file.endswith(".pt"):
#                 number_of_files += 1
#
#         if type_model_desc == 'mean':
#             data_description = torch.empty(number_of_files, 768)
#             index = 0
#             for idx in range(number_of_files):
#                 data_description[index, :] = torch.mean(torch.load(data_description_path + os.sep + "desc_" + str(idx) + ".pt"), 0)
#                 index += 1
#             return data_description
#         else:
#             for idx in range(number_of_files):
#                 tensor_lists.append(torch.load(data_description_path + os.sep + "desc_" + str(idx) + ".pt"))
#             return tensor_lists


# def get_related_scenes_house(data_scene_path, indices, type_model_scene):
#
#     tensor_lists = []
#
#     if indices is not None:
#         if type_model_scene == 'mean':
#             pass
#         elif type_model_scene == 'oneDimensional':
#             pass
#     else:
#         number_of_files = 0
#         for file in os.listdir(data_description_path):
#             if file.endswith(".pt"):
#                 number_of_files += 1
#
#         if type_model_desc == 'mean':
#             data_description = torch.empty(number_of_files, 768)
#             index = 0
#             for idx in range(number_of_files):
#                 data_description[index, :] = torch.mean(torch.load(data_description_path + os.sep + "desc_" + str(idx) + ".pt"), 0)
#                 index += 1
#             return data_description
#         else:
#             for idx in range(number_of_files):
#                 tensor_lists.append(torch.load(data_description_path + os.sep + "desc_" + str(idx) + ".pt"))
#             return tensor_lists


def get_list_tensor_description(data_description_path, indices):

    tensor_lists = get_related_descriptions(data_description_path=data_description_path,
                                            indices=indices,
                                            type_model_desc='gru')

    tmp = pad_sequence(tensor_lists, batch_first=True)
    desc_ = pack_padded_sequence(tmp,
                                 torch.tensor([len(x) for x in tensor_lists]),
                                 batch_first=True,
                                 enforce_sorted=False)
    return desc_


def get_list_tensor_description_gru_house(data_description_path, indices):

    tensor_lists = get_related_descriptions_house(data_description_path=data_description_path,
                                                  indices=indices,
                                                  type_model_desc='gru')

    tmp = pad_sequence(tensor_lists, batch_first=True)
    desc_ = pack_padded_sequence(tmp,
                                 torch.tensor([len(x) for x in tensor_lists]),
                                 batch_first=True,
                                 enforce_sorted=False)
    return desc_


def get_list_tensor_description_gru_combined(data_description_path_living, data_description_path_bedroom, indices):
    tensor_lists = get_related_descriptions_combined(data_description_path_living=data_description_path_living,
                                                     data_description_path_bedroom=data_description_path_bedroom,
                                                     indices=indices,
                                                     type_model_desc='gru')

    tmp = pad_sequence(tensor_lists, batch_first=True)
    desc_ = pack_padded_sequence(tmp,
                                 torch.tensor([len(x) for x in tensor_lists]),
                                 batch_first=True,
                                 enforce_sorted=False)
    return desc_


def get_list_tensor_description_self_attention(data_description_path, indices):

    tensor_lists = get_related_descriptions(data_description_path=data_description_path,
                                            indices=indices,
                                            type_model_desc="self_attention")

    desc_ = pad_sequence(tensor_lists, batch_first=True)

    lengths = torch.tensor([len(x) for x in tensor_lists])
    max_length = lengths.max().item()
    range_tensor = torch.arange(max_length)
    # Create a mask tensor by comparing the range tensor to the length of each unpadded tensor
    mask = range_tensor.unsqueeze(0) < lengths.unsqueeze(1)
    mask = ~mask

    return desc_, mask


def get_list_tensor_description_self_attention_combined(data_description_path_living, data_description_path_bedroom, indices):
    tensor_lists = get_related_descriptions_combined(data_description_path_living=data_description_path_living,
                                                     data_description_path_bedroom=data_description_path_bedroom,
                                                     indices=indices,
                                                     type_model_desc="self_attention")

    desc_ = pad_sequence(tensor_lists, batch_first=True)

    lengths = torch.tensor([len(x) for x in tensor_lists])
    max_length = lengths.max().item()
    range_tensor = torch.arange(max_length)
    # Create a mask tensor by comparing the range tensor to the length of each unpadded tensor
    mask = range_tensor.unsqueeze(0) < lengths.unsqueeze(1)
    mask = ~mask

    return desc_, mask


class MILNCELoss(torch.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, pairwise_similarity):
        nominator = pairwise_similarity * torch.eye(pairwise_similarity.shape[0]).cuda()
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((pairwise_similarity,
                                 pairwise_similarity.permute(1, 0)), dim=1).view(pairwise_similarity.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        return torch.mean(denominator - nominator)


def evaluate(output_description, output_scene, section):
    avg_rank_scene = 0
    ranks_scene = []
    avg_rank_description = 0
    ranks_description = []

    ndcg_10_list = []
    ndcg_entire_list = []

    for j, i in enumerate(output_scene):
        rank, sorted_list = create_rank(i, output_description, j)
        avg_rank_scene += rank
        ranks_scene.append(rank)

    for j, i in enumerate(output_description):
        rank, sorted_list = create_rank(i, output_scene, j)
        avg_rank_description += rank
        ranks_description.append(rank)

    ranks_scene = np.array(ranks_scene)
    ranks_description = np.array(ranks_description)

    n_q = len(output_scene)
    sd_r1 = 100 * len(np.where(ranks_scene < 1)[0]) / n_q
    sd_r5 = 100 * len(np.where(ranks_scene < 5)[0]) / n_q
    sd_r10 = 100 * len(np.where(ranks_scene < 10)[0]) / n_q
    sd_medr = np.median(ranks_scene) + 1
    sd_meanr = ranks_scene.mean() + 1

    n_q = len(output_description)
    ds_r1 = 100 * len(np.where(ranks_description < 1)[0]) / n_q
    ds_r5 = 100 * len(np.where(ranks_description < 5)[0]) / n_q
    ds_r10 = 100 * len(np.where(ranks_description < 10)[0]) / n_q
    ds_medr = np.median(ranks_description) + 1
    ds_meanr = ranks_description.mean() + 1

    ds_out, sc_out = "", ""
    for mn, mv in [["R@1", ds_r1],
                   ["R@5", ds_r5],
                   ["R@10", ds_r10],
                   ["median rank", ds_medr],
                   ["mean rank", ds_meanr],
                   ]:
        ds_out += f"{mn}: {mv:.4f}   "

    for mn, mv in [("R@1", sd_r1),
                   ("R@5", sd_r5),
                   ("R@10", sd_r10),
                   ("median rank", sd_medr),
                   ("mean rank", sd_meanr),
                   ]:
        sc_out += f"{mn}: {mv:.4f}   "

    print(section + " data: ")
    print("Scenes ranking: " + ds_out)
    print("Descriptions ranking: " + sc_out)
    if section == "test" and len(ndcg_10_list) > 0:
        avg_ndcg_10_entire = 100 * sum(ndcg_10_list) / len(ndcg_10_list)
        avg_ndcg_entire = 100 * sum(ndcg_entire_list) / len(ndcg_entire_list)
    else:
        avg_ndcg_10_entire = -1
        avg_ndcg_entire = -1

    return ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, avg_ndcg_10_entire, avg_ndcg_entire, ds_medr, sd_medr


# def evaluate_model(model_descriptor, model_scene, data_description, data_scene, data_tag,
#                          type_model_desc, type_model_scene, section, indices, is_tag_used):
#     model_descriptor = model_descriptor.eval()
#     model_scene = model_scene.eval()
#
#     if type_model_desc in ['gru', 'bigru']:
#         data_description = get_list_tensor_description_gru_house(data_description_path=data_description,
#                                                                  indices=indices)
#         if not is_tag_used:
#             data_scene, list_length = get_scene_data(data_scene=data_scene,
#                                                      indices=indices,
#                                                      type_model_scene=type_model_scene)
#         else:
#             data_scene, list_length = get_scene_data_with_tag(data_scene=data_scene,
#                                                               data_tag=data_tag,
#                                                               indices=indices,
#                                                               type_model_scene=type_model_scene)
#     output_description = model_descriptor(data_description.cuda())
#
#     if type_model_scene == Constants.model_scene_mean:
#         output_scene = model_scene(data_scene.cuda())
#     elif type_model_scene == Constants.model_scene_onedimensional:
#         output_scene = model_scene(data_scene.cuda(), list_length)
#
#     avg_rank_scene = 0
#     ranks_scene = []
#     avg_rank_description = 0
#     ranks_description = []
#
#     ndcg_10_list = []
#     ndcg_entire_list = []
#
#     for j, i in enumerate(output_scene):
#         rank, sorted_list = create_rank(i, output_description, j)
#         avg_rank_scene += rank
#         ranks_scene.append(rank)
#
#     for j, i in enumerate(output_description):
#         rank, sorted_list = create_rank(i, output_scene, j)
#         avg_rank_description += rank
#         ranks_description.append(rank)
#
#     ranks_scene = np.array(ranks_scene)
#     ranks_description = np.array(ranks_description)
#
#     n_q = len(output_scene)
#     ds_r1 = 100 * len(np.where(ranks_scene < 1)[0]) / n_q
#     ds_r5 = 100 * len(np.where(ranks_scene < 5)[0]) / n_q
#     ds_r10 = 100 * len(np.where(ranks_scene < 10)[0]) / n_q
#     ds_medr = np.median(ranks_scene) + 1
#     ds_meanr = ranks_scene.mean() + 1
#
#     n_q = len(output_description)
#     sd_r1 = 100 * len(np.where(ranks_description < 1)[0]) / n_q
#     sd_r5 = 100 * len(np.where(ranks_description < 5)[0]) / n_q
#     sd_r10 = 100 * len(np.where(ranks_description < 10)[0]) / n_q
#     sd_medr = np.median(ranks_description) + 1
#     sd_meanr = ranks_description.mean() + 1
#
#     ds_out, sc_out = "", ""
#     for mn, mv in [["R@1", ds_r1],
#                    ["R@5", ds_r5],
#                    ["R@10", ds_r10],
#                    ["median rank", ds_medr],
#                    ["mean rank", ds_meanr],
#                    ]:
#         ds_out += f"{mn}: {mv:.4f}   "
#
#     for mn, mv in [("R@1", sd_r1),
#                    ("R@5", sd_r5),
#                    ("R@10", sd_r10),
#                    ("median rank", sd_medr),
#                    ("mean rank", sd_meanr),
#                    ]:
#         sc_out += f"{mn}: {mv:.4f}   "
#
#     print(section + " data: ")
#     print("Scenes ranking: " + ds_out)
#     print("Descriptions ranking: " + sc_out)
#     if section == "test" and len(ndcg_10_list) > 0:
#         avg_ndcg_10_entire = 100 * sum(ndcg_10_list) / len(ndcg_10_list)
#         avg_ndcg_entire = 100 * sum(ndcg_entire_list) / len(ndcg_entire_list)
#     else:
#         avg_ndcg_10_entire = -1
#         avg_ndcg_entire = -1
#
#     if section == 'test':
#         print_results(avg_ndcg_10_entire, avg_ndcg_entire, None, None, None, None)
#
#     model_descriptor = model_descriptor.train()
#     model_scene = model_scene.train()
#
#     return ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, avg_ndcg_10_entire, avg_ndcg_entire, ds_medr, sd_medr


# def evaluate_model_late_fusion(model_descriptor, model_scene, model_scene_2, data_description, data_scene, data_tag,
#                          type_model_desc, type_model_scene, section, indices, is_tag_used):
#     model_descriptor = model_descriptor.eval()
#     model_scene = model_scene.eval()
#
#     if type_model_desc in ['gru', 'bigru']:
#         data_description = get_list_tensor_description_gru_house(data_description_path=data_description,
#                                                                  indices=indices)
#         if not is_tag_used:
#             data_scene, list_length = get_scene_data(data_scene=data_scene,
#                                                      indices=indices,
#                                                      type_model_scene=type_model_scene)
#         else:
#             data_scene, data_tag, list_length = get_scene_data_with_tag_late_fusion(data_scene=data_scene,
#                                                                                     data_tag=data_tag,
#                                                                                     indices=indices,
#                                                                                     type_model_scene=type_model_scene)
#     if type_model_desc == "self_attention":
#         # mask = mask.repeat_interleave(n_heads, 0)
#         output_description = model_descriptor(data_description.cuda(), mask.cuda())
#     else:
#         output_description = model_descriptor(data_description.cuda())
#
#     if type_model_scene == Constants.model_scene_mean:
#         output_scene = model_scene(data_scene.cuda())
#     elif type_model_scene == Constants.model_scene_onedimensional:
#         output_scene_img = model_scene(torch.transpose(data_scene, 1, 2).cuda(), list_length)
#         output_scene_tag = model_scene(torch.transpose(data_tag, 1, 2).cuda(), list_length)
#         # data_tag = torch.transpose(data_tag, 1, 2).cuda()
#         entire_data_scene = torch.cat((output_scene_img, output_scene_tag), 1)
#         output_scene = model_scene_2(entire_data_scene, list_length)
#
#     avg_rank_scene = 0
#     ranks_scene = []
#     avg_rank_description = 0
#     ranks_description = []
#
#     ndcg_10_list = []
#     ndcg_entire_list = []
#
#     for j, i in enumerate(output_scene):
#         rank, sorted_list = create_rank(i, output_description, j)
#         avg_rank_scene += rank
#         ranks_scene.append(rank)
#
#     for j, i in enumerate(output_description):
#         rank, sorted_list = create_rank(i, output_scene, j)
#         avg_rank_description += rank
#         ranks_description.append(rank)
#
#     ranks_scene = np.array(ranks_scene)
#     ranks_description = np.array(ranks_description)
#
#     n_q = len(output_scene)
#     ds_r1 = 100 * len(np.where(ranks_scene < 1)[0]) / n_q
#     ds_r5 = 100 * len(np.where(ranks_scene < 5)[0]) / n_q
#     ds_r10 = 100 * len(np.where(ranks_scene < 10)[0]) / n_q
#     ds_medr = np.median(ranks_scene) + 1
#     ds_meanr = ranks_scene.mean() + 1
#
#     n_q = len(output_description)
#     sd_r1 = 100 * len(np.where(ranks_description < 1)[0]) / n_q
#     sd_r5 = 100 * len(np.where(ranks_description < 5)[0]) / n_q
#     sd_r10 = 100 * len(np.where(ranks_description < 10)[0]) / n_q
#     sd_medr = np.median(ranks_description) + 1
#     sd_meanr = ranks_description.mean() + 1
#
#     ds_out, sc_out = "", ""
#     for mn, mv in [["R@1", ds_r1],
#                    ["R@5", ds_r5],
#                    ["R@10", ds_r10],
#                    ["median rank", ds_medr],
#                    ["mean rank", ds_meanr],
#                    ]:
#         ds_out += f"{mn}: {mv:.4f}   "
#
#     for mn, mv in [("R@1", sd_r1),
#                    ("R@5", sd_r5),
#                    ("R@10", sd_r10),
#                    ("median rank", sd_medr),
#                    ("mean rank", sd_meanr),
#                    ]:
#         sc_out += f"{mn}: {mv:.4f}   "
#
#     print(section + " data: ")
#     print("Scenes ranking: " + ds_out)
#     print("Descriptions ranking: " + sc_out)
#     if section == "test" and len(ndcg_10_list) > 0:
#         avg_ndcg_10_entire = 100 * sum(ndcg_10_list) / len(ndcg_10_list)
#         avg_ndcg_entire = 100 * sum(ndcg_entire_list) / len(ndcg_entire_list)
#     else:
#         avg_ndcg_10_entire = -1
#         avg_ndcg_entire = -1
#
#     if section == 'test':
#         print_results(avg_ndcg_10_entire, avg_ndcg_entire, None, None, None, None)
#
#     model_descriptor = model_descriptor.train()
#     model_scene = model_scene.train()
#
#     return ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, avg_ndcg_10_entire, avg_ndcg_entire, ds_medr, sd_medr


def cosine_sim(im, s):
    '''cosine similarity between all the image and sentence pairs
    '''
    inner_prod = im.mm(s.t())
    im_norm = torch.sqrt((im ** 2).sum(1).view(-1, 1) + 1e-18)
    s_norm = torch.sqrt((s ** 2).sum(1).view(1, -1) + 1e-18)
    sim = inner_prod / (im_norm * s_norm)
    return sim


def print_results(avg_ndcg_10_entire, avg_ndcg_entire, avg_r_10_entire, avg_r_1_entire, avg_r_50_entire,avg_r_5_entire):
    print("avg_r_1_entire:", avg_r_1_entire)
    print("avg_r_5_entire:", avg_r_5_entire)
    print("avg_r_10_entire:", avg_r_10_entire)
    print("avg_r_50_entire:", avg_r_50_entire)
    print("avg_ndcg_10_entire:", avg_ndcg_10_entire)
    print("avg_ndcg_entire:", avg_ndcg_entire)


def analyze_ndcg_queries(scene_result, entire_descriptor, desired_output_indexes):
    similarity = torch.nn.functional.cosine_similarity(entire_descriptor, scene_result, dim=1)
    similarity = similarity.squeeze()
    sorted_indices = torch.argsort(similarity, descending=True)
    sorted_indices = sorted_indices.tolist()

    ndcg_10 = dcg_idcg_calculator(10, desired_output_indexes, sorted_indices)
    ndcg = dcg_idcg_calculator("entire", desired_output_indexes, sorted_indices)

    return ndcg, ndcg_10


def get_embeddings(sentence, device, tokenizer, model_bert):
    with torch.no_grad():
        inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
        inputs = inputs.to(device)
        outputs = model_bert(**inputs)
        sentence_embeddings = outputs.last_hidden_state
        sentence_embeddings = torch.mean(sentence_embeddings, dim=1)
        obtained_tensor = sentence_embeddings
    return obtained_tensor


# def evaluate_style_queries(model_descriptor, model_scene, data_scene, data_scene_living, data_scene_bedroom, indices, type_room, desc_model_type, batch_size):
#     if type_room not in ['combined']:
#         with open(
#                 '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main_data/sentence_features/desc_strings_' + type_room + '.pkl',
#                 'rb') as f:
#             scenes_info = pickle.load(f)
#             scenes_info = [scenes_info[i] for i in indices.numpy()]
#     else:
#         with open(
#                 '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main_data/sentence_features/desc_strings_' + 'living' + '.pkl',
#                 'rb') as f:
#             scenes_info_living = pickle.load(f)
#         with open(
#                 '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main_data/sentence_features/desc_strings_' + 'bedroom' + '.pkl',
#                 'rb') as f:
#             scenes_info_bedroom = pickle.load(f)
#
#             scenes_info = []
#             with open('indices/indices_combined_guidance.pkl', 'rb') as f:
#                 indices_guidance = pickle.load(f)
#             for idx in indices:
#                 retrieve_orig = indices_guidance[idx]
#                 if retrieve_orig['type'] == 'living':
#                     scenes_info.append(scenes_info_living[retrieve_orig['index']])
#                 elif retrieve_orig['type'] == 'bedroom':
#                     scenes_info.append(scenes_info_bedroom[retrieve_orig['index']])
#
#     entire_theme_list = []
#     entire_style_list = []
#     entire_material_list = []
#
#     containing_index = {'material': {}, 'style': {}, 'theme': {}}
#
#     number_of_queries = 0
#
#     for i, s in enumerate(scenes_info):
#         theme_list = []
#         style_list = []
#         material_list = []
#
#         no_object = sum(i['number'] for i in s)
#
#         for o in s:
#             if 'theme' in o.keys():
#                 if o['theme'] is not None and o['theme'] != 'Others':
#                     theme_list.extend([o['theme']] * o['number'])
#             if 'style' in o.keys():
#                 if o['style'] is not None and o['style'] != 'Others':
#                     style_list.extend([o['style']] * o['number'])
#             if 'material' in o.keys():
#                 if o['material'] is not None and o['material'] != 'Others':
#                     material_list.extend([o['material']] * o['number'])
#
#         themes_set = set(theme_list)
#         style_set = set(style_list)
#         material_set = set(material_list)
#
#         for t in themes_set:
#             c = theme_list.count(t)
#             if c * 2 >= no_object:
#                 entire_theme_list.append(t)
#                 if t in containing_index['theme']:
#                     containing_index['theme'][t].append(i)
#                 else:
#                     containing_index['theme'][t] = [i]
#
#         for t in style_set:
#             c = style_list.count(t)
#             if c * 2 >= no_object:
#                 entire_style_list.append(t)
#                 if t in containing_index['style']:
#                     containing_index['style'][t].append(i)
#                 else:
#                     containing_index['style'][t] = [i]
#
#         for t in material_set:
#             c = material_list.count(t)
#             if c * 2 >= no_object:
#                 entire_material_list.append(t)
#                 if t in containing_index['material']:
#                     containing_index['material'][t].append(i)
#                 else:
#                     containing_index['material'][t] = [i]
#
#     print(set(entire_theme_list))
#     print(set(entire_material_list))
#     print(set(entire_style_list))
#
#     device = ("cuda" if torch.cuda.is_available() else "cpu")
#     if type_room not in ['combined']:
#         data_scene = data_scene[indices]
#     else:
#         data_scene = get_scene_data(data_scene_living=data_scene_living,
#                                     data_scene_bedroom=data_scene_bedroom,
#                                     indices=indices)
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model_bert = BertModel.from_pretrained('bert-base-uncased')
#     model_bert.to(device)
#
#     r_50_entire = []
#     r_10_entire = []
#     r_5_entire = []
#     r_1_entire = []
#
#     ndcg_10_entire = []
#     ndcg_entire = []
#
#     model_descriptor = model_descriptor.eval()
#     model_scene = model_scene.eval()
#
#     for t in entire_theme_list:
#         sent = "I look for a room with " + t + " theme."
#         embedding = get_embeddings(sent, device, tokenizer, model_bert)
#         number_of_queries += 1
#         if desc_model_type == 'mean':
#             embedding = embedding.repeat(batch_size, 1)
#         output_description = model_descriptor(embedding)
#         output_scene = model_scene(data_scene.cuda())
#         if desc_model_type == 'mean':
#             output_description = output_description[0]
#         r_50, r_10, r_5, r_1 = create_percent_queries(output_description[0], output_scene, containing_index['theme'][t])
#
#         r_50_entire.append(r_50)
#         r_10_entire.append(r_10)
#         r_5_entire.append(r_5)
#         r_1_entire.append(r_1)
#
#         ndcg, ndcg_10 = analyze_ndcg_queries(output_description[0], output_scene, containing_index['theme'][t])
#         if ndcg_10 is not None:
#             ndcg_10_entire.append(ndcg_10)
#         if ndcg is not None:
#             ndcg_entire.append(ndcg)
#
#     for t in entire_style_list:
#         sent = "I look for a room with " + t + " style."
#         embedding = get_embeddings(sent, device, tokenizer, model_bert)
#         number_of_queries += 1
#         if desc_model_type is not None:
#             embedding = embedding.repeat(batch_size, 1)
#         output_description = model_descriptor(embedding)
#         output_scene = model_scene(data_scene.cuda())
#         if desc_model_type is not None:
#             r_50, r_10, r_5, r_1 = create_percent_queries(output_description[0], output_scene,containing_index['style'][t])
#         else:
#             r_50, r_10, r_5, r_1 = create_percent_queries(output_description, output_scene, containing_index['style'][t])
#
#         r_50_entire.append(r_50)
#         r_10_entire.append(r_10)
#         r_5_entire.append(r_5)
#         r_1_entire.append(r_1)
#
#         ndcg, ndcg_10 = analyze_ndcg_queries(output_description[0], output_scene, containing_index['style'][t])
#         if ndcg_10 is not None:
#             ndcg_10_entire.append(ndcg_10)
#         if ndcg is not None:
#             ndcg_entire.append(ndcg)
#
#     for t in entire_material_list:
#         sent = "I look for a room with " + t + " material."
#         embedding = get_embeddings(sent, device, tokenizer, model_bert)
#         number_of_queries += 1
#         if desc_model_type is not None:
#             embedding = embedding.repeat(batch_size, 1)
#         output_description = model_descriptor(embedding)
#         output_scene = model_scene(data_scene.cuda())
#         if desc_model_type is not None:
#             r_50, r_10, r_5, r_1 = create_percent_queries(output_description[0], output_scene, containing_index['material'][t])
#         else:
#             r_50, r_10, r_5, r_1 = create_percent_queries(output_description, output_scene, containing_index['material'][t])
#
#         r_50_entire.append(r_50)
#         r_10_entire.append(r_10)
#         r_5_entire.append(r_5)
#         r_1_entire.append(r_1)
#
#         ndcg, ndcg_10 = analyze_ndcg_queries(output_description[0], output_scene, containing_index['material'][t])
#         if ndcg_10 is not None:
#             ndcg_10_entire.append(ndcg_10)
#         if ndcg is not None:
#             ndcg_entire.append(ndcg)
#
#     avg_r_50_entire = 100 * sum(r_50_entire) / len(r_50_entire)
#     avg_r_10_entire = 100 * sum(r_10_entire) / len(r_10_entire)
#     avg_r_5_entire = 100 * sum(r_5_entire) / len(r_5_entire)
#     avg_r_1_entire = 100 * sum(r_1_entire) / len(r_1_entire)
#     avg_ndcg_10_entire = 100 * sum(ndcg_10_entire) / len(ndcg_10_entire)
#     avg_ndcg_entire = 100 * sum(ndcg_entire) / len(ndcg_entire)
#     print("Number of Style Queries", number_of_queries)
#
#     print_results(avg_ndcg_10_entire, avg_ndcg_entire, avg_r_10_entire, avg_r_1_entire, avg_r_50_entire, avg_r_5_entire)
#     model_descriptor = model_descriptor.train()
#     model_scene = model_scene.train()
#     return avg_r_1_entire, avg_r_5_entire, avg_r_10_entire, avg_ndcg_10_entire, avg_ndcg_entire


def process_captions(captions):
    new_captions = []
    verbs = []
    nlp = spacy.load("en_core_web_sm")
    for c in captions:
        tmp_sent = "I " + c
        split_sent = c.split()
        res = nlp(tmp_sent)
        # print("* " * 30)
        # print(tmp_sent)
        for token in res:
            # print(token.text, " ", token.pos_)
            if token.pos_ == "VERB":
                verb = token.text
                if verb not in verbs:
                    verbs.append(verb)
                if verb[-1] == 's':
                    continue
                if split_sent.index(verb) > 1:
                    if split_sent[split_sent.index(verb) - 1] != "and":
                        continue
                if verb[-1] == 't' and verb != "heat":
                    ing_form = Word(verb).lemmatize('v') + 't' + 'ing'
                elif verb[-1] == 'e':
                    ing_form = Word(verb[:-1]).lemmatize('v') + 'ing'
                else:
                    ing_form = Word(verb).lemmatize('v') + 'ing'

                split_sent[split_sent.index(verb)] = ing_form
                # tmp_sent = tmp_sent.replace(verb, ing_form)
                # ing_form = conjugate(verb, tense='part')
                # tmp_sent2 = tmp_sent2.replace(verb, ing_form)
        # print(' '.join(split_sent))
        new_captions.append(' '.join(split_sent))

    return new_captions, verbs


# def evaluate_video_queries(model_descriptor, model_scene, data_scene, data_scene_living, data_scene_bedroom, indices, no_of_videos, type_room, desc_model_type, batch_size):
#     device = ("cuda" if torch.cuda.is_available() else "cpu")
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model_bert = BertModel.from_pretrained('bert-base-uncased')
#     model_bert.to(device)
#
#     if type_room not in ['combined']:
#         data_scene = data_scene[indices]
#     else:
#         data_scene = get_scene_data(data_scene_living=data_scene_living,
#                                     data_scene_bedroom=data_scene_bedroom,
#                                     indices=indices)
#
#     # processing video related queries
#     video_captions = open("/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/data_youcook2/captions.pkl", 'rb')
#     video_captions = pickle.load(video_captions)
#     video_captions = video_captions[:no_of_videos]
#     processed_captions, _ = process_captions(video_captions)
#
#     r_50_entire = []
#     r_10_entire = []
#     r_5_entire = []
#     r_1_entire = []
#
#     ndcg_10_entire = []
#     ndcg_entire = []
#
#     model_descriptor = model_descriptor.eval()
#     model_scene = model_scene.eval()
#
#     if type_room not in ['combined']:
#         video_indexes = [x % no_of_videos for x in indices.tolist()]
#     else:
#         video_indexes = []
#         with open('indices/indices_combined_guidance.pkl', 'rb') as f:
#             indices_guidance = pickle.load(f)
#         for idx in indices:
#             video_indexes.append(indices_guidance[idx]['index'] % no_of_videos)
#     set_videos = set(video_indexes)
#     print("Video Queries len_set: ", len(set_videos))
#
#     for t in set_videos:
#         sent = "I look for a room in which the TV showing  " + processed_captions[t] + "."
#         embedding = get_embeddings(sent, device, tokenizer, model_bert)
#         if desc_model_type == 'mean':
#             embedding = embedding.repeat(batch_size, 1)
#         output_description = model_descriptor(embedding)
#         output_scene = model_scene(data_scene.cuda())
#         desired = [i for i, x in enumerate(video_indexes) if x == t]
#         if desc_model_type == 'mean':
#             output_description = output_description[0]
#         r_50, r_10, r_5, r_1 = create_percent_queries(output_description, output_scene, desired)
#
#         r_50_entire.append(r_50)
#         r_10_entire.append(r_10)
#         r_5_entire.append(r_5)
#         r_1_entire.append(r_1)
#
#         ndcg, ndcg_10 = analyze_ndcg_queries(output_description[0], output_scene, desired)
#
#         if ndcg_10 is not None:
#             ndcg_10_entire.append(ndcg_10)
#         if ndcg is not None:
#             ndcg_entire.append(ndcg)
#
#     avg_r_50_entire = 100 * sum(r_50_entire) / len(r_50_entire)
#     avg_r_10_entire = 100 * sum(r_10_entire) / len(r_10_entire)
#     avg_r_5_entire = 100 * sum(r_5_entire) / len(r_5_entire)
#     avg_r_1_entire = 100 * sum(r_1_entire) / len(r_1_entire)
#
#     avg_ndcg_10_entire = -1
#     avg_ndcg_entire = -1
#     if len(ndcg_10_entire) != 0:
#         avg_ndcg_10_entire = sum(ndcg_10_entire) / len(ndcg_10_entire)
#     if len(ndcg_entire) != 0:
#         avg_ndcg_entire = sum(ndcg_entire) / len(ndcg_entire)
#
#     print_results("avg_ndcg_10_entire", "avg_ndcg_entire", avg_r_10_entire, avg_r_1_entire, avg_r_50_entire, avg_r_5_entire)
#
#     model_descriptor = model_descriptor.train()
#     model_scene = model_scene.train()
#     return avg_r_1_entire, avg_r_5_entire, avg_r_10_entire, avg_ndcg_10_entire, avg_ndcg_entire


# def evaluate_distance_queries(model_descriptor, model_scene, data_scene, data_scene_living, data_scene_bedroom, indices, type_room, desc_model_type, batch_size):
#     if type_room not in ['combined']:
#         with open(
#                 '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main_data/sentence_features/entire_positional_info_' + type_room + '.pkl',
#                 'rb') as f:
#             dist_info = pickle.load(f)
#         dist_info = [dist_info[i] for i in indices.numpy()]
#     else:
#         with open(
#                 '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main_data/sentence_features/entire_positional_info_' + 'living' + '.pkl',
#                 'rb') as f:
#             dist_info_living = pickle.load(f)
#         with open(
#                 '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main_data/sentence_features/entire_positional_info_' + 'bedroom' + '.pkl',
#                 'rb') as f:
#             dist_info_bedroom = pickle.load(f)
#
#             dist_info = []
#             with open('indices/indices_combined_guidance.pkl', 'rb') as f:
#                 indices_guidance = pickle.load(f)
#             for idx in indices:
#                 retrieve_orig = indices_guidance[idx]
#                 if retrieve_orig['type'] == 'living':
#                     dist_info.append(dist_info_living[retrieve_orig['index']])
#                 elif retrieve_orig['type'] == 'bedroom':
#                     dist_info.append(dist_info_bedroom[retrieve_orig['index']])
#
#     unique_list = []
#     for i in dist_info:
#         for j in i["info"]:
#             is_found = False
#             for o in unique_list:
#                 if o['obj1'] == j['obj1'] and o['obj2'] == j['obj2'] and o['rel'] == j['rel']:
#                     o["indexes"].append(i['scene_index'])
#                     is_found = True
#                     break
#                 elif o['obj1'] == j['obj2'] and o['obj2'] == j['obj1'] and o['rel'] == j['rel']:
#                     o["indexes"].append(i['scene_index'])
#                     is_found = True
#                     break
#             if not is_found:
#                 new_dict = {"obj1": j['obj1'], "rel": j['rel'], "obj2": j['obj2'], "indexes": [i['scene_index']]}
#                 unique_list.append(new_dict)
#
#     # repetitive_count = 0
#     # for i in range(len(unique_list)):
#     #     for j in range(i+1, len(unique_list)):
#     #         if unique_list[i]['obj1'] == unique_list[j]['obj1'] and unique_list[i]['obj2'] == unique_list[j]['obj2'] and unique_list[i]['rel'] == unique_list[j]['rel']:
#     #             repetitive_count += 1
#     #         elif unique_list[i]['obj1'] == unique_list[j]['obj2'] and unique_list[i]['obj2'] == unique_list[j]['obj1'] and unique_list[i]['rel'] == unique_list[j]['rel']:
#     #             repetitive_count += 1
#     #         pass
#
#     device = ("cuda" if torch.cuda.is_available() else "cpu")
#     if type_room not in ['combined']:
#         data_scene = data_scene[indices]
#     else:
#         data_scene = get_scene_data(data_scene_living=data_scene_living,
#                                     data_scene_bedroom=data_scene_bedroom,
#                                     indices=indices)
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model_bert = BertModel.from_pretrained('bert-base-uncased')
#     model_bert.to(device)
#
#     r_50_entire = []
#     r_10_entire = []
#     r_5_entire = []
#     r_1_entire = []
#
#     ndcg_10_entire = []
#     ndcg_entire = []
#
#     temp_hist_dist = []
#
#     model_descriptor = model_descriptor.eval()
#     model_scene = model_scene.eval()
#
#     for t in unique_list:
#         if len(t["indexes"]) >= 10:
#             temp_hist_dist.append(len(t["indexes"]))
#             sent = "I look for a room in which " + t['obj1'] + " is " + t['rel'] + " " + t['obj2'] + "."
#             embedding = get_embeddings(sent, device, tokenizer, model_bert)
#             if desc_model_type is not None:
#                 embedding = embedding.repeat(batch_size, 1)
#             output_description = model_descriptor(embedding)
#             output_scene = model_scene(data_scene.cuda())
#             if desc_model_type is not None:
#                 r_50, r_10, r_5, r_1 = create_percent_queries(output_description[0], output_scene, t["indexes"])
#             else:
#                 r_50, r_10, r_5, r_1 = create_percent_queries(output_description, output_scene, t["indexes"])
#
#             r_50_entire.append(r_50)
#             r_10_entire.append(r_10)
#             r_5_entire.append(r_5)
#             r_1_entire.append(r_1)
#
#             if desc_model_type is not None:
#                 ndcg, ndcg_10 = analyze_ndcg_queries(output_description[0], output_scene, t["indexes"])
#             else:
#                 ndcg, ndcg_10 = analyze_ndcg_queries(output_description, output_scene, t["indexes"])
#
#             if ndcg_10 is not None:
#                 ndcg_10_entire.append(ndcg_10)
#             if ndcg is not None:
#                 ndcg_entire.append(ndcg)
#
#     print("Distance Queries ")
#
#     print("len length>10: ", len(temp_hist_dist))
#     print("max length>10: ", max(temp_hist_dist))
#     print("min length>10: ", min(temp_hist_dist))
#
#     avg_r_50_entire = 100 * sum(r_50_entire) / len(r_50_entire)
#     avg_r_10_entire = 100 * sum(r_10_entire) / len(r_10_entire)
#     avg_r_5_entire = 100 * sum(r_5_entire) / len(r_5_entire)
#     avg_r_1_entire = 100 * sum(r_1_entire) / len(r_1_entire)
#     avg_ndcg_10_entire = 100 * sum(ndcg_10_entire) / len(ndcg_10_entire)
#     avg_ndcg_entire = 100 * sum(ndcg_entire) / len(ndcg_entire)
#
#     print_results(avg_ndcg_10_entire, avg_ndcg_entire, avg_r_10_entire, avg_r_1_entire, avg_r_50_entire, avg_r_5_entire)
#
#     model_descriptor = model_descriptor.train()
#     model_scene = model_scene.train()
#
#     return avg_r_1_entire, avg_r_5_entire, avg_r_10_entire, avg_ndcg_10_entire, avg_ndcg_entire


def plot_procedure(train_losses, val_losses):
    plt.plot(train_losses, color='red', label='Training loss')
    plt.plot(val_losses, color='blue', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.show()
