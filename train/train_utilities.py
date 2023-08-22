import os
import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
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


# def get_scene_data(data_scene, indices, type_model_scene):
#     root_path = data_scene
#     with open('/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/houses_data/houses_data.pkl',
#             'rb') as f:
#         pickle_file = pickle.load(f)
#
#     data_scene_list = []
#     list_length = []
#     if indices is not None:
#         if type_model_scene == Constants.model_scene_mean:
#             tmp_scene = torch.load(root_path + os.sep + pickle_file[0]['json_file'] + '.pt')
#             data_scene = torch.empty(len(indices), tmp_scene.shape[1])
#         index = 0
#         for idx in indices:
#             data_scene_ = torch.load(root_path + os.sep + pickle_file[idx]['json_file'] + '.pt')
#             if type_model_scene == 'mean':
#                 data_scene_ = torch.mean(data_scene_, 0)
#                 data_scene[index, :] = data_scene_
#                 index += 1
#             elif type_model_scene == Constants.model_scene_onedimensional:
#                 # getting the padding scenes
#                 data_scene_list.append(data_scene_)
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
#         data_scene = pad_sequence(data_scene_list, batch_first=True)
#         list_length = [len(x) for x in data_scene_list]
#         data_scene = torch.transpose(data_scene, 1, 2)
#     return data_scene, list_length


# def get_scene_data_with_tag(data_scene, data_tag, indices, type_model_scene):
#     root_path_scene = data_scene
#     root_path_tag = data_tag
#     with open('/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/houses_data/houses_data.pkl',
#             'rb') as f:
#         pickle_file = pickle.load(f)
#
#     entire_data_scene_list = []
#     list_length = []
#     if indices is not None:
#         if type_model_scene == Constants.model_scene_mean:
#             tmp_scene = torch.load(root_path_scene + os.sep + pickle_file[0]['json_file'] + '.pt')
#             tmp_tag = torch.load(root_path_tag + os.sep + pickle_file[0]['json_file'] + '.pt')
#             tmp_entire_data_scene = torch.cat((tmp_scene, tmp_tag), 1)
#             entire_data_scene = torch.empty(len(indices), tmp_entire_data_scene.shape[1])
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
#     if type_model_scene == Constants.model_scene_onedimensional:
#         entire_data_scene = pad_sequence(entire_data_scene_list, batch_first=True)
#         list_length = [len(x) for x in entire_data_scene_list]
#         entire_data_scene = torch.transpose(entire_data_scene, 1, 2)
#     return entire_data_scene, list_length


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


# def get_list_tensor_description(data_description_path, indices):
#
#     tensor_lists = get_related_descriptions(data_description_path=data_description_path,
#                                             indices=indices,
#                                             type_model_desc='gru')
#
#     tmp = pad_sequence(tensor_lists, batch_first=True)
#     desc_ = pack_padded_sequence(tmp,
#                                  torch.tensor([len(x) for x in tensor_lists]),
#                                  batch_first=True,
#                                  enforce_sorted=False)
#     return desc_


# def get_list_tensor_description_self_attention(data_description_path, indices):
#
#     tensor_lists = get_related_descriptions(data_description_path=data_description_path,
#                                             indices=indices,
#                                             type_model_desc="self_attention")
#
#     desc_ = pad_sequence(tensor_lists, batch_first=True)
#
#     lengths = torch.tensor([len(x) for x in tensor_lists])
#     max_length = lengths.max().item()
#     range_tensor = torch.arange(max_length)
#     # Create a mask tensor by comparing the range tensor to the length of each unpadded tensor
#     mask = range_tensor.unsqueeze(0) < lengths.unsqueeze(1)
#     mask = ~mask
#
#     return desc_, mask


# class MILNCELoss(torch.nn.Module):
#     def __init__(self):
#         super(MILNCELoss, self).__init__()
#
#     def forward(self, pairwise_similarity):
#         nominator = pairwise_similarity * torch.eye(pairwise_similarity.shape[0]).cuda()
#         nominator = torch.logsumexp(nominator, dim=1)
#         denominator = torch.cat((pairwise_similarity,
#                                  pairwise_similarity.permute(1, 0)), dim=1).view(pairwise_similarity.shape[0], -1)
#         denominator = torch.logsumexp(denominator, dim=1)
#         return torch.mean(denominator - nominator)


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


def get_embeddings(sentence, device, tokenizer, model_bert):
    with torch.no_grad():
        inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
        inputs = inputs.to(device)
        outputs = model_bert(**inputs)
        sentence_embeddings = outputs.last_hidden_state
        sentence_embeddings = torch.mean(sentence_embeddings, dim=1)
        obtained_tensor = sentence_embeddings
    return obtained_tensor


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
        new_captions.append(' '.join(split_sent))

    return new_captions, verbs


def plot_procedure(train_losses, val_losses):
    plt.plot(train_losses, color='red', label='Training loss')
    plt.plot(val_losses, color='blue', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.show()
