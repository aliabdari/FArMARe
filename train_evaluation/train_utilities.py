import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import Constants


def get_entire_data(clip_ones, token_level):
    roots = {}
    if clip_ones:
        roots[Constants.root_scene_path] = '../open_clip_features/images'
        if not token_level:
            roots[Constants.root_description_path] = '../open_clip_features/descriptions/sentences'
        else:
            roots[Constants.root_description_path] = '../open_clip_features/descriptions/tokens'
    else:
        roots[Constants.root_scene_path] = '../resnet_bert_features/images'
        if not token_level:
            roots[Constants.root_description_path] = '../resnet_bert_features/bert_features/sentences'
        else:
            roots[Constants.root_description_path] = '../resnet_bert_features/bert_features/tokens'

    return roots


def retrieve_indices():
    if os.path.isfile('indices/indices.pkl'):
        indices_pickle = open('indices/indices.pkl', "rb")
        indices_pickle = pickle.load(indices_pickle)
        train_indices = indices_pickle["train"]
        val_indices = indices_pickle["val"]
        test_indices = indices_pickle["test"]
    else:
        with open('../houses_data/houses_data.pkl', 'rb') as f:
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


def write_models_evaluation_to_file(data, file_name):
    path = './results'
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + os.sep + file_name
    tags = ["ds1", "ds5", "ds10", "sd1", "sd5", "sd10", "median_rank_ds", "median_rank_sd"]
    with open(path, 'a') as f:
        for i, d in enumerate(data):
            f.write(tags[i] + " : " + str(d))
            f.write('\n')
        f.write('-' * 30)
        f.write('\n')


def create_rank(result, entire_descriptor, desired_output_index):
    similarity = torch.nn.functional.cosine_similarity(entire_descriptor, result, dim=1)
    similarity = similarity.squeeze()
    sorted_indices = torch.argsort(similarity, descending=True)
    position = torch.where(sorted_indices == desired_output_index)
    return position[0].item(), sorted_indices


def contrastive_loss(pairwise_distances, margin=0.25):
    batch_size = pairwise_distances.shape[0]
    diag = pairwise_distances.diag().view(batch_size, 1)
    pos_masks = torch.eye(batch_size).bool().to(pairwise_distances.device)
    d1 = diag.expand_as(pairwise_distances)
    cost_s = (margin + pairwise_distances - d1).clamp(min=0)
    cost_s = cost_s.masked_fill(pos_masks, 0)
    cost_s = cost_s / (batch_size * (batch_size - 1))
    cost_s = cost_s.sum()

    d2 = diag.t().expand_as(pairwise_distances)
    cost_d = (margin + pairwise_distances - d2).clamp(min=0)
    cost_d = cost_d.masked_fill(pos_masks, 0)
    cost_d = cost_d / (batch_size * (batch_size - 1))
    cost_d = cost_d.sum()

    return (cost_s + cost_d) / 2


def evaluate(output_description, output_scene, section):
    avg_rank_scene = 0
    ranks_scene = []
    avg_rank_description = 0
    ranks_description = []

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

    return ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, ds_medr, sd_medr


def cosine_sim(im, s):
    '''
    cosine similarity between all the image and sentence pairs
    '''
    inner_prod = im.mm(s.t())
    im_norm = torch.sqrt((im ** 2).sum(1).view(-1, 1) + 1e-18)
    s_norm = torch.sqrt((s ** 2).sum(1).view(1, -1) + 1e-18)
    sim = inner_prod / (im_norm * s_norm)
    return sim


def plot_procedure(train_losses, val_losses):
    plt.plot(train_losses, color='red', label='Training loss')
    plt.plot(val_losses, color='blue', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.show()
