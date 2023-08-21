import os
from tqdm import tqdm
import numpy as np
import pickle
import torch


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def calculate_accuracy(feature_set_1, feature_set_2):
    ranks = []
    for idx in range(feature_set_2.shape[0]):
        distances = np.array([euclidean_distance(feature_set_2[idx], x) for x in feature_set_1])
        sorted_indexes = np.argsort(distances)
        # sorted_array = feature_set_1[sorted_indexes]
        ranks.append(np.where(sorted_indexes == idx)[0].tolist()[0])
    ranks = np.array(ranks)
    n_q = feature_set_2.shape[0]
    r1 = 100 * len(np.where(ranks < 1)[0]) / n_q
    r5 = 100 * len(np.where(ranks < 5)[0]) / n_q
    r10 = 100 * len(np.where(ranks < 10)[0]) / n_q
    medr = np.median(ranks) + 1
    meanr = ranks.mean() + 1

    ranks = np.array(ranks)
    print('r1', r1)
    print('r5', r5)
    print('r10', r10)
    print('Rank Median:', np.median(ranks) + 1)
    print('Rank Mean:', ranks.mean() + 1)


def start_process():
    '''
    Notice: For the resnet and bert combination it does not work
    because their final feature sizes are different
    '''
    type_features = 'open_clip'

    root_path_open_clip = '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/open_clip_features'
    root_path_bert_resnet = '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/resnet_features'

    root_path = root_path_open_clip
    if type_features == 'resnet_bert':
        root_path = root_path_bert_resnet

    indices = open(root_path + '/../train/indices/indices.pkl', 'rb')
    indices = pickle.load(indices)
    indices_test = indices['test'].tolist()

    if type_features == 'open_clip':
        imgs_features_path = root_path + '/images'
        desc_features_sent_level_path = root_path + '/descriptions/sentences'
        desc_features_token_level_path = root_path + '/descriptions/tokens'
    else:
        imgs_features_path = '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/mio_codes/houses_imgs_proc_version/imagery_features'
        desc_features_sent_level_path = root_path + '/sentences'
        desc_features_token_level_path = root_path + '/tokens'

    images_features_mean = os.path.isfile(root_path + '/images_means.npy')
    sentences_features_mean = os.path.isfile(root_path + '/sentence_means.npy')
    tokens_features_mean = os.path.isfile(root_path + '/tokens_means.npy')
    if not images_features_mean and not sentences_features_mean and not tokens_features_mean:
        # load images features
        images_files = os.listdir(imgs_features_path)
        sample_img = torch.load(imgs_features_path + os.sep + images_files[0])
        images_features_mean = torch.empty(len(indices_test), sample_img.shape[1])

        sample_desc_sent = torch.load(desc_features_sent_level_path + os.sep + images_files[0])
        sentences_features_mean = torch.empty(len(indices_test), sample_desc_sent.shape[1])

        sample_desc_token = torch.load(desc_features_token_level_path + os.sep + images_files[0])
        tokens_features_mean = torch.empty(len(indices_test), sample_desc_token.shape[1])

        for idx, idx_orig in tqdm(enumerate(indices_test), total=len(indices_test)):
            tmp_img = torch.load(imgs_features_path + os.sep + images_files[idx_orig])
            images_features_mean[idx, :] = torch.mean(tmp_img, dim=0)

            tmp_sent = torch.load(desc_features_sent_level_path + os.sep + images_files[idx_orig])
            sentences_features_mean[idx, :] = torch.mean(tmp_sent, dim=0)

            tmp_token = torch.load(desc_features_token_level_path + os.sep + images_files[idx_orig])
            tokens_features_mean[idx, :] = torch.mean(tmp_token, dim=0)

        np.save(root_path + '/images_means.npy', images_features_mean)
        np.save(root_path + '/sentence_means.npy', sentences_features_mean)
        np.save(root_path + '/tokens_means.npy', tokens_features_mean)

    images_features_mean = np.load(root_path + '/images_means.npy')
    sentences_features_mean = np.load(root_path + '/sentence_means.npy')
    tokens_features_mean = np.load(root_path + '/tokens_means.npy')

    print('retrieving images based on the sentences:')
    calculate_accuracy(images_features_mean, sentences_features_mean)
    print('retrieving sentences based on the images:')
    calculate_accuracy(sentences_features_mean, images_features_mean)
    print('retrieving images based on the tokens:')
    calculate_accuracy(images_features_mean, tokens_features_mean)


if __name__ == '__main__':
    start_process()
