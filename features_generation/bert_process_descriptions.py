'''
This script is developed to extract descriptions features using BERT model
'''
import torch
import os
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from nltk.tokenize import WordPunctTokenizer


def get_embeddings(description, model, tokenizer, device):
    with torch.no_grad():
        existing_tokens = tokenize_paragraph_with_punctuations(description)
        inputs = tokenizer(existing_tokens, padding=True, truncation=True, return_tensors='pt')
        inputs = inputs.to(device)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        token_embeddings = torch.mean(embeddings, dim=1)

    tokenized_sentence = description.split('.')
    if tokenized_sentence[-1].replace(" ", "") != '':
        print('tokenized sentence last part: ', tokenized_sentence[-1])
        exit(0)
    obtained_tensor_sent_level = torch.empty(len(tokenized_sentence) - 1, 768)
    cnt = 0
    with torch.no_grad():
        for idx in range(len(tokenized_sentence) - 1):
            inputs = tokenizer(tokenized_sentence[idx], padding=True, truncation=True, return_tensors='pt')
            inputs = inputs.to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
            sentence_embeddings = torch.mean(embeddings, dim=1)
            obtained_tensor_sent_level[cnt, :] = sentence_embeddings.cpu()
            cnt += 1
    return obtained_tensor_sent_level, token_embeddings


def tokenize_paragraph_with_punctuations(paragraph):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(paragraph)
    return tokens


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()
bert_model = bert_model.to(device)

path_descriptions = '../descriptions'
list_txt_files = os.listdir(path_descriptions)

path_sentence_level_features = '../resnet_bert_features/bert_features/sentences/'
path_token_level_features = '../resnet_bert_features/bert_features/tokens'

os.makedirs(path_sentence_level_features, exist_ok=True)
os.makedirs(path_token_level_features, exist_ok=True)

for jf in tqdm(list_txt_files):
    file = open(path_descriptions + os.sep + jf)
    text = file.read()
    sent_level_features, token_level_features = get_embeddings(description=text, model=bert_model, tokenizer=tokenizer, device=device)

    file_name = jf.replace('.txt', '')
    features_sentences = sent_level_features.cpu()
    torch.save(features_sentences, f'../resnet_bert_features/bert_features/sentences/{file_name}.pt')

    features_tokens = token_level_features.cpu()
    torch.save(features_tokens, f'../resnet_bert_features/bert_features/tokens/{file_name}.pt')
