import sys
import os

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from typing import List
from time import time
import json

nltk.download('stopwords')
print("Loading word2vec model...")
st = time()
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('/home/datasets/log_data/crawl-300d-2M.vec', binary=False)
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
print("Loaded word2vec model in {:.2f} seconds".format(time() - st))
data_dir = "/home/datasets/log_data/"
template_dir = {}
template_dir['HDFS'] = os.path.join(data_dir, 'HDFS/output/')
template_dir['BGL'] = os.path.join(data_dir, 'BGL/output/')
template_dir['Thunderbird'] = os.path.join(data_dir, 'Thunderbird/output/')
template_path = {}
template_path['HDFS'] = os.path.join(data_dir, 'HDFS/output/HDFS.log_templates.csv')
template_path['BGL'] = os.path.join(data_dir, 'BGL/output/BGL.log_templates.csv')
template_path['Thunderbird'] = os.path.join(data_dir, 'Thunderbird/output/Thunderbird_20M.log_templates.csv')


# remove stop word and  punctuation, split by camel case
def clean_template(template: str, remove_stop_words: bool = True):
    template = " ".join([word.lower() if word.isupper() else word for word in template.strip().split()])
    template = re.sub('[A-Z]', lambda x: " " + x.group(0), template)  # camel case
    word_tokens = tokenizer.tokenize(template)  # tokenize
    word_tokens = [w for w in word_tokens if not w.isdigit()]  # remove digital
    if remove_stop_words:  # remove stop words, we can close this function
        filtered_sentence = [w.lower() for w in word_tokens if w not in stop_words]
    else:
        filtered_sentence = [w.lower() for w in word_tokens]

    template_clean = " ".join(filtered_sentence)
    return template_clean  # return string


# get word vec of words in log key, using weight
def log_key2vec(log_template: str, weight: List[float] = None):
    """
    Get word vec of words in log key, using weight
    Parameters
    ----------
    log_template
    weight

    Returns
    -------
    log_template_vec: list of word vec
    """
    words = log_template.strip().split()
    log_template_vec = []

    if not weight:  # if not weight, uniform weight
        weight = [1] * len(words)

    for index, word in enumerate(words):
        try:  # catch the exception when word not in pre-trained word vector dictionary
            log_template_vec.append(word2vec_model[word] * weight[index])
        except Exception as _:
            pass
    if len(log_template_vec) == 0:
        log_template_vec = np.zeros(300)
    return log_template_vec


def generate_embeddings_fasttext(templates: List[str], strategy: str = 'average') -> dict:
    """
    Generate embeddings for templates using fasttext
    Parameters
    ----------
    templates: list of templates
    strategy: average or tfidf

    Returns
    -------
    embeddings: dict of embeddings
    """
    clean_templates = [clean_template(template) for template in templates]
    templates = zip(clean_templates, templates)
    embeddings = {}
    if strategy == 'average':
        for template, k in templates:
            embeddings[k] = np.mean(log_key2vec(template), axis=0).tolist()
    elif strategy == 'tfidf':
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        X = vectorizer.fit_transform(clean_templates)
        tfidf = transformer.fit_transform(X)
        tfidf = tfidf.toarray()
        words = vectorizer.get_feature_names_out().tolist()
        single_weights = []
        for i, (template, k) in enumerate(templates):
            for word in template.strip().split():
                if word in words:
                    single_weights.append(tfidf[i][words.index(word)])
                else:
                    single_weights.append(0)
            embeddings[k] = np.mean(log_key2vec(template, single_weights), axis=0).tolist()
    else:
        raise ValueError('Invalid strategy')

    return embeddings


def load_embeddings_fasttext(embedding_path: str) -> dict:
    """
    Load embeddings for templates using fasttext
    Parameters
    ----------
    embedding_path: path to embeddings (json)

    Returns
    -------
    embeddings: dict of embeddings
    """
    with open(embedding_path, 'r') as f:
        embeddings = json.load(f)
    return embeddings


if __name__ == '__main__':
    dataset = sys.argv[1]
    strategy = sys.argv[2]
    print(f'Generating embeddings for {dataset} using {strategy}...')
    # template_df = pd.read_csv(f'./{dataset}/{dataset}.log_templates.csv')
    template_df = pd.read_csv(template_path[dataset])
    templates = template_df['EventTemplate'].tolist()
    embeddings = generate_embeddings_fasttext(templates, strategy=strategy)
    # with open(f'./{dataset}/{dataset}.log_embeddings_{strategy}.json', 'w') as f:
    #     json.dump(embeddings, f)
    with open(os.path.join(template_dir[dataset], f'{dataset}.log_embeddings_{strategy}.json'), 'w') as f:
        json.dump(embeddings, f)