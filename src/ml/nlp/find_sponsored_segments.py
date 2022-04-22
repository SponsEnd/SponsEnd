import numpy as np
import pandas as pd
from gensim import utils
import gensim.parsing.preprocessing as gsp
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.parsing.preprocessing import preprocess_string
from sklearn.base import BaseEstimator
from sklearn import utils as skl_utils
from tqdm import tqdm
import multiprocessing
import numpy as np


class Doc2VecTransformer(BaseEstimator):

    def __init__(self, vector_size=100, learning_rate=0.02, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = multiprocessing.cpu_count() - 1

    def fit(self, df_x, df_y=None):
        tagged_x = [TaggedDocument(str(row['sents']).split(), [index]) for index, row in df_x.iterrows()]
        model = Doc2Vec(documents=tagged_x, vector_size=self.vector_size, workers=self.workers)

        for epoch in range(self.epochs):
            model.train(skl_utils.shuffle([x for x in tqdm(tagged_x)]), total_examples=len(tagged_x), epochs=1)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha

        self._model = model
        return self

    def transform(self, df_x):
        return np.asmatrix(np.array([self._model.infer_vector(str(row['sents']).split())
                                     for index, row in df_x.iterrows()]))


def check_sponsor(data):
    df = pd.read_csv('src/ml/nlp/data.csv')
    df = df.drop(columns = ['Unnamed: 0'])


    filters = [gsp.strip_tags, gsp.strip_punctuation, gsp.strip_multiple_whitespaces, gsp.strip_numeric, gsp.remove_stopwords, gsp.strip_short, gsp.stem_text ]

    def clean_text(s):
        s = s.lower()
        s = utils.to_unicode(s)
        for f in filters:
            s = f(s)
        return s

    for i in range(df.shape[0]):
        df['Sentence'][i] = clean_text(str(df['Sentence'][i]))


    aggregate_counter = Counter()
    for row_index,row in df.iterrows():
        c = Counter(row['Sentence'].split())
        aggregate_counter += c

    common_words = [word[0] for word in aggregate_counter.most_common(50)]
    common_words_counts = [word[1] for word in aggregate_counter.most_common(50)]

    dat = pd.DataFrame()
    dat['sents'] = list(df.Sentence)

    doc2vec_tr = Doc2VecTransformer(vector_size=200)
    doc2vec_tr.fit(dat)
    doc2vec_vectors = doc2vec_tr.transform(dat)

    from sklearn.neural_network import MLPRegressor

    auto_encoder = MLPRegressor(hidden_layer_sizes=(
                                                    600,
                                                    150, 
                                                    600,
                                                ))
    auto_encoder.fit(doc2vec_vectors, doc2vec_vectors)
    predicted_vectors = auto_encoder.predict(doc2vec_vectors)

    str(round(auto_encoder.score(predicted_vectors, doc2vec_vectors)*100,3))+'%'

    data = [clean_text(i) for i in data]
    #data = [data[1]]
    dd = pd.DataFrame()
    dd['sents'] = data
    data = dd
    # data

    dv = doc2vec_tr.transform(data)
    pv = auto_encoder.predict(dv)

    sponsored_segments = []
    cos_sim = 0

    from scipy.spatial.distance import cosine

    def key_consine_similarity(tupple):
        return tupple[1]

    def get_computed_similarities(vectors, predicted_vectors, reverse=False):
        data_size = len(data)
        cosine_similarities = []
        for i in range(data_size):
            cosine_sim_val = (cosine(vectors[i], predicted_vectors[i]))
            cosine_similarities.append((i, cosine_sim_val))

        return sorted(cosine_similarities, key=key_consine_similarity, reverse=reverse)

    def display_top_n(sorted_cosine_similarities, n=len(data)):
        for i in range(n):
            index, cosine_sim_val = sorted_cosine_similarities[i]
            cosine_sim_val = round(cosine_sim_val,4)
            print('Text Title: ', data.iloc[index][0])
            tag = ''
            if cosine_sim_val > 0.70:
                tag='Sponsored'
                sponsored_segments.append({"text": data.iloc[index][0], "tag": tag}) 
            else:
                tag='Content'
            print('Tag :', cosine_sim_val, tag)
            cosine_sim_val
            print('---------------------------------')
            return cosine_sim_val

    
    sorted_cosine_similarities = get_computed_similarities(vectors=dv, predicted_vectors=pv)
    cos_sim = display_top_n(sorted_cosine_similarities=sorted_cosine_similarities)

    # print(sponsored_segments)
    return cos_sim

def find_sponsored_segments(captions):
    # sentences = captions
    sponsored_segments = []
    limit = 2
    index = 0
    for sentence in captions:
        if index < limit:
            print(sentence["text"])
            cosine_similarity = check_sponsor([sentence["text"]])
            print("Output: ", cosine_similarity)
            if cosine_similarity >= 0.8:
                print(sentence["text"])
                sponsored_segments.append(
                    {
                        "text": sentence["text"], 
                        "start": sentence["start"],
                        "duration": sentence["duration"],
                        "end": float(sentence["start"]) + float(sentence["duration"])
                    }
                )
            print(sponsored_segments)
            index += 1
    return str(sponsored_segments)

    # sentences = ['todays video is sponsored by skillshare. Back to the video.','So my friend told me about her problems and I gave her some advice storytime','this video is sponsored by best fiends. it is a mobile multiplayer game that keeps you engaged.','i am a mukbang youtuber',"thank you raycon for sponsoring today's video",'we will be discussing this exam paper from Indias joint entrance exam','Use my coupon code EATMISS to get flat 30% off your first three purchases']

    # sponsored_segments = check_sponsor(sentences)

# find_sponsored_segments("a. b")
