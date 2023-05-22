import pandas as pd
import multiprocessing
import numpy as np
from re import sub

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from time import time
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec, KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


def to_word_list(text):
    return text.split()


def create_tfidf_dictionary(x, transformed_file, features):
    vector_coo = transformed_file[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo


def replace_tfidf_words(x, transformed_file, features):
    dictionary = create_tfidf_dictionary(x, transformed_file, features)
    return list(map(lambda y: dictionary[f'{y}'], x.tweets.split()))


def replace_sentiment_words(word, sentiment_dict):
    try:
        out = sentiment_dict[word]
    except KeyError:
        out = 0
    return out


data = pd.read_csv("clean_data.csv")
data_prepared = data.dropna().drop_duplicates().reset_index(drop=True).rename(columns={'Tweets': 'tweets'})
print(data.info)

data_prepared.tweets = data.Tweets.apply(lambda x: to_word_list(x))
data_prepared = data_prepared[data_prepared.tweets.str.len() > 1]
data_prepared.reset_index(inplace=True)
#
sent = [row for row in data_prepared.tweets]
phrases = Phrases(sent, min_count=1, progress_per=1000)
bigrams = Phraser(phrases)
sentences = bigrams[sent]
print(sentences[1])

data_prepared_with_bigrams = data_prepared.copy()
data_prepared_with_bigrams['unmodified_tweets'] = data_prepared.tweets
data_prepared_with_bigrams.unmodified_tweets = data_prepared_with_bigrams.unmodified_tweets.str.join(' ')
data_prepared_with_bigrams.tweets = data_prepared_with_bigrams.tweets.apply(lambda x: " ".join(bigrams[x]))
data_prepared_with_bigrams[["tweets"]].to_csv('prepared_data.csv', index=False)

# k-means & word2vec
w2v = Word2Vec(min_count=5,
               window=5,
               vector_size=500,
               sample=1e-5,
               alpha=0.03,
               min_alpha=0.0007,
               negative=20,
               workers=multiprocessing.cpu_count() - 1)
w2v.build_vocab(sentences, progress_per=50000)

w2v.train(sentences,
          total_examples=w2v.corpus_count,
          epochs=300,
          report_delay=1)
w2v.init_sims(replace=True)

w2v.save('word2vec.model')

wv = Word2Vec.load('word2vec.model').wv
model = KMeans(n_clusters=2,
               max_iter=5000,
               random_state=True,
               n_init=50)
model.fit(X=wv.vectors.astype('double'))

print(wv.similar_by_vector(model.cluster_centers_[1], topn=50, restrict_vocab=None))
print(wv.similar_by_vector(model.cluster_centers_[0], topn=50, restrict_vocab=None))

words = pd.DataFrame(wv.index_to_key)
words.columns = ['words']
words['vectors'] = words.words.apply(lambda x: wv[f'{x}'])
words['cluster'] = words.vectors.apply(lambda x: model.predict([np.array(x)]))
words.cluster = words.cluster.apply(lambda x: x[0])
# words['cluster_value'] = [1 if i == 1 else 0 if i == 0 else -1 for i in words.cluster]
words['cluster_value'] = [1 if i == 1 else -1 for i in words.cluster]
words['closeness_score'] = words.apply(lambda x: 1 / (model.transform([x.vectors]).min()), axis=1)
words['sentiment_coeff'] = words.closeness_score * words.cluster_value
words[['words', 'sentiment_coeff']].to_csv('sentiment_dictionary.csv', index=False)

# final_data = pd.read_csv('prepared_data.csv')
# sentiment_map = pd.read_csv('sentiment_dict.csv')
# # sentiment_dict = dict(zip(sentiment_map.words.values, sentiment_map.sentiment_coeff.values))
#
# file_weighting = final_data.copy()
# tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
# tfidf.fit(file_weighting.tweets)
# features = pd.Series(tfidf.get_feature_names_out())
# transformed = tfidf.transform(file_weighting.tweets)
#
# replaced_tfidf_scores = file_weighting.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)
#
# # replaced_closeness_scores = file_weighting.tweets.apply(
# #     lambda x: list(map(lambda y: replace_sentiment_words(y, sentiment_dict), x.split())))
#
# # replacement_df = pd.DataFrame(
# #     data=[replaced_closeness_scores, replaced_tfidf_scores, file_weighting.tweets]).T
# # replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'tweet']
# # replacement_df['sentiment_rate'] = replacement_df.apply(
# #     lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
# # replacement_df['prediction'] = (replacement_df.sentiment_rate > 0).astype('int8')
# #
# # replacement_df[["tweet", "sentiment_rate", "prediction"]].to_csv("tagged_data.csv")
#
# replacement_df = pd.DataFrame(
#     data=[replaced_tfidf_scores, file_weighting.tweets]).T
# replacement_df.columns = ['tfidf_scores', 'tweet']
#
# replacement_df.to_csv("tagged_data.csv")

# final_data = pd.read_csv('prepared_data.csv')
# # words_dict = dict(zip(words.words, words.cluster_value))
# # print(words_dict)
# # final_data['sentiment'] = final_data.apply(lambda x: get_sentiments(x, words_dict))
# # final_data.to_csv('tagged_data.csv')
# words_dict = dict(zip(words.words, words.cluster_value))
# print(words_dict)
# data_df_plot = final_data.copy()
# data_df_plot['sentiment'] = final_data.apply(get_sentiments, args=(words_dict,))
# print(data_df_plot.head(10))
# data_df_plot.to_csv("final.csv")


# *** LABELLING ***
final_data = pd.read_csv('prepared_data.csv')
sentiment_map = pd.read_csv('sentiment_dictionary.csv')
sentiment_dict = dict(zip(sentiment_map.words.values, sentiment_map.sentiment_coeff.values))

file_weighting = final_data.copy()
tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
tfidf.fit(file_weighting.tweets)
features = pd.Series(tfidf.get_feature_names_out())
transformed = tfidf.transform(file_weighting.tweets)

replaced_tfidf_scores = file_weighting.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)

replaced_closeness_scores = file_weighting.tweets.apply(
    lambda x: list(map(lambda y: replace_sentiment_words(y, sentiment_dict), x.split())))

replacement_df = pd.DataFrame(
    data=[replaced_closeness_scores, replaced_tfidf_scores, file_weighting.tweets]).T
replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'tweet']
replacement_df['sentiment_rate'] = replacement_df.apply(
    lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
replacement_df['prediction'] = (replacement_df.sentiment_rate > 0).astype('int8')

print(replacement_df.groupby(['prediction']).count())
replacement_df[["tweet", "sentiment_rate", "prediction"]].to_csv("tagged_data.csv")
