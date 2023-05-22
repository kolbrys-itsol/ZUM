import pandas as pd
import re
import spacy
import numpy as np

from tqdm import tqdm

# progress bary
tqdm.pandas()

# ładujemy model nlp
nlp = spacy.load("en_core_web_md", disable=['ner', 'parser'])
nlp.add_pipe('sentencizer')


# helper function for removing stop words
def remove_stopwords(text):
    return " ".join(filter(lambda x: x not in nlp.Defaults.stop_words, text.split()))


# helper function for lematization
def lemmatize(text):
    return " ".join([x.lemma_ for x in nlp(text)])


def drop_shorter_than_three_words(text):
    text_len = len(text)
    if text_len > 3:
        return text
    else:
        return ''


data = pd.read_csv('tweets_dataset.csv', on_bad_lines='skip')
print("INPUT:")
print(data.info())


# tweets = data["Tweets"]
# usuwamy wielkie znaki
# tweets = tweets.apply(lambda x: x.lower())
# # usuwamy li nki
# re_url = re.compile(r'((www.[^\s]+)|(https?://[^\s]+))')
# tweets = tweets.apply(lambda x: re_url.sub("", x))
# # usuwamy user handles
# re_handle = re.compile(r'@[^\s]+')
# tweets = tweets.apply(lambda x: re_handle.sub("", x))
# # usuwamy znaki specjalne (jak np. @ powszechne przy ksywkach na Twitterze)
# re_characters = re.compile(r"[^a-zA-Z\s']")
# tweets = tweets.apply(lambda x: re_characters.sub("", x))
# # usuwamy stop words
# tweets = tweets.apply(remove_stopwords)
# # usuwamy tweety krótsze niż 3 słowa (w ten sposób odpada dużo tweetów gdzie tylko jest tagowany inny użytkownik)
# tweets = tweets.apply(lambda x: drop_shorter_than_three_words(x))
# # usuwamy nulle
# tweets.replace('', np.nan, inplace=True)
# tweets.dropna(inplace=True)
#
# # lemmatyzacja
# tweets = tweets.progress_apply(lemmatize)
#
# tweets = tweets.apply(lambda x: x.lower())
# # usuwamy duplikaty
# tweets.drop_duplicates(keep="first", inplace=True)
# tweets = tweets.reset_index()

# usuwamy wielkie znaki
data["Tweets"] = data["Tweets"].apply(lambda x: x.lower())

#usuwamy linki
data["Tweets"] = data["Tweets"].apply(lambda x: re.sub(r"((www.[^\s]+)|(https?://[^\s]+))", " ", x))

#usuwamy user handles
data["Tweets"] = data["Tweets"].apply(lambda x: re.sub(r"@[^\s]+", " ", x))

# czyszczenie
data["Tweets"] = data["Tweets"].apply(lambda x: re.sub(r"[^a-zA-Z\s']", " ", x))
# data["Tweets"] = data["Tweets"].apply(lambda x: re.sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", x))
data["Tweets"] = data["Tweets"].apply(lambda x: re.sub(r"\+", " plus ", x))
data["Tweets"] = data["Tweets"].apply(lambda x: re.sub(r",", " ", x))
data["Tweets"] = data["Tweets"].apply(lambda x: re.sub(r"\.", " ", x))
data["Tweets"] = data["Tweets"].apply(lambda x: re.sub(r"!", " ! ", x))
data["Tweets"] = data["Tweets"].apply(lambda x: re.sub(r"\?", " ? ", x))
data["Tweets"] = data["Tweets"].apply(lambda x: re.sub(r"'", " ", x))
data["Tweets"] = data["Tweets"].apply(lambda x: re.sub(r":", " : ", x))
data["Tweets"] = data["Tweets"].apply(lambda x: re.sub(r"\s{3,}", " ", x))
data["Tweets"] = data["Tweets"].apply(lambda x: re.sub(r"\n", " ", x))

# usuwamy stop words
data["Tweets"] = data["Tweets"].apply(remove_stopwords)
# lemmatyzacja
data["Tweets"] = data["Tweets"].progress_apply(lemmatize)
data["Tweets"] = data["Tweets"].apply(lambda x: x.lower())
# usuwamy duplikaty
data.replace('', np.nan, inplace=True)
# usuwamy krótkie tweety
data = data.apply(lambda x: drop_shorter_than_three_words(x))
# wyrzucamy nulle
data.dropna(inplace=True)
data.drop_duplicates(subset=["Tweets"], inplace=True)

print("OUTPUT:")
print(data.info())
data.to_csv("clean_data.csv")

