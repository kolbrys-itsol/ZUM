import pandas as pd
import re
# en_core_web_sm pobrać z https://spacy.io/models/en
import en_core_web_sm
import numpy as np

from tqdm import tqdm

# progress bary
tqdm.pandas()

# ładujemy model nlp
nlp = en_core_web_sm.load(disable=['ner', 'parser'])
nlp.add_pipe('sentencizer')


# helper function for removing stop words
def remove_stopwords(text):
    return " ".join(filter(lambda x: x not in nlp.Defaults.stop_words, text.split()))


# helper function for lematization
def lemmatize(text):
    return " ".join([x.lemma_ for x in nlp(text)])


def drop_shorter_than_three_words(text):
    text_len = len(text.split())
    if text_len > 3:
        return text
    else:
        return ''


data = pd.read_csv("tweets_dataset.csv")
print(data.info())


tweets = data["Tweets"]
# usuwamy wielkie znaki
tweets = tweets.apply(lambda x: x.lower())
# usuwamy li nki
re_url = re.compile(r'((www.[^\s]+)|(https?://[^\s]+))')
tweets = tweets.apply(lambda x: re_url.sub("", x))
# usuwamy user handles
re_handle = re.compile(r'@[^\s]+')
tweets = tweets.apply(lambda x: re_handle.sub("", x))
# usuwamy znaki specjalne (jak np. @ powszechne przy ksywkach na Twitterze)
re_characters = re.compile(r"[^a-zA-Z\s']")
tweets = tweets.apply(lambda x: re_characters.sub("", x))
# usuwamy stop words
tweets = tweets.apply(remove_stopwords)
# usuwamy tweety krótsze niż 3 słowa (w ten sposób odpada dużo tweetów gdzie tylko jest tagowany inny użytkownik)
tweets = tweets.apply(lambda x: drop_shorter_than_three_words(x))
# usuwamy nulle
tweets.replace('', np.nan, inplace=True)
tweets.dropna(inplace=True)
# usuwamy duplikaty
tweets.drop_duplicates(keep="first", inplace=True)

# lemmatyzacja - wyłączone, zostanie przeprowadzona po etykietowaniu
# tweets = tweets.progress_apply(lemmatize)
tweets = tweets.reset_index()
print(tweets.info())
tweets.to_csv("clean_data.csv", index=False)

