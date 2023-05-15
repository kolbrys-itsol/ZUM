import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# próbowałem dużo bardziej złożony sposób tagowania danych, ale mimo licznych prób nie udało mi się osiągnąć sukcesu
# w jednej grupie znajdowało się 95% tweetów, nie wiem czy to przez mój błąd programistyczny czy przez dane
# stary skrypt znajduje się w data classification.py

# ładujemy dane i usuwamy indeks
data = pd.read_csv("clean_data.csv")
data.drop(columns=['index'], inplace=True)

# wektoryzacja
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(data['Tweets'])

# model KNN

model = KMeans(n_clusters=2,
               max_iter=1000,
               random_state=True,
               n_init=50)

model.fit(tfidf)

data['cluster'] = model.labels_

print(data[data["cluster"] == 1])
print(data[data["cluster"] == 0])

data.to_csv("tagged_data.csv")
