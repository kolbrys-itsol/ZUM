# ZUM NLP Project

Do poprawnego działania potrzebne jest wykonanie komendy:
`python3 -m spacy download en_core_web_sm` 

Dodatkowe potrzebne moduły są wypisane w poszczególnych modułach i zeszytach Jupiter.

# Przebieg projektu:

## **1A**

Dane pozyskiwane za pomocą skryptu zawartego w zeszycie `data_scraping.ipynb`.
Z powodu problemów z siecią na lokalnym komputerze, skorzystano z Google Colab.
Wstępna obróbka danych poprzez skrypt `data_preprocessing.py`

Następnie etykietowanie danych w `data_labelling.py`
Ostatecznie zdecydowałem się na prostszy sposób, ponieważ miałem problemy
przy użyciu gotowych embeddings z `https://wikipedia2vec.github.io/wikipedia2vec/pretrained/#english`
Skrypt przydzielał do jednej grupy 95% wpisów. Skrypt z próbą zaawansowanego tagowania znajduje się w `data_classification.py`
By wadliwy skrypt działał  należy pobrać i rozpakować wersję 
enwiki_20180420 (window=5, iteration=10, negative=15) 100d .txt z:
`https://wikipedia2vec.github.io/wikipedia2vec/pretrained/`

## ETAP 2

Trzy klasyczne modele nauczania maszynowego: `classic_ml.ipynb`

## ETAP 3

Model neuronowy: `neural_ml.ipynb`

## ETAP 4
Zeszyt uruchomiono w środowisku Google Colab (należy wgrać plik `tagged_data.csv`
do sesji). Niestety trening nie działa przez problemy podczas runtime.
Plik: `language_model.ipynb`.
W ostatniej komórce zaimplementowałem prosty klasyfikator z wykorzystaniem gotowego modelu.
