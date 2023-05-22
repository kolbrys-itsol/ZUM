# ZUM NLP Project

Dodatkowe potrzebne moduły są wypisane w poszczególnych modułach i zeszytach Jupiter.
Żeby rozpocząć pracę całkowicie od nowa, należy usunąć wszystkie pliki `.csv` poza `tweets_dataset.csv` oraz zapisane modele.
i uruchamiać moduły w kolejności przedstawionej poniżej:

# Przebieg projektu:

## **1A**

Dane pozyskiwane za pomocą skryptu zawartego w zeszycie `data_scraping.ipynb`.
Z powodu problemów z siecią na lokalnym komputerze, skorzystano z Google Colab.
Wstępna obróbka danych poprzez skrypt `data_preprocessing.py`.
Żeby powyższy moduł działał potrzebne są biblioteki `pandas`, `spacy`, `numpy`, `tqdm`
oraz po zainstalowaniu spacy wykonanie komendy: `python -m spacy download en_core_web_md`
Po poprawnym wykonaniu się skryptu otrzymamy plik `clean_data.csv` potrzebny do następnego kroku.

Następnie etykietowanie danych w `data_classification.py`
Poza bibliotekami zainstalowanymi w poprzednim kroku, potrzebne będzie `sklearn` oraz `gensim`.
W wyniku działania modułu pojawią się pliki `prepared_data.csv`, potrzebny do pracy programu oraz plik wynikowy
`tagged_data.csv` z otagowanymi tweetami.
## ETAP 2
Poza bibliotekami z poprzednich kroków, potrzebny będzie `mathplotlib`.
Trzy klasyczne modele nauczania maszynowego: `classic_ml.ipynb`
W wyniku poprawnego działania zeszytu powinny się pojawić krzywe i macierze dla trzech klasycznych modeli
nauczania maszynowego.
## ETAP 3
Tutaj dodatkowo należy zainstalować `keras` oraz `tensorflow`.
Model neuronowy: `neural_ml.ipynb`.


## ETAP 4
Zeszyt uruchomiono w środowisku Google Colab (należy wgrać plik `tagged_data.csv`
do sesji).
Model językowy: `language_model.ipynb`.
Potrzebny jest Colab z włączonym GPU (inaczej trening będzie trwał dośc długo).


W razie jakichkolwiek wątpliwości proszę o kontakt na Teams.
