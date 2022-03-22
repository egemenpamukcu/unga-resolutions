
import pandas as pd 
import numpy as np
from mpi4py import MPI
import nltk
from nltk import WordNetLemmatizer
from nltk import word_tokenize
import os


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

dirname=os.path.dirname

FILE_PATH = os.path.abspath(__file__)
ROOT_DIR = dirname(dirname(dirname(FILE_PATH)))
INIT_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'initial')
INTER_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'inter')
FINAL_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'final')

# nltk.download("stopwords")
stop_words = set(nltk.corpus.stopwords.words('english'))
lemma = WordNetLemmatizer()

def tokenize(text): 
    return [lemma.lemmatize(lemma.lemmatize(w, 'v'), 'n') for w in word_tokenize(text.lower()) if w not in stop_words and w.isalpha() and len(w) > 1]

if rank == 0:
	corpus = pd.read_json(os.path.join(FINAL_DATA_PATH, 'unga_corpus_clean.json'))
	metadata = pd.read_json(os.path.join(FINAL_DATA_PATH, 'metadata.json'))
	# un_corpus = metadata[['Resolution', 'Vote date', 'Title']].merge(corpus, on='Resolution', how='inner')
	# un_corpus.sort_values('Vote date', inplace=True)
	splits = np.array_split(corpus, size)

else:
	splits = None

corpus = comm.scatter(splits, root=0)
corpus['tokens'] = corpus['Text'].apply(tokenize)
splits = comm.gather(corpus, root=0)

if rank == 0:
	corpus = pd.concat(splits).reset_index(drop=True)
	corpus.drop(['Text', 'url'], inplace=True, axis=1)
	corpus.to_json(os.path.join(FINAL_DATA_PATH, 'unga_tokens.json'))

