import pandas as pd
import os
from gensim import corpora
import gensim
import pickle
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

dirname=os.path.dirname

FILE_PATH = os.path.abspath(__file__)
ROOT_DIR = dirname(dirname(dirname(FILE_PATH)))
FINAL_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'final')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'output', 'topic-modeling')


if rank == 0: 
    tokens = pd.read_json(os.path.join(FINAL_DATA_PATH, 'unga_tokens.json'))

    id2word = corpora.Dictionary(tokens['tokens'])
    id2word.filter_extremes(no_below=20, no_above=.75)
    corpus = [id2word.doc2bow(doc) for doc in tokens['tokens']]

    outfile = open(os.path.join(OUTPUT_PATH, 'lda-models', 'id2word'), 'wb')
    pickle.dump(id2word, outfile)
    outfile.close()

    outfile = open(os.path.join(OUTPUT_PATH, 'lda-models', 'corpus'),'wb')
    pickle.dump(corpus, outfile)
    outfile.close()
else: 
    id2word = None
    corpus = None

id2word = comm.bcast(id2word, root=0)
corpus = comm.bcast(corpus, root=0)

# Build LDA model

# assuming there are 8 parallel processes, this will create topic models 
# with [6, 8, 10, 12, 14, 16, 18, 20] number of topics in parallel
n_topic = 6 + rank * 2

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=n_topic, 
                                        random_state=1,
                                        update_every=1,
                                        chunksize=100,
                                        passes=10,
                                        alpha='auto',
                                        per_word_topics=True)

outfile = open(os.path.join(OUTPUT_PATH, 'lda-models', f'lda-model-{n_topic}'), 'wb')
pickle.dump(lda_model, outfile)
outfile.close()