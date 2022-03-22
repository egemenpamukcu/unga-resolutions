import requests
import io
import pandas as pd
import json
from mpi4py import MPI
import math
import pdftotext
import os

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

dirname=os.path.dirname

FILE_PATH = os.path.abspath(__file__)
ROOT_DIR = dirname(dirname(dirname(FILE_PATH)))
INIT_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'initial')

def create_corpus(pdf_urls):
    '''
    Iterates through URLs directing to UNGA Resolution PDFs, converts them to string,
    and stores them in a DataFrame. The resulting DataFrame will be incomplete because
    some of the PDFs have images of scanned documents instead of text. To get the text
    of these resolutions through Tesseract Optical Character Recognition engine, run
    complete_corpus_ocr.py using the output of this file as an input.
    '''

    # pdf_urls = [[k, v] for k, v in all_pdfs.items()]
    for i, pdf_url in enumerate(pdf_urls):
        req = requests.get(pdf_url[1])
        with io.BytesIO(req.content) as f:
            try:
                pdf = pdftotext.PDF(f)
            except pdftotext.Error:
                try: # try again
                    pdf = pdftotext.PDF(f)
                except pdftotext.Error:
                    print(pdf_url[0], 'ERRORED OUT')
                    continue
        text = ''
        for page in pdf:
            text += page
        pdf_url.append(text.strip())
    return pd.DataFrame(pdf_urls, columns=['Resolution', 'url', 'Text'])


if __name__ == "__main__": 
    if rank == 0: 
        with open(os.path.join(INIT_DATA_PATH, "pdf_urls.txt")) as f:
            pdf_urls = json.load(f)

        pdf_urls = [[k, v] for k, v in pdf_urls.items()]

        n = math.ceil(len(pdf_urls) / size)
        pdf_urls = [pdf_urls[i * n:(i + 1) * n] for i in range((len(pdf_urls) + n - 1) // n )]
    else:
        pdf_urls = None
    
    pdf_urls = comm.scatter(pdf_urls, root=0)
    corpus = create_corpus(pdf_urls)
    corpus = comm.gather(corpus, root=0)

    if rank == 0:
        corpus = pd.concat(corpus).reset_index(drop=True)
        corpus.to_json(os.path.join(INIT_DATA_PATH, 'unga_corpus_raw.json'))