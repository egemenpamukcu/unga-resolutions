import os 
import requests
import bs4
import re
import json
from mpi4py import MPI
import math

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

if rank == 0: 
    urls = list(range(16))
    n = math.ceil(len(urls) / size)
    urls = [urls[i * n:(i + 1) * n] for i in range((len(urls) + n - 1) // n )]
    print(urls)
else:
    urls = None

urls = comm.scatter(urls, root=0)

print(rank, urls)

urls = comm.gather(urls, root=0)

if rank == 0:
    urls = [i for j in urls for i in j]

    print('final', urls)


