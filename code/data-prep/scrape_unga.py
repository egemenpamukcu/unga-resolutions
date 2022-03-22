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

dirname=os.path.dirname

FILE_PATH = os.path.abspath(__file__)
ROOT_DIR = dirname(dirname(dirname(FILE_PATH)))
INIT_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'initial')

def get_resolution_urls(start_year=1946, end_year=2022):
    '''
    Crawls through the UN Digital Library and fetches the URLs for each resolution's
    profile between the given start and end years.
    Input:
        start_year (int): the year program will start scraping from
        end_year (int): the year program will scrape the last
    Output:
        List of strings: URLs of each resolution's webpage
    '''
    all_urls = []
    for year in range(start_year, end_year + 1):
        for i in [1, 201]:
            url = 'https://digitallibrary.un.org/search?ln=en&c=Voting+Data&rg=200\
            &jrec={}&fct__3={}&fct__2=General+Assembly&fct__2=General+Assembly&cc=Voting+Data'.format(i, year)
            req = requests.get(url)
            soup = bs4.BeautifulSoup(req.text, 'html.parser')
            as_ = soup.find_all('a', class_='moreinfo', text='Detailed record')
            all_urls += [a['href'] for a in as_]
    return list(set(all_urls))


def get_metadata(urls):
    '''
    Given the URLs of the UNGA resolutions, fetches the metadata associated with
    each resolution.
    Input:
        urls (list): list of UNGA resolution URLs
            (output of the get_resolultion_urls() function)
    Output:
        List of dicts: contains metadata from all UNGA resolutions provided
            as input

    '''
    urls = list(map(lambda x: 'https://digitallibrary.un.org' + x, urls))
    metadata = []
    for url in urls:
        req = requests.get(url)
        soup = bs4.BeautifulSoup(req.text, 'html.parser')
        dic = {'url': url}
        divs = soup.find_all('div', class_='metadata-row')
        for div in divs:
            k = div.find_all('span')[0].text.strip()
            v = div.find_all('span')[1].text.strip()
            dic[k] = v
            as_ = div.find_all('a')
            for a in as_:
                dic[k + '_url'] = a['href']
        if 'Vote' in dic:
            decisions = ['Yes', 'No', 'Abstentions', 'Non-voting', 'Total']
            votes = re.findall(r':(\s+\S+)', dic['Vote summary'])
            for i, vote in enumerate(votes):
                votes[i] = vote.strip()
                if votes[i] == '|':
                    votes[i] = 0
                votes[i] = int(votes[i])
            dic['Votes'] = dict(zip(decisions, votes))
            dic['Votes_url'] = url.replace('?ln=en', '/export/xm')
        metadata.append(dic)
    return metadata


def get_voting_data(metadata):
    '''
    Given the metadata of UNGA resolutions, fetches the voting records for each resolution.
    Input:
        metadata (list of dicts): contains the metadata of all UNGA resolutions
            (output of the get_metadata() function)
    Output:
        List of dicts: matching each resolution ID with a dictionary of voting records.
    '''
    voting_data = {}
    for res in metadata:
        try:
            req = requests.get(res['Votes_url'])
        except KeyError:
            continue
        voting_data[res['Resolution']] = []
        soup = bs4.BeautifulSoup(req.text, 'html.parser')
        datafields = soup.find_all('datafield', tag='967')
        for field in datafields:
            votes = {}
            codes = field.find_all('subfield', code='c')
            for code in codes:
                votes['Code'] = code.text
            countries = field.find_all('subfield', code='e')
            for country in countries:
                votes['Country'] = country.text
            vs = field.find_all('subfield', code='d')
            for v in vs:
                votes['Vote'] = v.text
            voting_data[res["Resolution"]].append(votes)

    return voting_data


def get_pdf_urls(metadata):
    pdf_urls = {}
    for res in metadata:
        try:
            req = requests.get(res['Resolution_url'].replace('?ln=en', '/export/xm'))
        except KeyError:

            continue
        soup = bs4.BeautifulSoup(req.text, 'html.parser')
        subfields = soup.find_all('subfield', code='u')
        for sf in subfields:
            if sf.text.endswith('-EN.pdf'):
                pdf_urls[res['Resolution']] = sf.text
                break
    return pdf_urls


if __name__ == "__main__":
    
    # fetch and write URLs to a txt file

    if rank == 0: 
        urls = get_resolution_urls(start_year=1946, end_year=2022)
        with open(os.path.join(INIT_DATA_PATH, 'urls.txt'), 'w') as outfile:
            json.dump(urls, outfile)
        
        n = math.ceil(len(urls) / size)
        urls = [urls[i * n:(i + 1) * n] for i in range((len(urls) + n - 1) // n )]
    else:
        urls = None
    
    urls = comm.scatter(urls, root=0)

    metadata = get_metadata(urls)
    voting_data = get_voting_data(metadata)
    pdf_urls = get_pdf_urls(metadata)

    metadata = comm.gather(metadata, root=0)
    voting_data = comm.gather(voting_data, root=0)
    pdf_urls = comm.gather(pdf_urls, root=0)

    if rank == 0: 
        metadata = [i for j in metadata for i in j]

        voting_dict = {}
        for vote in voting_data:
            voting_dict.update(vote)

        pdf_dict = {}
        for url in pdf_urls: 
            pdf_dict.update(url)

        with open(os.path.join(INIT_DATA_PATH, 'metadata.txt'), 'w') as outfile:
            json.dump(metadata, outfile)
        
        with open(os.path.join(INIT_DATA_PATH, 'voting_data.txt'), 'w') as outfile:
            json.dump(voting_dict, outfile)

        with open(os.path.join(INIT_DATA_PATH, 'pdf_urls.txt'), 'w') as outfile:
            json.dump(pdf_dict, outfile)

        print("### DONE ###")