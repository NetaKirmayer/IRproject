import math

from flask import Flask, request, jsonify
from combine_helper import *
from bm25_bodyindex import *

import gensim
from gensim.models import Word2Vec
import gensim.downloader
import gzip
import pandas as pd

read_title_id_voc = None
read_body_index = None
read_doc_TFIDFnorma = None
read_DL = None
read_title_index = None
read_anchor_index = None
glove_vectors = None
pagerank = None
pageview = None
max_pagerank = None


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        # General
        global read_title_id_voc
        read_title_id_voc = read_object('./general_files', 'title_and_id')
        # global glove_vectors
        # glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
        global pagerank
        pagerank = {}
        with gzip.open('./pr_part-00000-be37bd2a-7509-483d-95a3-55e4c77d3524-c000.csv.gz') as f:
            pr = pd.read_csv(f, header=None)
        pagerank = pr.set_index(0).to_dict()[1]
        global max_pagerank
        max_pagerank = max(pagerank.values())
        global pageview
        pageview = read_object('.', 'pageviews-202108-user')

        # Body
        global read_body_index
        read_body_index = InvertedIndex.read_index('./postings_body', 'body_index')
        global read_doc_TFIDFnorma
        read_doc_TFIDFnorma = read_object('./general_files', 'body_TFIDFnorma2')
        global read_DL
        read_DL = read_object('./general_files', 'DL_bodyIndex')

        # Title
        global read_title_index
        read_title_index = InvertedIndex.read_index('./postings_title', 'title_index')

        # Anchor
        global read_anchor_index
        read_anchor_index = InvertedIndex.read_index('./postings_anchor', 'anchor_index')
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    terms_postings = {}
    bm25_object = BM25_from_index(read_body_index, read_DL)
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    query_terms = []

    for x in tokens:
        if x in read_body_index.df:
            query_terms.append(x)

    w_title = 0
    w_body = 0

    if len(query_terms) < 2:
        w_title = 0.85
        w_body = 0.15
    else:
        w_title = 0.25
        w_body = 0.75

    query_terms_final = []
    for x in query_terms:
        if x in read_body_index.df:
            query_terms_final.append(x)
            terms_postings[x] = read_posting_list(read_body_index, './postings_body', x)

    return_body = cosine_simlarity(query_terms_final, terms_postings, read_DL, read_body_index, read_doc_TFIDFnorma)
    return_title, sumOfTtile = title_score_list(query_terms_final, read_title_index)
    merge_res = merge(return_body, return_title, w_body, w_title)
    pagerank_res = addPageRank(merge_res, pagerank)
    bm25 = BM25_from_index(read_body_index, read_DL)
    bm25_results = bm25.search_from_list(query_terms_final, terms_postings, list(map(lambda x: x[0], pagerank_res)))
    final_res = sorted(bm25_results.items(), key=lambda x: x[1], reverse=True)[:40]
    res = list(map(lambda x: (x[0], read_title_id_voc.get(x[0])), final_res))

    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    new_tokens = []
    for token in tokens:
        if token in read_body_index.df.keys():
            new_tokens.append(token)
    N_corpus = len(read_DL)
    cosine_score = defaultdict(list)
    freq_query = Counter(new_tokens)

    for term in freq_query:

        post_for_term = read_posting_list(read_body_index, './postings_body', term)
        normalization_term = freq_query[term] / len(new_tokens)
        idf = math.log(N_corpus / len(post_for_term), 2)
        quer_nurma = np.sqrt(
            sum([(freq_query[x] / len(new_tokens)) * (freq_query[x] / len(new_tokens) * math.pow(idf, 2)) for x in
                 freq_query]))

        for tup in post_for_term:
            normalization_doc = tup[1] / read_DL.get(tup[0], 1)  # fij/|d|

            if tup[0] not in cosine_score:
                cosine_score[tup[0]] = (normalization_doc * normalization_term * math.pow(idf, 2)) / (
                        quer_nurma * read_doc_TFIDFnorma.get(tup[0], 1))
            else:
                cosine_score[tup[0]] += (normalization_doc * normalization_term * math.pow(idf, 2)) / (
                        quer_nurma * read_doc_TFIDFnorma.get(tup[0], 1))
    sorted_list = sorted(cosine_score.items(), key=lambda x: x[1], reverse=True)[:100]
    res = list(map(lambda x: (x[0], read_title_id_voc[x[0]]), sorted_list))

    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    rank_dict = {}
    for term in np.unique(tokens):
        if term in read_title_index.df:
            post_for_term = read_posting_list(read_title_index, './postings_title', term)
            for tup in post_for_term:
                if tup[0] not in rank_dict:
                    rank_dict[tup[0]] = 1
                else:
                    rank_dict[tup[0]] += 1
    sorted_list = sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)
    res = list(map(lambda x: (x[0], read_title_id_voc[x[0]]), sorted_list))

    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    rank_dict = {}
    for term in np.unique(tokens):
        if term in read_anchor_index.df:
            post_for_term = sorted(read_posting_list(read_anchor_index, './postings_anchor', term), key=lambda x: x[0])
            for tup in post_for_term:
                if tup[0] not in rank_dict:
                    rank_dict[tup[0]] = 1
                else:
                    rank_dict[tup[0]] += 1
    sorted_list = sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)
    res = list(map(lambda x: (x[0], read_title_id_voc.get(x[0])), sorted_list))

    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    for doc in wiki_ids:
        res.append(pagerank.get(doc,0))

    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)

    for doc in wiki_ids:
        res.append(pageview.get(doc,0))

    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
