
from inverted_index_gcp_final import *
from write_to_memory import *
from collections import defaultdict, Counter
import re
import nltk
import numpy as np

nltk.download('stopwords')

from nltk.corpus import stopwords
from contextlib import closing

import math

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)


# this function reads a posting list for a term in a certain index from a directory in the disk.
def read_posting_list(inverted, directory, w):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, directory, inverted.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


# this function calculates the cosine similarity for the index on the body of the text
def cosine_simlarity(query, terms_postings, read_DL, read_body_index, read_doc_TFIDFnorma):
    N_corpus = len(read_DL)
    cosine_score = {}
    freq_query = Counter(query)
    sumOfAll = 0
    for term in freq_query:

        if term in read_body_index.df.keys():
            post_for_term = terms_postings[term]
            normalization_term = freq_query[term] / len(query)
            idf = math.log(N_corpus / len(post_for_term), 2)
            quer_nurma = np.sqrt(
                sum([(freq_query[x] / len(query)) * (freq_query[x] / len(query) * math.pow(idf, 2)) for x in
                     freq_query]))

            for tup in post_for_term:
                normalization_doc = tup[1] / read_DL.get(tup[0], 1)  # fij/|d|

                if tup[0] not in cosine_score:
                    score = (normalization_doc * normalization_term * math.pow(idf, 2)) / (
                            quer_nurma * read_doc_TFIDFnorma.get(tup[0], 1))
                    cosine_score[tup[0]] = score
                    sumOfAll += score
                else:
                    score = (normalization_doc * normalization_term * math.pow(idf, 2)) / (
                            quer_nurma * read_doc_TFIDFnorma.get(tup[0], 1))
                    cosine_score[tup[0]] += score
                    sumOfAll += score

    # this part of the function filters only the documents that their cosine score is bigger than average of the scores multiuple 1.25
    average = sumOfAll / len(cosine_score)
    final_res = {}
    for key, value in cosine_score.items():
        if value >= average * 1.25:
            final_res[key] = value

    return final_res


# this function merges between the results of the similarity calculation on the body of the article and the similarity calculation on the title of the article
def merge(body_list, title_list, body_W, title_W):
    final_res = {}

    for doc_id, body_score in body_list.items():
        score = body_score * body_W + title_list.get(doc_id, 0) * title_W
        final_res[doc_id] = score
        title_list.pop(doc_id, 0)

    for doc_id, title_score in title_list.items():
        score = title_score * title_W
        final_res[doc_id] = score

    return sorted(final_res.items(), key=lambda x: x[1], reverse=True)[:100]


# this function maps between the documents in score_dictionary to their page rank score
def addPageRank(score_dictionary, pagerank):
    pagerank_list = []

    for doc_id, score in score_dictionary:
        pagerank_list.append((doc_id, pagerank.get(doc_id, 0)))

    return sorted(pagerank_list, key=lambda x: x[1], reverse=True)[:50]


# def getTopK(items, k):
#     results = []
#     if k >= len(items):
#         return sorted(items.items(), key=lambda x: x[1], reverse=True)
#     else:
#         for i in range(k):
#             maxValue = 0
#             maxItem = 0
#             for key, value in items.items():
#                 if value > maxValue:
#                     maxValue = value
#                     maxItem = key
#             results.append((maxItem, maxValue))
#             items.pop(maxItem)
#
#     return results


# this function calculates the similarity between the titles of the documents and the query
def title_score_list(query, read_title_index):
    count = 0
    rank_dict = {}
    for term in np.unique(query):
        if term in read_title_index.df:
            post_for_term = read_posting_list(read_title_index, './postings_title', term)
            for tup in post_for_term:
                if tup[0] not in rank_dict:
                    rank_dict[tup[0]] = 1
                    count = count + 1
                else:
                    rank_dict[tup[0]] += 1
                    count = count + 1

    return rank_dict, count


# def expend_query(query, glove_vectors):
#     most_sim = glove_vectors.most_similar(positive=query)[:2]
#     for word, score in most_sim:
#         # if score >= 0.7:
#         #     query.append(word)
#         query.append(word)
#     return query
