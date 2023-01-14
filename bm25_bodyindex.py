
from combine_helper import *


class BM25_from_index:

    def __init__(self, index, DL, k1=1.2, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N  # average document size
        self.DL = DL

    # this function calculates the idf for all the terms in the query
    def calc_idf(self, list_of_tokens):

        idf = {}
        for term in list_of_tokens:
            if term in self.index.df:
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    # this function calculates the BM25 results for the documents relevant to the query
    def search(self, query, terms_postings):
        query_idf = self.calc_idf(query)

        documents_scores = {}
        sum_of_all = 0

        for key, value in terms_postings.items():
            for doc_id, tf in value:
                doc_len = self.DL[doc_id]
                numerator = query_idf[key] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                score = (numerator / denominator)
                sum_of_all += score
                if doc_id in documents_scores:
                    documents_scores[doc_id] = documents_scores[doc_id] + score
                else:
                    documents_scores[doc_id] = score

        sum_of_all = max(documents_scores.values())
        return documents_scores, sum_of_all

    # this function calculates the BM25 results to the documents relevant to the query but only to the documents in the dicuments list the function is receving
    def search_from_list(self, query, terms_postings, documents):
        query_idf = self.calc_idf(query)

        documents_scores = {}
        sum_of_all = 0

        for key, value in terms_postings.items():
            for doc_id, tf in value:
                if doc_id in documents:
                    doc_len = self.DL[doc_id]
                    numerator = query_idf[key] * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score = (numerator / denominator)
                    sum_of_all += score
                    if doc_id in documents_scores:
                        documents_scores[doc_id] = documents_scores[doc_id] + score
                    else:
                        documents_scores[doc_id] = score

        # sum_of_all = max(documents_scores.values())
        return documents_scores
