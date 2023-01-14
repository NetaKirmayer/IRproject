# IRproject - Neta Kirmayer & Rony Shir

### List of files:

1. **FinalAllIndexCreation.ipynb** - this file contains all the creation of our indexes. For the body index we calculated additional dictionaries such as document length and norma of the document for later similarity checks.
2. **bm25_bodyindex.py** - this file contains a class that calculates the BM25 scores for a given index and query. The class recives an index in creation for it to work on and the search/search_from_list function do the actual calculations.
3. **combine_helper.py** - this file containes some of the helper functions we used for our main search function in the file 'search_frontend.py'.
4. **inverted_index_gcp_final.py** - this file containes the InvertedIndex class that keeps the global dictionaries of our index. 
                                     in addition to that it containes the classes MultiFileWriter and MultiFileReader that write/read to/from to the memory.
5. **search_frontend.py** - this file containes all of the search functions we created.
6. **write_to_memory.py** - this  file containes reading and writing pickle file functions from the local memory.


### The main search function:
In our main search function the data from out index moves throw several stages entil we reach a final list of documents that we return.
1. The first step in our function is tokenizing the query and removeing stopwords/ terms that are not in our index. 
2. According to the length of the reduced query we determine the weigths for our merge function. If the query is only one word we give a higher weight to the title and if the query is longer than a word we give a higher weigth to the body of the document.
3. We calculate similarity between the body of the document and the query using cosine similarity and the weigths are tf-idf. We return fron the function only the documents that their similarity score is higher than a threshold value.
4. We calculate the similarity between the query and the document's title using binery similarity.
5. We use a merge function to merge the results of the similerity with the title and the body according to the weigths we determined. We return from the function only the documents that their merged scores are the top 100.
6. On the documents the merge function returned we map between the doc-id and pageRank value a and we return the documents that their pageRank value are the top 50.
7. On the documents we returned from the pageRank filtering we preform a calculation of the BM25 score. After the calculation we resort the documents according to the BM25 scores and return the top 40.
8. We map between the doc id and the Title of the document and return a tuples list.  
