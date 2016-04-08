Features:

73 - basic features, counts, ratios
5 - brand-related
111 - variations of above, with completely different typo cleaning
79 - special words in query, title or description that seemed to correlate with difference in relevance
6 - (dangerous - best removed to avoid leak) depends on split, tries to extract information from 
similar words - counts word co-occurences between query and title/description (a bit like simplified ALS)
2 - dot product between word2vec of query and title/descr
2 - Euclidean distance between word2vec of query and title/descr
900 - full averaged word2vec vectors for query, title, desc
