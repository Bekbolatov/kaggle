import re
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import nltk.stem.wordnet
from nltk.tokenize import word_tokenize
import gensim
import numpy as np

#morphy_tag = {'NN':wordnet.NOUN,'JJ':wordnet.ADJ,'VB':wordnet.VERB,'RB':wordnet.ADV}
morphy_tag = {'NN':'n','JJ':'a','VB':'v','RB':'r'}
def tag(penn_tag):
    tag = penn_tag[:2]
    if tag in morphy_tag:
        return morphy_tag[tag]
    else:
        return 'n'

def ascii(word):
    try:
        return word.encode('ascii', 'ignore')
    except Exception as e:
        return ''

stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatiser = WordNetLemmatizer()
def lemmatize(s):
    tokens = word_tokenize(s) # Generate list of tokens
    tokens = [ascii(token) for token in tokens]
    tokens = [token for token in tokens if token.lower() not in stopwords]
    tokens_pos = pos_tag(tokens) 
    return [lemmatiser.lemmatize(word, pos=tag(t)) for word, t in tokens_pos if len(word) > 2]


word_splitter = re.compile(r'[a-z]{3,}')
def words(text):
    lemmas = lemmatize(text)
    lemmas = [lem for lemma in lemmas for lem in word_splitter.findall(lemma.lower())]
    return lemmas

def word_set(text):
    return set(words(text))
    
word2vec_model = gensim.models.word2vec.Word2Vec.load_word2vec_format('/home/ec2-user/data/word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)
def sentence2vec(s):
    words = lemmatize(s)
    total = np.zeros(300)
    for word in words:
        if word in word2vec_model:
            total += word2vec_model[word]
    norm = np.dot(total, total)
    if norm > 0:
        return total / np.sqrt(norm)
    else:
        return total

