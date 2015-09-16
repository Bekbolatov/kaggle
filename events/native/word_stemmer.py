from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def stem(word):
    if not word:
        return ''
    if len(word) < 4:
        return word

    word = stemmer.stem(word)

    if word.endswith('ea') or word.endswith('ia'):
        return word[:-2] # russia -> russ, asia -> as
    if word.endswith('ean') or word.endswith('ian'):
        return word[:-3] # russian -> russ, asian -> as

    return word

