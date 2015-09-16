
def stem(word):
    if not word:
        return ''
    if len(word) < 4:
        return word
    if word.endswith('s') or word.endswith('e') or word.endswith('y'): 
        return stem(word[:-1]) # books -> book, create -> creat, assembly -> assembl
    if word.endswith('ment'):
        return stem(word[:-4]) #establishment -> establish
    if word.endswith('tion'):
        return stem(word[:-3]) #constriction -> constrict
    if word.endswith('ed') or word.endswith('or') or word.endswith('er'):
        return stem(word[:-2]) # started -> start, creator -> creat, builder -> build
    if word.endswith('ea') or word.endswith('ia'):
        return stem(word[:-2]) # russia -> russ, asia -> as
    if word.endswith('ean') or word.endswith('ian'):
        return stem(word[:-3]) # russian -> russ, asian -> as
    if word.endswith('ies'):
        return stem(word[:-3]) # responsibilities -> responsibilit
    return word
