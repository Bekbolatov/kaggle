
def stem(word):
    if not word or len(word) < 4:
        return ''
    if word.endswith('s') or word.endswith('e'):
        return stem(word[:-1]) # books -> book
    if word.endswith('ment'):
        return stem(word[:-4]) #establishment -> establish
    if word.endswith('tion'):
        return stem(word[:-3]) #constriction -> constrict
    if word.endswith('ed') or word.endswith('or') or word.endswith('er'):
        return stem(word[:-2]) # started -> start, creator -> creat, builder -> build
    return word
