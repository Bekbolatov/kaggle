
def lemmatize(w):
    n = len(w)
    if n <= 4:
        return w

    l1 = w[-1]
    l21 = w[-2:]
    l321 = w[-3:]
    l4321 = w[-4:]
    if l1 == 's' and l21 != 'ss' and l21 != 'os' and l21 != 'us' and n >= 3:
        return lemmatize(w[:-1])
    if (l21 == 'ly' or l21 == 'ic' or l21 == 'ed') and n >= 4:
        return w[:-2]
    if l1 == 'y':
        return w[:-1] + 'i'
    if l321 == 'ion' and n > 6:
        return w[:-3]
    if (l4321 == 'able' or l4321 == 'ment') and n > 5: #noble
        return w[:-4]
    if l4321 == 'ible' and n > 5: #noble
        return w[:-4]
    if l321 == 'ing' and n > 5: #noble
        return w[:3]
    if (l321 == 'est' or l321 == 'ost' or l321 == 'ial' or l321 == 'eal') and n > 5 :
        return w[:-3]
    if l21 == 'or' or l21 == 'er' or l21 == 'al':
        return lemmatize(w[:-2])
    if (l4321 == 'ance') and n > 6:
        return w[:-4]
    if (l4321 == 'ence') and n > 6:
        return w[:-2] + 't'
    if (l4321 == 'ince') and n > 6:
        return w[:-2] + 'c'
    if l21 == 'ee':
        return w
    if l1 == 'e':
        return w[:-1]

    return w


def testit():
    print(lemmatize('dogs'))
    print(lemmatize('spy'))
    print(lemmatize('spies'))
    print(lemmatize('vacationer'))
    print(lemmatize('vacation'))
    print(lemmatize('vacations'))
    print(lemmatize('vacate'))
    print(lemmatize('noble'))
    print(lemmatize('despicable'))
    print(lemmatize('able'))
    print(lemmatize('horrible'))
    print(lemmatize('loving'))
    print(lemmatize('eating'))
    print(lemmatize('pavers'))
    print(lemmatize('paver'))
    print(lemmatize('paving'))
    print(lemmatize('stronger'))
    print(lemmatize('strongest'))
    print(lemmatize('automatic'))
    print(lemmatize('automated'))
    print(lemmatize('beads'))
    print(lemmatize('harnesses'))
    print(lemmatize('presidential'))
    print(lemmatize('presidentials'))
    print(lemmatize('presidence'))
    print(lemmatize('president'))
    print(lemmatize('presidents'))
    print(lemmatize('province'))
    print(lemmatize('provinces'))
    print(lemmatize('provincial'))
    print(lemmatize('disposal'))
    print(lemmatize('disposals'))
    print(lemmatize('disposer'))
    print(lemmatize('disposers'))
    print(lemmatize('establishment'))
    print(lemmatize('establishments'))
    print(lemmatize('gullible'))
    print(lemmatize('bee'))
    print(lemmatize('tree'))
    print(lemmatize('trees'))
    print(lemmatize('governance'))
    print(lemmatize('govern'))
    print(lemmatize('governor'))
    print(lemmatize('user'))
    print(lemmatize('use'))
    print(lemmatize('error'))


#testit()

