

text_data = pd.DataFrame({'query': queries['query']})
text_data['qt'] = queries['query'] + ' ' + queries['product_title']
text_data['qd'] = queries['query'] + ' ' + queries['product_description']
text_data['qtd'] = queries['query'] + ' ' + queries['product_title'] + ' ' + queries['product_description']
text_data['qtda'] = queries['query'] + ' ' + queries['product_title'] + ' ' + queries['product_description'] + ' ' + queries['attrs']

text_data = gl.SFrame(text_data)


vect_qt = gl.feature_engineering.TFIDF(features='qt', min_document_frequency=8.0/1e6)
vect_qd = gl.feature_engineering.TFIDF(features='qd', min_document_frequency=8.0/1e6)
vect_qtd = gl.feature_engineering.TFIDF(features='qtd', min_document_frequency=8.0/1e6)
vect_qtda = gl.feature_engineering.TFIDF(features='qtda', min_document_frequency=8.0/1e6)


vect_qt.fit(text_data)
vect_qd.fit(text_data)
vect_qtd.fit(text_data)
vect_qtda.fit(text_data)



item_qt_query = gl.SFrame(pd.DataFrame({'qt': queries['query'].copy()}))
item_qt_title = gl.SFrame(pd.DataFrame({'qt': queries['product_title'].copy()}))
item_qt_query = vect_qt.transform(item_qt_query)
item_qt_title = vect_qt.transform(item_qt_title)


item_qd_query = gl.SFrame(pd.DataFrame({'qd': queries['query'].copy()}))
item_qd_desc = gl.SFrame(pd.DataFrame({'qd': queries['product_description'].copy()}))
item_qd_query = vect_qd.transform(item_qd_query)
item_qd_desc = vect_qd.transform(item_qd_desc)



