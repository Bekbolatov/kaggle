
queries: { id: query}
product_title:
product_description:
attributes: { name: value }

brand
corrected: 0/1

brand_none: 0/1
brand_unbranded: 0/1
brand_hampton: 0/1
brand_kohler: 0/1
brand_ever: 0/1
brand_home: 0/1
brand_ge: 0/1


attributes_len

queries_len
product_title_len
product_description_len
brand_len

queries_let
product_title_let
product_description_let
brand_let

product_uid



a=[
u'queries',
u'product_title',
u'product_description',
u'attributes',
u'brand',

u'corrected',

u'brand_none',
u'brand_unbranded',
u'brand_hampton',
u'brand_kohler',
u'brand_ever',
u'brand_home',
u'brand_ge',


u'attributes_len',
u'queries_len',
u'product_title_len',
u'product_description_len',
u'brand_len',

u'queries_wlen',
u'product_title_wlen',
u'product_description_wlen',
u'brand_wlen',

u'queries_let',
u'product_title_let',
u'product_description_let',
u'brand_let',

u'query_in_product_features',

]

######################
queries                       object # map{ id:text }
product_title                 object # text
product_description           object # text
attributes                    object # text
brand                         object # text

brand_none                   float64
brand_unbranded              float64
brand_hampton                float64
brand_kohler                 float64
brand_ever                   float64
brand_home                   float64
brand_ge                     float64
attributes_len                 int64
product_title_len              int64
product_description_len        int64
brand_len                      int64
product_title_let              int64
product_description_let        int64
brand_let                      int64
product_title_wlen             int64
product_description_wlen       int64
brand_wlen                     int64

corrected                     object # map{ id: num }
queries_len                   object # map{ id: num }
queries_let                   object # map{ id: num }
queries_wlen                  object # map{ id: num }
query_in_product_features     object # map{ id: [num] }
#################

direct_features = ["brand_none",
"brand_unbranded",
"brand_hampton",
"brand_kohler",
"brand_ever",
"brand_home",
"brand_ge",
"attributes_len",
"product_title_len",
"product_description_len",
"brand_len",
"product_title_let",
"product_description_let",
"brand_let",
"product_title_wlen",
"product_description_wlen",
"brand_wlen"]

query_features = ['corrected', 'queries_len', 'queries_let', queries_wlen]



