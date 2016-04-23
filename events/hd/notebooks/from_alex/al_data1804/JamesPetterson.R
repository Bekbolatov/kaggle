#James



dtr <- fread('~/data/kaggle/homedepot/train.csv')
dte <- fread('~/data/kaggle/homedepot/test.csv')
length(intersect(
  dtr[, sprintf('%s_%s', product_title, search_term)],
  dte[, sprintf('%s_%s', product_title, search_term)]
))


