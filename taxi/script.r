

library(rgl)

ds2 <- read.csv(pipe("hdfs dfs -cat /tmp/first_times_train.txt/*"), header = FALSE)


ds <- subset(ds2, ds2$V1 < -8.5 & ds2$V1 > -8.7 & ds2$V2 > 41.10 & ds2$V2 < 41.25 & ds2$V3 < 2500)
%1658086/1704769 = 97.26%  2.74% missing


palette <- heat.colors(20)
colors <- sapply(ds$V3, function(c) palette[  16 - max(min( 15 * (c/2200.0)  ,15),1)  ])
plot3d(ds, col=colors)







////
ds <- subset(ds2, ds2$V1 < -8.5 & ds2$V1 > -8.7 & ds2$V2 > 41.10 & ds2$V2 < 41.25 & ds2$V3 < 3500)
colors <- sapply(ds$V3, function(c) palette[  16 - max(min( 15 * (c/3500.0)  ,15),1)  ])
plot3d(ds, col=colors)




/////
library(RColorBrewer)
rf <- colorRampPalette(rev(brewer.pal(11,'Spectral')))
r <- rf(32)

library(gplots)

df <- data.frame(ds$V1, ds$V2)
hist2d(df, nbins=25, col=r, FUN=function(x) log(length(x)))




//
dsx <- read.csv(pipe("hdfs dfs -cat /user/renatb/speed_train/*"), header = FALSE, stringsAsFactors = FALSE)
dsm <- dsx[sample(nrow(dsx), size=1000000), ]
