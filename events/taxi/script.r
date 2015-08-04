

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


//
paths <- read.csv("/Users/renatb/data/kaggle/taxi_trip/pathsForTripA.csv", header=FALSE)
ps <- subset(paths, paths$V2 < -8.5 & paths$V2 > -8.7 & paths$V3 < 41.25 & paths$V3 > 41.10)
plot(a$V1*15, a$V2, pch = 20, cex=0.3, xlab="Trip Time", ylab="Count", main='Trip Time distribution')
plot(log(a$V1*15+1), a$V2, pch = 20, cex=0.3, xlab="log(Trip Time)", ylab="Count", main='Trip Time distribution')

legend(-8.8,41.7,unique(ps$V6),col=1:length(ps$V6),pch=19)
> table(ps$V6)

   10    11    12    13    14    15    18    19    23    24    25    27    32    34    52    55    57    58    60    61
  802    43    80   346   263 49671   971   157   341   102   136   200    76    63   209    57   498    59   113   884
    9     O
 1830 13942
> plot(ps$V1, ps$V2, col=ps$V6, pch = 20, cex=0.1, xlab="Lon", ylab="Lat", main="Co-occuring paths")



