
import java.io.FileWriter

import com.sparkydots.kaggle.avito.features.FeatureGeneration
import com.sparkydots.kaggle.avito.features.FeatureGeneration._
import com.sparkydots.kaggle.avito.functions.DFFunctions._
import com.sparkydots.kaggle.avito.load.LoadSave
import com.sparkydots.kaggle.avito.optimization.LogisticRegressionLogLoss

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SQLContext}

import com.sparkydots.kaggle.avito._

Logger.getLogger("amazon.emr.metrics").setLevel(Level.OFF)
Logger.getLogger("com.amazon.ws.emr").setLevel(Level.WARN)
Logger.getLogger("org").setLevel(Level.WARN)
Logger.getLogger("akka").setLevel(Level.WARN)


//sqlContext.udf.register("strLen", (s: String) => s.length())
//sqlContext.udf.register("errf", _error)



val (sqlContext, users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests, evalData, trainData, validateData, smallData) = Script.run(sc)

evalData.cache()
smallData.cache()
LoadSave.saveDF(sqlContext, evalData, "ANAL_EVAL")
LoadSave.saveDF(sqlContext, smallData, "ANAL_SMALL")
LoadSave.saveDF(sqlContext, trainData, "ANAL_TRAIN")
LoadSave.saveDF(sqlContext, validateData, "ANAL_VALIDATE")

val trainData = LoadSave.loadDF(sqlContext, "ANAL_TRAIN")
val validateData = LoadSave.loadDF(sqlContext, "ANAL_VALIDATE")
val train = featurize(trainData).toDF.cache()
val validate = featurize(validateData).toDF.cache()

val lr = new LogisticRegression()
lr.setMaxIter(10).setRegParam(0.01)

val paramMap = ParamMap(lr.maxIter -> 10)

val model = lr.fit(train, ParamMap(lr.maxIter -> 40, lr.regParam -> 0.01))
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._
val tre = model.transform(train).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0)))
val vae = model.transform(validate).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0)))
df_calcError(tre.toDF)
df_calcError(vae.toDF)


val model2 = lr.fit(train, ParamMap(lr.maxIter -> 50, lr.regParam -> 0.01))
val tre2 = model2.transform(train).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0)))
val vae2 = model2.transform(validate).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0)))
df_calcError(tre2.toDF)
df_calcError(vae2.toDF)


val model3 = lr.fit(train, ParamMap(lr.maxIter -> 50, lr.regParam -> 0.1))
val tre3 = model3.transform(train).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0)))
val vae3 = model3.transform(validate).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0)))
df_calcError(tre3.toDF)
df_calcError(vae3.toDF)









val modelx = lr.fit(eval, ParamMap(lr.maxIter -> 40, lr.regParam -> 0.01))
import sqlContext.implicits._
val smallPred = modelx.transform(small).select("label", "probability").map(x => (x.getDouble(0).toInt, x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1))).toDF("id", "pred").orderBy("id").cache()

val preds = smallPred.collect()
preds.foreach { case () =>

}

val dtr1 = sqlContext.load(s"s3n://sparkydotsdata/kaggle/avito/processed/data_train_1.parquet")






val smallPred = predsRaw.orderBy("label").map({ case Row(p: Double, l: Double) => (l.toInt, p) }).cache

show()

val sub = new FileWriter("/home/hadoop/sub2.csv", true)
sub.write("ID,IsClick\n")

val preds = smallPred.collect()
preds.foreach { case (id, prob) =>
  sub.write(id + "," + f"$prob%1.6f" + "\n")
}
sub.close()

val preds = predsRaw.orderBy("label").map({ case Row(p: Double, l: Double) => (l.toInt, p) }).collect
val sub = new FileWriter("/home/hadoop/subr2.csv", true)
sub.write("ID,IsClick\n")
preds.foreach { case (id, prob) =>
  sub.write(id + "," + f"$prob%1.8f" + "\n")
}
sub.close()


smallPred.repartition(1).save(s"s3n://sparkydotsdata/kaggle/avito/processed/pred23.csv", "com.databricks.spark.csv")

smallPred.save(s"s3n://sparkydotsdata/kaggle/avito/processed/smallPred", "com.databricks.spark.csv")


val trainData = LoadSave.loadDF(sqlContext, "FINAL_TRAIN")
val validateData = LoadSave.loadDF(sqlContext, "FINAL_VALIDATE")

val train = featurize(trainData).toDF.cache()
val validate = featurize(validateData).toDF.cache()

val eval = featurize(evalData).toDF.cache()
val small = featurize(smallData).toDF.cache()


val lr = new LogisticRegression()
lr.setMaxIter(10).setRegParam(0.01)

val paramMap = ParamMap(lr.maxIter -> 10)



scala> counts.describe("cnt").show
summary cnt
  count   24371
mean    3196.972672438554
stddev  2632.969040531403
min     1
max     10097377

scala> counts.describe("cnt").show
summary cnt
  count   24786
mean    3143.4447268619383
stddev  2782.0324339154226
min     1
max     1841593

scala> counts.describe("cnt").show
summary cnt
  count   28252
mean    2757.8019609231205
stddev  3131.674696169024
min     1
max     1735418






scala> rawEval.filter("title like 'chic-robot%'").show
isClick os   uafam visitCount phoneCount impCount clickCount searchTime searchQuery searchLoc searchCat searchParams     loggedIn position histctr  category params           price   title                adImpCount adClickCount searchLocLevel searchLocPar searchCatLevel searchCatPar adCatLevel adCatPar
0       null null  null       null       null     null       null       null        null      null      null             null     7        0.004065 47       ArrayBuffer(127) 39500.0 chic-robot smart ... 15575      104          1              -1           1              10           3          3
0       30   9     null       null       null     null       2061652                3953      47        ArrayBuffer()    0        7        0.004381 47       ArrayBuffer(127) 39500.0 chic-robot smart ... 15575      104          2              47           3              3            3          3
0       30   64    48         4          26       1          2321990                900       47        ArrayBuffer(127) 0        7        0.003502 47       ArrayBuffer(127) 39500.0 chic-robot smart ... 15575      104          3              75           3              3            3          3
0       20   25    28         4          20       null       2231568                3198      47        ArrayBuffer()    0        7        0.00461  47       ArrayBuffer(127) 39500.0 chic-robot smart ... 15575      104          3              56           3              3            3          3


trainData.filter("isClick = 1 ").agg(avg("price"), max("price"), min("price"), avg("phoneCount"), max("phoneCount"), min("phoneCount"), avg("impCount"), avg("clickCount")).show(30)
trainData.filter("isClick = 0 ").agg(avg("price"), max("price"), min("price"), avg("phoneCount"), max("phoneCount"), min("phoneCount"), avg("impCount"), avg("clickCount")).show(30)
trainData.filter("isClick = 0 ").withColumn("huy", trainData("clickCount")/trainData("impCount")).agg(avg("huy"), avg("visitCount"), max("visitCount"), min("visitCount"), avg("phoneCount"), max("phoneCount"), min("phoneCount"), avg("impCount"), avg("clickCount")).show(30)


trainData.filter("isClick = 1 ").withColumn("huy", trainData("clickCount")/trainData("impCount")).withColumn("ahuy", trainData("adClickCount")/trainData("adImpCount")).
  agg(avg("huy"), avg("ahuy"), avg("visitCount"), avg("impCount"), avg("clickCount"), avg("phoneCount"), max("phoneCount"), min("phoneCount")).show(30)

trainData.filter("isClick = 0").withColumn("huy", trainData("clickCount")/trainData("impCount")).withColumn("ahuy", trainData("adClickCount")/trainData("adImpCount")).
  agg(avg("huy"), avg("ahuy"), avg("visitCount"), avg("impCount"), avg("clickCount"), avg("phoneCount"), max("phoneCount"), min("phoneCount")).show(30)


scala>

trainData.filter("isClick = 1 ").withColumn("huy", trainData("clickCount")/trainData("impCount")).agg(avg("huy"), avg("visitCount"), max("visitCount"), min("visitCount"), avg("phoneCount"), max("phoneCount"), min("phoneCount"), avg("impCount"), avg("clickCount")).show(30)
AVG(huy)            AVG(visitCount)   MAX(visitCount) MIN(visitCount) AVG(phoneCount)    MAX(phoneCount) MIN(phoneCount) AVG(impCount)      AVG(clickCount)
0.06670347475079208 203.5222677995898 3920            1               10.448871960152358 1444            1               122.45912686785819 2.231614415470261

scala> trainData.filter("isClick = 0 ").withColumn("huy", trainData("clickCount")/trainData("impCount")).agg(avg("huy"), avg("visitCount"), max("visitCount"), min("visitCount"), avg("phoneCount"), max("phoneCount"), min("phoneCount"), avg("impCount"), avg("clickCount")).show(30)
AVG(huy)            AVG(visitCount)  MAX(visitCount) MIN(visitCount) AVG(phoneCount)    MAX(phoneCount) MIN(phoneCount) AVG(impCount)     AVG(clickCount)
0.03143051744022667 288.206488627826 37953           1               12.579180042346835 2230            1               273.9703708763063 2.155349702889147




val (sqlContext, users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests, evalData, trainData, validateData, smallData) = Script.run(sc)


LoadSave.saveDF(sqlContext, trainData, "FINAL_TRAIN")
LoadSave.saveDF(sqlContext, evalData, "FINAL_EVAL")
LoadSave.saveDF(sqlContext, validateData, "FINAL_VALIDATE")
LoadSave.saveDF(sqlContext, smallData, "FINAL_SMALL")




val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._

val (sqlContext, rawTrain, rawValidate, rawTest, lr, paramMap) = com.sparkydots.kaggle.avito.Script.loadThem(sc)

val (train, validate, test, model)  = Script.fit(sc, sqlContext, rawTrain, rawValidate, lr, paramMap)
train.cache()
validate.cache()





-- bits = 15
val numBuckets = 15
val featureGen15 = new FeatureGeneration(sqlContext, numBuckets)
val numFeatures15 = math.pow(2, numBuckets).toInt
val train15 = featureGen15.featurize(rawTrain).cache()
val validate15 = featureGen15.featurize(rawValidate).cache()

val model15 = lr.fit(train15, ParamMap(lr.maxIter -> 40, lr.regParam -> 0.01))
df_calcError(model15.transform(train15).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
df_calcError(model15.transform(validate15).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

//  bits = 20
val numBits = 20
val featureGen20 = new FeatureGeneration(sqlContext, numBits)
val numFeatures20 = math.pow(2, numBits).toInt
val train20 = featureGen20.featurize(rawTrain).cache()
val validate20 = featureGen20.featurize(rawValidate).cache()

val model20 = lr.fit(train20, ParamMap(lr.maxIter -> 40, lr.regParam -> 0.01))
val tte20 = df_calcError(model20.transform(train20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
val tve20 = df_calcError(model20.transform(validate20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

//  bits = 20, iter= 50
val model20_50 = lr.fit(train20, ParamMap(lr.maxIter -> 50, lr.regParam -> 0.01))
val tte20_50 = df_calcError(model20_50.transform(train20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
val tve20_50 = df_calcError(model20_50.transform(validate20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

//  bits = 20, iter= 20
val model20_20 = lr.fit(train20, ParamMap(lr.maxIter -> 20, lr.regParam -> 0.01))
val tte20_20 = df_calcError(model20_20.transform(train20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
val tve20_20 = df_calcError(model20_20.transform(validate20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)


//  bits = 20, iter= 10
val model20_10 = lr.fit(train20, ParamMap(lr.maxIter -> 10, lr.regParam -> 0.01))
val tte20_10 = df_calcError(model20_10.transform(train20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
val tve20_10 = df_calcError(model20_10.transform(validate20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)


val model20_30_r3 = lr.fit(train20, ParamMap(lr.maxIter -> 30, lr.regParam -> 0.03))
val tte20_30_3 = df_calcError(model20_30_r3.transform(train20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
val tve20_30_3 = df_calcError(model20_30_r3.transform(validate20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)


val mymodel1_20 = LogisticRegressionLogLoss.fit(train20, 30, 0.01, math.pow(2, 20).toInt)
val tte20_m1_20 = df_calcError(mymodel1_20.transform(train20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
val tve20_m1_20 = df_calcError(mymodel1_20.transform(validate20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

val mymodel1_15 = LogisticRegressionLogLoss.fit(train15, 30, 0.01)
val tte15_m1 = df_calcError(mymodel1_15.transform(train15).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
val tve15_m1 = df_calcError(mymodel1_15.transform(validate15).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

val mymodel1_15_40 = LogisticRegressionLogLoss.fit(train15, 40, 0.01)
val tte15_m1_40 = df_calcError(mymodel1_15_40.transform(train15).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
val tve15_m1_40 = df_calcError(mymodel1_15_40.transform(validate15).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)


//  bits = 20
val numBits = 20
val featureGen20 = new FeatureGeneration(sqlContext, numBits)
val numFeatures20 = math.pow(2, numBits).toInt
val train20 = featureGen20.featurize(rawTrain).cache()
val validate20 = featureGen20.featurize(rawValidate).cache()

val model20 = lr.fit(train20, ParamMap(lr.maxIter -> 40, lr.regParam -> 0.01))
val tte20 = df_calcError(model20.transform(train20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
val tve20 = df_calcError(model20.transform(validate20).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)








import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

// Train a DecisionTree model.
//  Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 5
val maxBins = 32

val oldDataset = dataset.select("label", "features").map { case Row(label: Double, features: Vector) => LabeledPoint(label, features) }

oldDataset.cache

val model = DecisionTree.trainClassifier(oldDataset, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)













>>>
val modelx = lr.fit(eval, ParamMap(lr.maxIter -> 40, lr.regParam -> 0.01))
df_calcError(modelx.transform(eval).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
val smallPred = modelx.transform(small).select("probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](0)(1))).toDF("pred")

>>>

---- overfitting

val model7 = lr.fit(train, ParamMap(lr.maxIter -> 70))
df_calcError(model7.transform(train).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
df_calcError(model7.transform(validate).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

///// Sparcity
val vec = train.rdd.map(x=>x.getAs[org.apache.spark.mllib.linalg.SparseVector](1)).take(1)(0)
