
import com.sparkydots.kaggle.avito.features.FeatureGeneration._
import com.sparkydots.kaggle.avito.functions.DFFunctions._

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



val trainData = LoadSave.loadDF(sqlContext, "FINAL_TRAIN")
val validateData = LoadSave.loadDF(sqlContext, "FINAL_VALIDATE")
val train = featurize(trainData).toDF.cache()
val validate = featurize(validateData).toDF.cache()

val lr = new LogisticRegression()
lr.setMaxIter(10).setRegParam(0.01)

val paramMap = ParamMap(lr.maxIter -> 10)



trainData.filter("isClick = 1 ").agg(avg("price"), max("price"), min("price"), avg("phoneCount"), max("phoneCount"), min("phoneCount"), avg("impCount"), avg("clickCount")).show(30)
trainData.filter("isClick = 0 ").agg(avg("price"), max("price"), min("price"), avg("phoneCount"), max("phoneCount"), min("phoneCount"), avg("impCount"), avg("clickCount")).show(30)
trainData.filter("isClick = 0 ").withColumn("huy", trainData("clickCount")/trainData("impCount")).agg(avg("huy"), avg("visitCount"), max("visitCount"), min("visitCount"), avg("phoneCount"), max("phoneCount"), min("phoneCount"), avg("impCount"), avg("clickCount")).show(30)



scala> trainData.filter("isClick = 1 ").withColumn("huy", trainData("clickCount")/trainData("impCount")).agg(avg("huy"), avg("visitCount"), max("visitCount"), min("visitCount"), avg("phoneCount"), max("phoneCount"), min("phoneCount"), avg("impCount"), avg("clickCount")).show(30)
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






val model = lr.fit(train, ParamMap(lr.maxIter -> 40, lr.regParam -> 0.01))
calcError(model.transform(train).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
calcError(model.transform(validate).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

---- overfitting

val model7 = lr.fit(train, ParamMap(lr.maxIter -> 70))
calcError(model7.transform(train).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
calcError(model7.transform(validate).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

///// Sparcity
val vec = train.rdd.map(x=>x.getAs[org.apache.spark.mllib.linalg.SparseVector](1)).take(1)(0)
