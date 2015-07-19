
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



val (sqlContext, rawTrain, rawValidate, rawTest, lr, paramMap) = com.sparkydots.kaggle.avito.Script.loadThem(sc)

Script.fit(sc, sqlContext, rawTrain, rawValidate, lr, paramMap)



fit(sc, sqlContext, rawTrain, rawValidate, lr, paramMap)

calcError(model.transform(train).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
calcError(model.transform(validate).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)





val model2 = lr.fit(train, ParamMap(lr.maxIter -> 20))
calcError(model2.transform(train).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
calcError(model2.transform(validate).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)




val model3 = lr.fit(train, ParamMap(lr.maxIter -> 30))
calcError(model3.transform(train).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
calcError(model3.transform(validate).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)




val model4 = lr.fit(train, ParamMap(lr.maxIter -> 40))
calcError(model4.transform(train).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
calcError(model4.transform(validate).select("label", "probability").map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
