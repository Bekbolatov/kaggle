package com.sparkydots.kaggle.avito

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


object Script {

  //   SPARK_REPL_OPTS="-XX:+CMSClassUnloadingEnabled -XX:MaxPermSize=512m -Xmx=4g" spark-shell --jars AvitoProject-assembly-1.0.jar
  //   import com.sparkydots.kaggle.avito._
  //   val (users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests, trainData, validateData, testData) = Script.run(sc)
  //   TrainingData.calcErrors(ctxAdImpressions, trainSet, validateSet, testSet)


  // val (sqlContext, rawTrain, rawValidate, rawTest, lr, paramMap) = com.sparkydots.kaggle.avito.Script.loadThem(sc)

  //sqlContext.udf.register("strLen", (s: String) => s.length())
  //sqlContext.udf.register("errf", _error)


  def run(sc: SparkContext, orig: Boolean = false) = {

    Logger.getLogger("amazon.emr.metrics").setLevel(Level.OFF)
    Logger.getLogger("com.amazon.ws.emr").setLevel(Level.WARN)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val (users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests) =
      if (orig)
        LoadSave.origLoad(sqlContext)
      else
        LoadSave.load(sqlContext)

    val (evalData, trainData, validateData, smallData) =
      TrainingData.split(sqlContext, users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests)

    (sqlContext, users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests, evalData, trainData, validateData, smallData)
  }


  def loadThem(sc: SparkContext) = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val rawTrain = LoadSave.loadDF(sqlContext, "data_train_1").cache()
    val rawValidate = LoadSave.loadDF(sqlContext, "data_validate_1").cache()
    val rawTest = LoadSave.loadDF(sqlContext, "data_test_1").cache()

    val lr = new LogisticRegression()
    lr.setMaxIter(10).setRegParam(0.01)

    val paramMap = ParamMap(lr.maxIter -> 10)

    (sqlContext, rawTrain, rawValidate, rawTest, lr, paramMap)
  }

  def fit(sc: SparkContext, sqlContext: SQLContext, trainData: DataFrame, testData: DataFrame, validateData: DataFrame, lr: LogisticRegression, paramMap: ParamMap) = {
    import sqlContext.implicits._

//    val eval = featurize(evalData).toDF.cache()
//    val small = featurize(smallData).toDF.cache()
    val train = featurize(trainData).toDF.cache()
    val validate = featurize(validateData).toDF.cache()
    val test = featurize(testData).toDF.cache()

    val model = lr.fit(train, paramMap)

    val errorTrain = calcError(model.transform(train).select("prediction", "label"))
    val errorValidate = calcError(model.transform(validate).select("prediction", "label"))

    println(s"Train error: $errorTrain, Test error: $errorValidate")

    (train, validate, test, model)
  }

  def fitExample(sc: SparkContext, sqlContext: SQLContext) = {

    import sqlContext.implicits._


    val training = sc.parallelize(Seq(
      LabeledPoint(1.0, Vectors.dense(0.0, 1.1, 0.1)),
      LabeledPoint(0.0, Vectors.dense(2.0, 1.0, -1.0)),
      LabeledPoint(0.0, Vectors.dense(2.0, 1.3, 1.0)),
      LabeledPoint(1.0, Vectors.dense(0.0, 1.2, -0.5))))

    val lr = new LogisticRegression() //This instance is an Estimator: LogisticRegression  - use 'lr.explainParams()' to see params
    lr.setMaxIter(10).setRegParam(0.01)

    val model1 = lr.fit(training.toDF) // Learn a LogisticRegression model, with parameters stored in lr.
    // Since model1 is a Model (i.e., a Transformer produced by an Estimator), we can view the parameters it used during fit().
    // This prints the parameter (name: value) pairs, where names are unique IDs for this LogisticRegression instance.
    println("Model 1 was fit using parameters: " + model1.fittingParamMap)

    val paramMap = ParamMap(lr.maxIter -> 20)
    paramMap.put(lr.maxIter, 30) // Specify 1 Param.  This overwrites the original maxIter.
    paramMap.put(lr.regParam -> 0.1, lr.threshold -> 0.55) // Specify multiple Params.
    val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability") // Change output column name
    val paramMapCombined = paramMap ++ paramMap2

    val model2 = lr.fit(training.toDF, paramMapCombined) //learn a new model using the paramMapCombined parameters
    println("Model 2 was fit using parameters: " + model2.fittingParamMap)

    val test = sc.parallelize(Seq(
      LabeledPoint(1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      LabeledPoint(0.0, Vectors.dense(3.0, 2.0, -0.1)),
      LabeledPoint(1.0, Vectors.dense(0.0, 2.2, -1.5))))

    // Make predictions on test data using the Transformer.transform() method.
    // LogisticRegression.transform will only use the 'features' column.
    // Note that model2.transform() outputs a 'myProbability' column instead of the usual
    // 'probability' column since we renamed the lr.probabilityCol parameter previously.
    model2.transform(test.toDF).
      select("features", "label", "myProbability", "prediction").
      collect().
      foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
      println(s"($features, $label) -> prob=$prob, prediction=$prediction")
    }

  }

}
