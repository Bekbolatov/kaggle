package com.sparkydots.kaggle.avito

import com.sparkydots.kaggle.avito.features.{FeatureHashing, FeatureGeneration}
import com.sparkydots.kaggle.avito.functions.DFFunctions._
import com.sparkydots.kaggle.avito.optimization.LogisticRegressionLogLoss

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SQLContext}

import org.apache.spark.ml.classification.LogisticRegression


object Script {

  /*

     SPARK_REPL_OPTS="-XX:+CMSClassUnloadingEnabled -XX:MaxPermSize=1512m -Xmx=8g" spark-shell --jars AvitoProject-assembly-1.0.jar

    import org.apache.log4j.{Level, Logger}
    import org.apache.spark.sql.{Row, SQLContext}
    import com.sparkydots.kaggle.avito._
    import com.sparkydots.kaggle.avito.functions.DFFunctions._
    import com.sparkydots.kaggle.avito.functions.Functions._
    import com.sparkydots.kaggle.avito.features.FeatureGeneration
    import com.sparkydots.kaggle.avito.features.FeatureHashing
    import com.sparkydots.kaggle.avito.optimization.LogisticRegressionLogLoss
    import org.apache.spark.mllib.linalg.Vectors
    import org.apache.spark.mllib.regression.LabeledPoint
    import org.apache.spark.rdd.RDD
    import org.apache.spark.sql.{SQLContext, DataFrame}
    import scala.util.Try

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val maxIter = 35
    val bits = 15
    val numBits = bits
    val numFeatures = math.pow(2, numBits).toInt
    val regParam = 0.01


    val (rawTrain, rawValidate, rawEval, rawSmall) = com.sparkydots.kaggle.avito.Script.run(sc, sqlContext)








    val results2 = Script.tryfit(sqlContext, rawTrain, rawValidate,  Seq(15), Seq(40))

    results.foreach { case Seq(Seq( (numBits, interactions, maxIter, errorTrain, errorValidate))) =>
          println(s"${numBits}\t${maxIter}\t${errorTrain}\t${errorValidate}")
    }


    val (sqlContext, rawTrain, rawValidate, rawEval, rawSmall, train, validate, errors) = com.sparkydots.kaggle.avito.Script.run(sc)
    val results2 = Script.tryfit(sqlContext, rawTrain, rawValidate,  Seq(15), Seq(40))


*/




  def run(sc: SparkContext, sqlContext: SQLContext) = {
    Logger.getLogger("amazon.emr.metrics").setLevel(Level.OFF)
    Logger.getLogger("com.amazon.ws.emr").setLevel(Level.WARN)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val (rawTrain, rawValidate, rawEval, rawSmall) = LoadSave.loadDatasets(sc, sqlContext)
    (rawTrain, rawValidate, rawEval, rawSmall)
  }


  def run0(sc: SparkContext) = {
    Logger.getLogger("amazon.emr.metrics").setLevel(Level.OFF)
    Logger.getLogger("com.amazon.ws.emr").setLevel(Level.WARN)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val (rawTrain, rawValidate, rawEval, rawSmall) = LoadSave.loadDatasets(sc, sqlContext)

    val (train, validate, errors) = Script.fit(sqlContext, rawTrain, rawValidate, 15)

  (sqlContext, rawTrain, rawValidate, rawEval, rawSmall, train, validate, errors)

  }


  def fit(sqlContext: SQLContext, trainData: DataFrame, validateData: DataFrame, numBits: Int = 15) = {


    val maxIter = 30
    val bits = 15
    val numBits = bits
    val numFeatures = math.pow(2, numBits).toInt
    val regParam = 0.01
    var hasher = new FeatureHashing(bits)

    val featureGen = new FeatureGeneration(sqlContext, numBits)


    val train = featureGen.featurize(trainData, sqlContext).cache()
    val validate = featureGen.featurize(validateData, sqlContext).cache()
    val model = LogisticRegressionLogLoss.fit(train, maxIter, regParam, numFeatures)
    val errorTrain = df_calcError(model.transform(train))
    val errorValidate = df_calcError(model.transform(validate))
    println(s"[maxIter=${maxIter}} numBits=${numBits}}] Train error: $errorTrain, Validate error: $errorValidate")
//
//
    val eval = featurize(rawEval, sqlContext).cache()
    val small = featurize(rawSmall, sqlContext).cache()
    val model = LogisticRegressionLogLoss.fit(eval, maxIter, regParam, numFeatures)
    val errorEval = df_calcError(model.transform(eval))
    val predsRaw = model.transform(small)
    println(s"[maxIter=${maxIter}} numBits=${numBits}}] Train error: $errorTrain, Validate error: $errorValidate")


    val errors = Seq(30).map { maxIter =>
      val model = LogisticRegressionLogLoss.fit(train, maxIter, regParam, numFeatures)
      val errorTrain = df_calcError(model.transform(train))
      val errorValidate = df_calcError(model.transform(validate))
      println(s"[maxIter=${maxIter}} numBits=${numBits}}] Train error: $errorTrain, Validate error: $errorValidate")
      (maxIter, (errorTrain, errorValidate))
    }

    (train, validate, errors)
  }

  def tryfit(sqlContext: SQLContext, trainData: DataFrame, validateData: DataFrame, numBitsSeq: Seq[Int] = Seq(9, 10, 11), maxIterSeq: Seq[Int] = Seq(20, 40)) = {
    val regParam = 0.01
    val results = numBitsSeq.map { numBits =>
      Seq(false).map { interactions =>
        val featureGen = new FeatureGeneration(sqlContext, numBits, interactions)
        val numFeatures = math.pow(2, numBits).toInt
        val train = featureGen.featurize(trainData, sqlContext).cache()
        val validate = featureGen.featurize(validateData, sqlContext).cache()
        val errors = maxIterSeq.map { maxIter =>
          val model = LogisticRegressionLogLoss.fit(train, maxIter, regParam, numFeatures)
          val errorTrain = df_calcError(model.transform(train))
          val errorValidate = df_calcError(model.transform(validate))
          println(s"[maxIter=${maxIter}} numBits=${numBits}} interactions=${interactions}}] Train error: $errorTrain, Validate error: $errorValidate")
          (numBits, interactions, maxIter, errorTrain, errorValidate)
        }
        train.unpersist()
        validate.unpersist()
        errors
      }
    }
    results
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


  def reprocessData(sc: SparkContext) = {
    Logger.getLogger("amazon.emr.metrics").setLevel(Level.OFF)
    Logger.getLogger("com.amazon.ws.emr").setLevel(Level.WARN)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val (users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests, locations, categories) = LoadSave.loadOrigCached(sqlContext)

    val (evalData, trainData, validateData, smallData) =
      TrainingData.split(sqlContext, users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests, locations, categories)

    val prefix = "BANANA_"
    LoadSave.saveDF(sqlContext, trainData, s"${prefix}TRAIN")
    LoadSave.saveDF(sqlContext, validateData, s"${prefix}VALIDATE")
    LoadSave.saveDF(sqlContext, evalData, s"${prefix}EVAL")
    LoadSave.saveDF(sqlContext, smallData, s"${prefix}SMALL")

    (sqlContext, users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests, locations, categories, evalData, trainData, validateData, smallData)
  }


}
