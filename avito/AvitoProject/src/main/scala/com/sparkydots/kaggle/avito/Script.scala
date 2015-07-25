package com.sparkydots.kaggle.avito

import java.io.FileWriter

import com.sparkydots.kaggle.avito.features.{FeatureHashing, FeatureGeneration}
import com.sparkydots.kaggle.avito.functions.DFFunctions._
import com.sparkydots.kaggle.avito.load.{LoadSave, TrainingData}
import com.sparkydots.kaggle.avito.optimization.{LogisticRegressionLogLossModel, LogisticRegressionLogLoss}

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
    import org.apache.spark.ml.param.ParamMap
    import org.apache.spark.sql.{Row, SQLContext}
    import com.sparkydots.kaggle.avito._
    import com.sparkydots.kaggle.avito.load._
    import com.sparkydots.kaggle.avito.functions.DFFunctions._
    import com.sparkydots.kaggle.avito.functions.Functions._
    import com.sparkydots.kaggle.avito.features.FeatureGeneration
    import com.sparkydots.kaggle.avito.features.FeatureHashing
    import com.sparkydots.kaggle.avito.features.WordsProcessing
    import com.sparkydots.kaggle.avito.optimization.LogisticRegressionLogLoss
    import org.apache.spark.mllib.linalg.Vectors
    import org.apache.spark.mllib.regression.LabeledPoint
    import org.apache.spark.rdd.RDD
    import org.apache.spark.sql.{SQLContext, DataFrame}
    import scala.util.Try
    import java.io.FileWriter

    Logger.getLogger("amazon.emr.metrics").setLevel(Level.OFF)
    Logger.getLogger("com.amazon.ws.emr").setLevel(Level.WARN)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val (rawTrain, rawValidate, rawEval, rawSmall) = LoadSave.loadDatasets(sc, sqlContext)

    val (train, validate, lr, featureGen) =  Script.fit(sqlContext, rawTrain, rawValidate, 30, 0.01)
/////
    val featureGen = new FeatureGeneration(sqlContext, "words1000")
    val train = featureGen.featurize(rawTrain, sqlContext).cache()
    val validate = featureGen.featurize(rawValidate, sqlContext).cache()

/////
    val paramMap = ParamMap(lr.maxIter -> 40, lr.regParam -> 0.0)
    val model = lr.fit(train, paramMap)

    val errorTrain = df_calcError(model.transform(train)
      .select("label", "probability")
      .map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

    val errorValidate = df_calcError(model.transform(validate)
      .select("label", "probability")
      .map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

    println(s"Train error: $errorTrain, Validate error: $errorValidate")
    //Train error: 0.0458574650724705, Validate error: 0.0473141878930015


// Now run against all data set ('eval') and submit

    val eval = featureGen.featurize(rawEval, sqlContext).cache()
    val small = featureGen.featurize(rawSmall, sqlContext).cache()

/////
    val paramMap = ParamMap(lr.maxIter -> 40, lr.regParam -> 0.0)
    val model = lr.fit(eval, paramMap)

    val errorTrain = df_calcError(model.transform(eval)
      .select("label", "probability")
      .map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

///\\\\\
WordsProcessing.generateAndSaveWordDictionaries(sc, sqlContext, rawEval, rawSmall)
////\\\\







    val maxIter = 35
    val bits = 15
    val numFeatures = math.pow(2, bits).toInt
    val regParam = 0.01




    val (rawTrain, rawValidate, rawEval, rawSmall) = Script.run(sc, sqlContext)

    val (train, validate) = Script.fit(sqlContext, rawTrain, rawValidate, bits, maxIter, regParam, numFeatures)


    results.foreach { case Seq(Seq( (numBits, interactions, maxIter, errorTrain, errorValidate))) =>
          println(s"${numBits}\t${maxIter}\t${errorTrain}\t${errorValidate}")
    }


*/

  def fit(sqlContext: SQLContext, rawTrain: DataFrame, rawValidate: DataFrame, maxIter: Int, regParam: Double) = {
    import sqlContext.implicits._

    val featureGen = new FeatureGeneration(sqlContext, "words100")
    val train = featureGen.featurize(rawTrain, sqlContext).cache()
    val validate = featureGen.featurize(rawValidate, sqlContext).cache()

    val lr = new LogisticRegression()
    lr.setMaxIter(maxIter).setRegParam(regParam)

    val model = lr.fit(train)

    val errorTrain = df_calcError(model.transform(train)
      .select("label", "probability")
      .map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

    val errorValidate = df_calcError(model.transform(validate)
      .select("label", "probability")
      .map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

    println(s"[maxIter=${maxIter}}] Train error: $errorTrain, Validate error: $errorValidate")
    (train, validate, lr, featureGen)
  }

  def saveSubmission(sqlContext: SQLContext, rawEval: DataFrame, rawSmall: DataFrame, featureGen: FeatureGeneration, filename: String, bits: Int, maxIter: Int, regParam: Double, numFeatures: Int) = {
    import sqlContext.implicits._

    val eval = featureGen.featurize(rawEval, sqlContext).cache()
    val small = featureGen.featurize(rawSmall, sqlContext).cache()
    val model = LogisticRegressionLogLoss.fit(eval, maxIter, regParam, numFeatures)
    val errorEval = df_calcError(model.transform(eval))

    val predsRaw = model.transform(small).select("label", "probability").
      map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF("probability", "label")

    println(s"[maxIter=${maxIter}} numBits=${bits}} regParam=${regParam}} eval] Eval error: $errorEval")

    val preds = predsRaw.orderBy("label").map({ case Row(p: Double, l: Double) => (l.toInt, p) }).collect
    val sub = new FileWriter(s"/home/hadoop/${filename}.csv", true)
    sub.write("ID,IsClick\n")
    println("saving file...")
    preds.foreach { case (id, prob) =>
      sub.write(id + "," + f"$prob%1.8f" + "\n")
    }
    sub.close()
    println("done")
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
