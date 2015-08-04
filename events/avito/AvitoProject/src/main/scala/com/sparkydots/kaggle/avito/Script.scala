package com.sparkydots.kaggle.avito

import java.io.FileWriter

import com.sparkydots.kaggle.avito.features.{SelectFeatures, FeatureHashing, FeatureGeneration}
import com.sparkydots.kaggle.avito.functions.DFFunctions._
import com.sparkydots.kaggle.avito.load.{LoadSave, TrainingData}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
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
*/
  /*
    import scala.util.Try
    import java.io.FileWriter
    import org.apache.log4j.{Level, Logger}
    import org.apache.spark.rdd.RDD
    import org.apache.spark.ml.param.ParamMap
    import org.apache.spark.sql.{Row, SQLContext, DataFrame}
    import org.apache.spark.mllib.linalg.{Vectors, Vector}
    import org.apache.spark.mllib.feature.PCA
    import org.apache.spark.ml.classification.LogisticRegression
    import org.apache.spark.mllib.regression.LabeledPoint
    import com.sparkydots.kaggle.avito._
    import com.sparkydots.kaggle.avito.load._
    import com.sparkydots.kaggle.avito.features._
    import com.sparkydots.kaggle.avito.functions.DFFunctions._
    import com.sparkydots.kaggle.avito.functions.Functions._

    Logger.getLogger("amazon.emr.metrics").setLevel(Level.OFF)
    Logger.getLogger("com.amazon.ws.emr").setLevel(Level.WARN)
    Logger.getLogger("com.amazonaws").setLevel(Level.WARN)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._


    val (rawTrain1, rawValidate1, rawEval1, rawSmall1) = LoadSave.loadDatasets(sc, sqlContext, "CARBON_")

    val (rawTrain, rawValidate, rawEval, rawSmall) = QueryAd.addQueryTitleAffinity(sqlContext, rawTrain1, rawValidate1, rawEval1, rawSmall1)

    rawTrain.cache()
    rawValidate.cache()
    rawEval.cache()
    rawSmall.cache()

    LoadSave.saveDF(sqlContext, rawTrain, "DUMBO_TRAIN")
    LoadSave.saveDF(sqlContext, rawValidate, "DUMBO_VALIDATE")
    LoadSave.saveDF(sqlContext, rawEval, "DUMBO_EVAL")
    LoadSave.saveDF(sqlContext, rawSmall, "DUMBO_SMALL")


    val maxIter = 100
    val regParam = 0.001
    //val elasticNetParam = 0.5
    val words = "words1350"  //900,...,1500  1250, 1350, 1450
    val words2 = None

    val featureGen = new FeatureGeneration(sqlContext, words, words2)
    val train = featureGen.featurize(rawTrain, sqlContext).cache()
    val validate = featureGen.featurize(rawValidate, sqlContext).cache()

    val lr = new LogisticRegression()
    lr.setMaxIter(maxIter).setRegParam(regParam) //.setElasticNetParam(1)


    val train = featurize(rawTrain, sqlContext).cache()
    val validate = featurize(rawValidate, sqlContext).cache()

    val model = lr.fit(train)

    val errorTrain = df_calcError(model.transform(train)
      .select("label", "probability")
      .map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

    val errorValidate = df_calcError(model.transform(validate)
      .select("label", "probability")
      .map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

    println(f"[${maxIter} ${regParam} ${words} $words2] Train error: $errorTrain%1.7f, Validate error: $errorValidate%1.7f")





    Script.saveSubmission(sqlContext, rawEval, rawSmall, "tryTue1", maxIter, regParam, words, words2)



    val train = featurize(rawTrain, sqlContext).cache()
    val validate = featurize(rawValidate, sqlContext).cache()




// For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
 //  * For 0 < alpha < 1, the penalty is a combination of L1 and L2.



  Script.saveSubmission(sqlContext, rawEval, rawSmall, "tryMon1", maxIter, regParam, words, words2)






    val maxIter = 40
    val regParam = 0.003
//    val words = "onlyWords1000"
    val words = "words1000"
    val words2 = None





WordsProcessing.generateAndSaveWordDictionaries(sc, sqlContext, rawEval, rawSmall, "words", Seq(900, 1000, 1500, 2000, 3000))

WordsProcessing.generateAndSaveWordDictionaries(sc, sqlContext, rawEval, rawSmall, "words", Seq(1400, 1450))

    val words = "words1300"
    val words2 = None







    val maxIter = 40
    val regParam = 0.003
    val words = "sloboWords1000"
val (train, validate, lr, featureGen) =  Script.fit(sqlContext, rawTrain, rawValidate, maxIter, regParam, words)
train.show(2)

    Script.run(sc, sqlContext, runs = 100, "stom", 500, 20, 10000)
//    Script.run(sc, sqlContext, runs = 100, "panang", 500, 20, 10000)
//    Script.run(sc, sqlContext, runs = 100, "rain", 600, 50, 5000)
//    Script.run(sc, sqlContext, runs = 100, "zoo", 600, 10000, 6000)




val (train, validate, lr, featureGen) =  Script.fit(sqlContext, reducedRawTain, rawValidate, maxIter, regParam, words)


//s (2.1 GB) is bigger than spark.driver.maxResultSize (1024.0 MB)


//    val (train, validate, lr, featureGen) =  Script.fit(sqlContext, rawTrain, rawValidate, maxIter, regParam, words)


  //WordsProcessing.generateAndSaveWordDictionaries(sc, sqlContext, rawEval, rawSmall, "words", Seq(50))
    val words = "words50"


    Script.saveSubmission(sqlContext, rawEval, rawSmall, featureGen, "tryMon1", maxIter, regParam, words)



saveSubmission(sqlContext: SQLContext, rawEval: DataFrame, rawSmall: DataFrame, featureGen: FeatureGeneration, filename: String, maxIter: Int, regParam: Double, words: String) = {
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


val (sqlContext, users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, nonCtxAdImpressions, ctxAdImpressionsToFind, nonCtxAdImpressionsToFind, visits, phoneRequests, locations, categories, evalData, trainData, validateData, smallData) = TrainingData.reprocessData(sc, "DUMB_")

WordsProcessing.generateAndSaveWordDictionaries(sc, sqlContext, rawEval, rawSmall, "sloboWords", Seq(900, 1000, 1500, 2000, 3000))

////\\\\
val (sqlContext, users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests, locations, categories, evalData, trainData, validateData, smallData) = Script.reprocessData(sc, "CANOPY_")
\\\\\\\///
    val (users, ads, ctxAds, nonCtxAds, searches,
    ctxAdImpressions, nonCtxAdImpressions, ctxAdImpressionsToFind, nonCtxAdImpressionsToFind,
    visits, phoneRequests, locations, categories,
    evalData, trainData, validateData, smallData) = LoadSave.reprocessData(sc, sqlContext, "CARBON_", false)

reprocessData(sc: SparkContext, sqlContext: SQLContext, prefix: String, orig: Boolean = false)







    val bits = 15
    val numFeatures = math.pow(2, bits).toInt




    val (rawTrain, rawValidate, rawEval, rawSmall) = Script.run(sc, sqlContext)

    val (train, validate) = Script.fit(sqlContext, rawTrain, rawValidate, bits, maxIter, regParam, numFeatures)


    results.foreach { case Seq(Seq( (numBits, interactions, maxIter, errorTrain, errorValidate))) =>
          println(s"${numBits}\t${maxIter}\t${errorTrain}\t${errorValidate}")
    }


*/

  def run(sc: SparkContext, sqlContext: SQLContext, runs: Int = 3, filename: String = "trySun4", kept: Int = 300, remove: Int = 6000, add: Int = 2500) = {

    import sqlContext.implicits._

    //
    //val (kept, remove, add)= (600, 10000, 6000)
    //val filename = "temp"
    //val runs = 10
    val (rawTrain, rawValidate, rawEval, rawSmall) = LoadSave.loadDatasets(sc, sqlContext, "CARBON_")

    val maxIter = 40
    val regParam = 0.003
    val words = "onlyWords1000"
    val words2 = None

    val featureGen = new FeatureGeneration(sqlContext, words, words2)

    val trainBefore = featureGen.featurize(rawTrain, sqlContext).cache()
    val validateBefore = featureGen.featurize(rawValidate, sqlContext).cache()
    val evalBefore = featureGen.featurize(rawEval, sqlContext).cache()
    val smallBefore = featureGen.featurize(rawSmall, sqlContext).cache()

    val numFeatures = trainBefore.map(_.getAs[Vector](1).size).first()

    val sf = new SelectFeatures(numFeatures, kept, remove, add, Some(s"featureSel_$filename"))

    val lr = new LogisticRegression()
    lr.setMaxIter(maxIter).setRegParam(regParam)

    var summary = ""
    (1 to runs).foreach { i =>

      sf.rejiggle()
      val train = sf.transform(sqlContext, trainBefore).cache()

      val model = lr.fit(train)

      val errorTrain = df_calcError(model.transform(train)
        .select("label", "probability")
        .map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)

      if (errorTrain < sf.bestValidateError) {
        val validate = sf.transform(sqlContext, validateBefore)
        validate.cache()
        val errorValidate = df_calcError(model.transform(validate)
          .select("label", "probability")
          .map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
        validate.unpersist()

        val betterFound = sf.report(errorTrain, errorValidate)
        val logLine = s"[${sf.transformId}}]*** Train error: $errorTrain%1.8f, Validate error: $errorValidate%1.8f ***"
        println(logLine)
        summary = summary + logLine + "\n"
        writeToFile("run.log", logLine + "\n" )
        if (betterFound) {
          sf.setBestTransform()
          val eval = sf.transform(sqlContext, evalBefore)
          val small = sf.transform(sqlContext, smallBefore)
          eval.cache()
          val model = lr.fit(eval)
          val errorEval = df_calcError(model.transform(eval)
            .select("label", "probability")
            .map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF)
          println(s"[${sf.bestTransformId}]*** errorEval: $errorEval%1.8f ***")
          summary = summary + s"\n*** errorEval: $errorEval ***\n\n"
          writeToFile("run.log", summary )
          eval.unpersist()
          val predsRaw = model.transform(small).select("label", "probability").
            groupBy("label").agg(first("probability")).
            map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).toDF("probability", "label")

          val preds = predsRaw.orderBy("label").map({ case Row(p: Double, l: Double) => (l.toInt, p) }).collect
          val sub = new FileWriter(s"/home/hadoop/${filename}_fs${sf.bestTransformId}.csv", true)
          sub.write("ID,IsClick\n")
          println("saving file...")
          preds.foreach { case (id, prob) =>
            sub.write(id + "," + f"$prob%1.8f" + "\n")
          }
          sub.close()
        }
      } else {
        sf.report(errorTrain, 100.0)
        val logLine = s"[${sf.bestTransformId}]*** Train error: $errorTrain%1.8f [skipped] ***"
        println(logLine)
        summary = summary + logLine + "\n"
        writeToFile("run.log", logLine + "\n" )
      }
      train.unpersist()
    }

    println(summary)

  }

  def writeToFile(filename: String, string: String) = {
    val sub = new FileWriter(s"/home/hadoop/${filename}", true)
    sub.write(string)
    sub.close()
  }

  def fit(sqlContext: SQLContext, rawTrain: DataFrame, rawValidate: DataFrame, maxIter: Int, regParam: Double, words: String = "words20000", wordDictFileNei: Option[String]) = {
    import sqlContext.implicits._

    val featureGen = new FeatureGeneration(sqlContext, words, wordDictFileNei)
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

    println(s"[maxIter=${maxIter} regParam=${regParam} words=${words} words2=$wordDictFileNei] Train error: $errorTrain, Validate error: $errorValidate")
    (train, validate, lr, featureGen)
  }

  def saveSubmission(sqlContext: SQLContext, rawEval: DataFrame, rawSmall: DataFrame, filename: String, maxIter: Int, regParam: Double, words: String, words2: Option[String]) = {
    import sqlContext.implicits._
    import org.apache.spark.sql.functions._


    val featureGen = new FeatureGeneration(sqlContext, words, words2)
    val eval = featureGen.featurize(rawEval, sqlContext).cache()
    val small = featureGen.featurize(rawSmall, sqlContext).cache()

    val lr = new LogisticRegression()
    lr.setMaxIter(maxIter).setRegParam(regParam)

    val model = lr.fit(eval)

    val errorEval = df_calcError(model.transform(eval)
      .select("label", "probability")
      .map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).
      toDF("probability", "label")
    )

    println(s"[maxIter=${maxIter} regParam=${regParam} words=${words} words2=$words2] errorEval: $errorEval")

    val predsRaw1 = model.transform(small).select("label", "probability").
      groupBy("label").agg(first("probability")).
      map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).
      toDF("probability", "label")

    val predsRaw = predsRaw1.withColumn("adjprob", predsRaw1("probability")).select("probability", "label")

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

  def saveSubmissionWithAdj(sqlContext: SQLContext, adj: Double, rawEval: DataFrame, rawSmall: DataFrame, filename: String, maxIter: Int, regParam: Double, words: String, words2: Option[String]) = {
    import sqlContext.implicits._
    import org.apache.spark.sql.functions._


    val featureGen = new FeatureGeneration(sqlContext, words, words2)
    val eval = featureGen.featurize(rawEval, sqlContext).cache()
    val small = featureGen.featurize(rawSmall, sqlContext).cache()

    val lr = new LogisticRegression()
    lr.setMaxIter(maxIter).setRegParam(regParam)

    val model = lr.fit(eval)

    def f(a: Double) = 1.0 / (1.0 + math.exp(-a))
    def fi(a: Double) = -math.log(1.0 / a - 1.0)
    def G(t: Double) = (p: Double) => f( fi(p) + t)


    val g = udf[Double, Double](  G(adj) )

    val errorEval = df_calcError(model.transform(eval)
      .select("label", "probability")
      .map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).
      toDF("probability", "label").
      withColumn("adjprob", g(col("probability"))).
      select("adjprob", "label")
    )

    println(s"[maxIter=${maxIter} regParam=${regParam} words=${words} words2=$words2] errorEval: $errorEval")

    val predsRaw1 = model.transform(small).select("label", "probability").
      groupBy("label").agg(first("probability")).
      map(x => (x.getAs[org.apache.spark.mllib.linalg.DenseVector](1)(1), x.getDouble(0))).
      toDF("probability", "label")

    val predsRaw = predsRaw1.withColumn("adjprob", g(predsRaw1("probability"))).select("probability", "label")

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

    val paramMap = ParamMap(lr.maxIter -> 20)
    paramMap.put(lr.maxIter, 30) // Specify 1 Param.  This overwrites the original maxIter.
    paramMap.put(lr.regParam -> 0.1, lr.threshold -> 0.55) // Specify multiple Params.
    val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability") // Change output column name
    val paramMapCombined = paramMap ++ paramMap2


    val model2 = lr.fit(training.toDF, paramMapCombined) //learn a new model using the paramMapCombined parameters

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
