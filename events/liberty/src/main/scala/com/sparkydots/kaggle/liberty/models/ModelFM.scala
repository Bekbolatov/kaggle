package com.sparkydots.kaggle.liberty.models

import com.github.cloudml.zen.ml.recommendation.FM
import com.sparkydots.kaggle.liberty.error.GiniError
import com.sparkydots.kaggle.liberty.features.OHE
import com.sparkydots.spark.dataframe.ReadWrite
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.storage.StorageLevel

object ModelFM {

  def run(sqlContext: SQLContext, rw: ReadWrite, typedKnown: DataFrame, typedLb: DataFrame): Unit = {
    import sqlContext.implicits._

    def error(labels: DataFrame, preds: DataFrame) = GiniError.error(labels.join(preds, "id").select("label", "pred"))

    val ohe = new OHE(sqlContext, typedKnown, typedLb)
        val classes = (d: Double) => if (d > 11) 1.0 else 0.0
//    val classes = (d: Double) => d

//    val Seq(known, lb) = Seq(typedKnown, typedLb).map(ohe.encodeFFM)
////      map { s =>
////      s.repartition(16).map { case (id, hazard, features) =>
////        (id.toLong, LabeledPoint(classes(hazard), features))
////      }
////    }
//
//    val Array(train, validate, test) = known.randomSplit(Array(0.50, 0.49, 0.01), 11101L).map(_.cache()) //.map(s => s.map(_._2).cache())

//    val Array(trainOrig, validateOrig, testOrig, lbOrig) = Array(train, validate, test, lb).map(rdd => rdd.map(r => (r._1, r._2.label)).toDF("id", "label"))

    //task 0 for Regression, and 1 for Binary Classification
    //val model = FMWithLBFGS.train(train, task = 1, numIterations = 20, numCorrections = 5, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1)
//    trainRegression(input: RDD[(Long, LabeledPoint)],
//      numIterations: Int,
//      stepSize: Double,
//      l2: (Double, Double, Double),
//      rank: Int,
//      useAdaGrad: Boolean = {},
//    miniBatchFraction: Double = {},
//    storageLevel: StorageLevel = {}): FMModel

//    val model =  FM.trainRegression(train, numIterations = 39, stepSize = 0.1, (0, 0, 0), 3, true, 1.0, StorageLevel.MEMORY_ONLY)

    //[val model =  FM.trainClassification(train, numIterations = 39, stepSize = 0.1, (0, 0, 0), 3, true, 1.0, StorageLevel.MEMORY_ONLY)]   d>11 0.46564795962944233, 0.43790897788434036
//    val model =  FM.trainClassification(train, numIterations = 80, stepSize = 0.1, (0, 0, 0), 7, true, 1.0, StorageLevel.MEMORY_ONLY)
//
//    println(error(trainOrig, model.predict(train.map(x => (x._1, x._2.features))).toDF("id", "pred")))
//    println(error(validateOrig, model.predict(validate.map(x => (x._1, x._2.features))).toDF("id", "pred")))
//    println(error(testOrig, model.predict(test.map(x => (x._1, x._2.features))).toDF("id", "pred")))
//
//
//
//
//    val dds = known.map(x => (x._2.label, x._2.label)).toDF("label", "pred")
//    val submissionModel =  FM.trainClassification(known, numIterations = 80, stepSize = 0.1, (0, 0, 0), 7, true, 1.0, StorageLevel.MEMORY_ONLY)
//    println(error(trainOrig, submissionModel.predict(train.map(x => (x._1, x._2.features))).map(x => (x._1, (x._2*1000).toInt.toDouble)).toDF("id", "pred")))
//    val submPreds = submissionModel.predict(lb.map(x => (x._1, x._2.features))).map(x => (x._1, (x._2*1000).toInt.toDouble)).toDF("id", "pred")
//    rw.writeLibertySubmissionToFile("Id,Hazard", submPreds, "submission02.csv")

    //    val errorTrain = GiniError.error(lapTrain)
//    val errorValidate = GiniError.error(lapValidate)
//    val errorTest = GiniError.error(lapTest)
//
//    println(f"Train: $errorTrain%1.6f Validate: $errorValidate%1.6f Test: $errorTest%1.6f")


//    def check(iters: Seq[Int], steps: Seq[Double], regParams: Seq[Double]) = {
//      for {
//        numIters <- iters
//        stepSize <- steps
//        regParam <- regParams
//      } yield {
//      val model =  FM.trainRegression(train, numIterations = numIters, stepSize = stepSize, (regParam, regParam, regParam), 3, true, 1.0, StorageLevel.MEMORY_ONLY)
//
//      val etr = error(trainOrig, model.predict(train.map(x => (x._1, x._2.features))).toDF("id", "pred"))
//      val ete = error(validateOrig, model.predict(validate.map(x => (x._1, x._2.features))).toDF("id", "pred"))
//        println(s"${(numIters, stepSize, regParam, etr, ete)}")
//        (numIters, stepSize, regParam, etr, ete)
//      }
//    }
//
//    val cheks = check((36 to 39 by 1).toSeq, Seq(0.09, 0.10, 0.12), Seq(0.0))
//    cheks.foreach(println)
//
//
//
  }


}
