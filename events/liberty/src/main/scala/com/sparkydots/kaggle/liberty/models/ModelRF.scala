package com.sparkydots.kaggle.liberty.models

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import com.sparkydots.kaggle.liberty.error.GiniError
import com.sparkydots.kaggle.liberty.features.OHE
import com.sparkydots.spark.dataframe.ReadWrite
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impurity.{Impurities, Variance}
import org.apache.spark.mllib.tree.loss.SquaredError
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}
import com.sparkydots.util.TimeTracker

object ModelRF {

   def run(sqlContext: SQLContext, rw: ReadWrite, typedKnown: DataFrame, typedLb: DataFrame): Unit = {

     val cores = 64
     val useOHE = false
     val hazardConv = (a: Double) =>   math.log(a.toDouble)*33  //version used in 4K_15
//     val hazardConvUDF = udf[Double, Int](     (a: Int) =>   (math.log(a.toDouble)*33).toInt )
//     val hazardConv2 = (a: Double) =>   math.log(a.toDouble + 5)*60 - 92

     import sqlContext.implicits._


     val timer1 = new TimeTracker()
     timer1.start("totalprep")

     val bohe = new OHE(sqlContext, typedKnown, typedLb)

     val Seq(known, lb) = Seq(typedKnown, typedLb).map(p => bohe.encode(p, ohe = useOHE).cache())  //OHE or not

//     val Array(trainSingles, validateSingles, testSingles) = known.randomSplit(Array(0.20, 0.10, 0.70), 11101L).map(_.cache())
     val Array(trainSingles, validateSingles, testSingles) = known.randomSplit(Array(0.50, 0.49, 0.01), 11101L).map(_.sortBy(_._2, false, cores).cache())

     val Array(train, validate, test, knownLabeledPoints, lbLabeledPoints) = Array(trainSingles, validateSingles, testSingles, known, lb).
       map(_.map { case (id, hz, vs) => LabeledPoint(hazardConv(hz), vs) }.cache())

     timer1.stop("totalprep")
     println(s"$timer1")

      // =====  Random Forest Regression  ====

     val categoricalFeaturesInfo = if (useOHE) Map[Int, Int]() else bohe.categoricalFeaturesInfo
     val featureSubsetStrategy = "onethird" // "sqrt" //"onethird" // "all" // "auto" Let the algorithm choose.
     val numTrees = 4000

     val strategy = new Strategy(Regression, Variance,
       maxDepth = 15, 0, maxBins = 32, Sort, categoricalFeaturesInfo = categoricalFeaturesInfo,
       minInstancesPerNode = 5,
       minInfoGain = 0,
       maxMemoryInMB = 256,
       subsamplingRate = 1,
       useNodeIdCache = true, checkpointInterval = 10)
//
//     val timer = new TimeTracker()
//     timer.start("total")
//
//     timer.start("rf1")
//     val modelRF1 = RandomForest.trainRegressor(train, strategy, numTrees, featureSubsetStrategy, 101)
//     val bcModelRF1 = sqlContext.sparkContext.broadcast(modelRF1)
//     timer.stop("rf1")
//
//     timer.start("rf2")
//     val modelRF2 = RandomForest.trainRegressor(validate, strategy, numTrees, featureSubsetStrategy, 101)
//     val bcModelRF2 = sqlContext.sparkContext.broadcast(modelRF2)
//     timer.stop("rf2")
//
//     val rfScoreAndLabelsValidate1 = validateSingles.map { case (id, label, feats) =>
//       val prediction = bcModelRF1.value.predict(feats)
//       (label, prediction)
//     }.toDF("label", "pred").cache()
//
//     val rfScoreAndLabelsTrain1 = trainSingles.map { case (id, label, feats) =>
//       val prediction = bcModelRF1.value.predict(feats)
//       (label, prediction)
//     }.toDF("label", "pred").cache()
//
//     val rfScoreAndLabelsValidate2 = validateSingles.map { case (id, label, feats) =>
//       val prediction = bcModelRF2.value.predict(feats)
//       (label, prediction)
//     }.toDF("label", "pred").cache()
//
//     val rfScoreAndLabelsTrain2 = trainSingles.map { case (id, label, feats) =>
//       val prediction = bcModelRF2.value.predict(feats)
//       (label, prediction)
//     }.toDF("label", "pred").cache()
//
//     timer.start("ginis")
//     val rfErrorTrain1 = GiniError.error(rfScoreAndLabelsTrain1)
//     val rfErrorValidate1 = GiniError.error(rfScoreAndLabelsValidate1)
//     val rfErrorTrain2 = GiniError.error(rfScoreAndLabelsTrain2)
//     val rfErrorValidate2 = GiniError.error(rfScoreAndLabelsValidate2)
//     timer.stop("ginis")
//
//     timer.stop("total")
//     val rfAvgTrain = (rfErrorTrain1 + rfErrorValidate2)/2
//     val rfAvgTest = (rfErrorValidate1 + rfErrorTrain2)/2
//
//     println(
//       s"""
//          | featsel: ${featureSubsetStrategy} numTrees: $numTrees strategy: $strategy
//          | $rfErrorTrain1 -> $rfErrorValidate1\n$rfErrorValidate2 -> $rfErrorTrain2
//          | $rfAvgTrain -> $rfAvgTest
//          | $timer
//        """.stripMargin)
//
//     bcModelRF1.unpersist()
//     bcModelRF2.unpersist()
//     rfScoreAndLabelsValidate1.unpersist()
//     rfScoreAndLabelsTrain1.unpersist()
//     rfScoreAndLabelsValidate2.unpersist()
//     rfScoreAndLabelsTrain2.unpersist()



     // === submision  ===

     val kn1 = knownLabeledPoints.repartition(cores).cache()

     val modelRFSubm = RandomForest.trainRegressor(kn1, strategy, numTrees, featureSubsetStrategy, 101)
     val bcModelRFSubm = sqlContext.sparkContext.broadcast(modelRFSubm)

     val rfSubmTest = lb.map { case (id, label, feats) =>
       val prediction = bcModelRFSubm.value.predict(feats)
       (id.toLong, prediction)
     }.toDF("id", "pred")

     rw.writeLibertySubmissionToFile("Id,Hazard", rfSubmTest, "RF0809_2.csv")

     modelRFSubm.save(lb.context, s"s3n://sparkydotsdata/kaggle/liberty/apple/RF_4K_15_log")


     val modelRFSubm2 = RandomForest.trainRegressor(kn1, strategy, 5000, featureSubsetStrategy, 101)
     val bcModelRFSubm2 = sqlContext.sparkContext.broadcast(modelRFSubm2)

     val rfSubmTest2 = lb.map { case (id, label, feats) =>
       val prediction = bcModelRFSubm2.value.predict(feats)
       (id.toLong, prediction)
     }.toDF("id", "pred")

     rw.writeLibertySubmissionToFile("Id,Hazard", rfSubmTest, "RF0809_3.csv")

     modelRFSubm.save(lb.context, s"s3n://sparkydotsdata/kaggle/liberty/apple/RF_5K_16_log")



     //
     //     val rfSubmTrain = known.map { case (id, label, feats) =>
     //       val prediction = modelRFSubm.predict(feats)
     //       (label, prediction)
     //     }.toDF("label", "pred")
     //     val rfSubmTrainError = GiniError.error(rfSubmTrain)










     //  ====   GBT   =====

     import org.apache.spark.mllib.tree.GradientBoostedTrees
     import org.apache.spark.mllib.tree.configuration.BoostingStrategy

     val Array(trainForGBT, validateForGBT) = train.randomSplit(Array(0.6, 0.4), 311L).map(_.cache())

     val treeStrategy = new Strategy(algo = Regression, impurity = Variance, maxDepth = 5, categoricalFeaturesInfo = categoricalFeaturesInfo, numClasses = 0) //Strategy.defaultStategy(algo)
     val boostingStrategy = new BoostingStrategy(treeStrategy, SquaredError, numIterations = 3) //BoostingStrategy.defaultParams("Regression")
     val model =  new GradientBoostedTrees(boostingStrategy).runWithValidation(trainForGBT, validateForGBT) //GradientBoostedTrees.train(train, boostingStrategy)


   }

}
