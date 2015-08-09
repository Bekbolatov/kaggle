package com.sparkydots.kaggle.liberty.models

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import com.sparkydots.kaggle.liberty.error.GiniError
import com.sparkydots.kaggle.liberty.features.OHE
import com.sparkydots.spark.dataframe.ReadWrite
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}
import com.sparkydots.util.TimeTracker

object ModelRF {

   def run(sqlContext: SQLContext, rw: ReadWrite, typedKnown: DataFrame, typedLb: DataFrame): Unit = {

     val useOHE = false
     val hazardConvUDF = udf[Double, Int](     (a: Int) =>   (math.log(a.toDouble)*33).toInt )
     val hazardConv = (a: Double) =>   math.log(a.toDouble)*33
     val hazardConv2 = (a: Double) =>   math.log(a.toDouble + 5)*60 - 92

     /*

      val ss = udf[Double, Int](     (a: Int) =>   (math.log(a.toDouble)*33).toInt )

      val ss = udf[Double, Int](     (a: Int) =>   math.log(a.toDouble + 5)*60 - 92 )
     typedKnown.select(ss(col("Hazard")).as("sd")).groupBy("sd").count.orderBy("sd").show

      */
     import sqlContext.implicits._


     val timer1 = new TimeTracker()
     timer1.start("totalprep")

     val bohe = new OHE(sqlContext, typedKnown, typedLb)

     val Seq(known, lb) = Seq(typedKnown, typedLb).map(p => bohe.encode(p, ohe = useOHE))  //OHE or not

     val Array(trainSingles, validateSingles, testSingles) = known.randomSplit(Array(0.50, 0.49, 0.01), 11101L).map(_.cache())

     val Array(train, validate, test, knownLabeledPoints, lbLabeledPoints) = Array(trainSingles, validateSingles, testSingles, known, lb).
       map(_.map { case (id, hz, vs) => LabeledPoint(hazardConv(hz), vs)
     }.repartition(64).cache())

     timer1.stop("totalprep")
     println(s"$timer1")

      // =====  Random Forest Regression  ====

     val numClasses = 2
     val categoricalFeaturesInfo = if (useOHE) Map[Int, Int]() else bohe.categoricalFeaturesInfo
     val impurity = "variance"
     val maxBins = 128 //32 //128
     val featureSubsetStrategy = "onethird" // "sqrt" //sqrt" //"onethird"  "all" // "auto" Let the algorithm choose.

     val numTrees = 400 //400 //200
     val maxDepth = 12 //11

     //

     val timer = new TimeTracker()
     timer.start("total")
     timer.start("rf1")
     val modelRF1 = RandomForest.trainRegressor(train, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
     timer.stop("rf1")
     timer.start("rf2")
     val modelRF2 = RandomForest.trainRegressor(validate, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
     timer.stop("rf2")

     val rfScoreAndLabelsTrain1 = trainSingles.map { case (id, label, feats) =>
       val prediction = modelRF1.predict(feats)
       (label, prediction)
     }.toDF("label", "pred").repartition(64)

     val rfScoreAndLabelsValidate1 = validateSingles.map { case (id, label, feats) =>
       val prediction = modelRF1.predict(feats)
       (label, prediction)
     }.toDF("label", "pred").repartition(64)

     val rfScoreAndLabelsTrain2 = trainSingles.map { case (id, label, feats) =>
       val prediction = modelRF2.predict(feats)
       (label, prediction)
     }.toDF("label", "pred").repartition(64)

     val rfScoreAndLabelsValidate2 = validateSingles.map { case (id, label, feats) =>
       val prediction = modelRF2.predict(feats)
       (label, prediction)
     }.toDF("label", "pred").repartition(64)


     timer.start("ginis")
     val rfErrorTrain1 = GiniError.error(rfScoreAndLabelsTrain1)
     val rfErrorValidate1 = GiniError.error(rfScoreAndLabelsValidate1)
     val rfErrorTrain2 = GiniError.error(rfScoreAndLabelsTrain2)
     val rfErrorValidate2 = GiniError.error(rfScoreAndLabelsValidate2)
     timer.stop("ginis")

     timer.stop("total")
     val rfAvgTrain = (rfErrorTrain1 + rfErrorValidate2)/2
     val rfAvgTest = (rfErrorValidate1 + rfErrorTrain2)/2

     println(
       s"""
          | featsel: ${featureSubsetStrategy} numTrees: $numTrees maxDepth: $maxDepth
          | $rfErrorTrain1 -> $rfErrorValidate1\n$rfErrorValidate2 -> $rfErrorTrain2
          | $rfAvgTrain -> $rfAvgTest
          | $timer
        """.stripMargin)








     // === submision  ===

     val modelRFSubm = RandomForest.trainRegressor(knownLabeledPoints, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

     val rfSubmTrain = known.map { case (id, label, feats) =>
       val prediction = modelRFSubm.predict(feats)
       (label, prediction)
     }.toDF("label", "pred")
     val rfSubmTrainError = GiniError.error(rfSubmTrain)

     val rfSubmTest = lb.map { case (id, label, feats) =>
       val prediction = modelRFSubm.predict(feats)
       (id.toLong, prediction)
     }.toDF("id", "pred")

     rw.writeLibertySubmissionToFile("Id,Hazard", rfSubmTest, "RF0808_1.csv")

   }

}
