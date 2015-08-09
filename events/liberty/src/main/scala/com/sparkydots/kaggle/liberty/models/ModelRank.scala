package com.sparkydots.kaggle.liberty.models

import com.sparkydots.kaggle.liberty.error.GiniError
import com.sparkydots.kaggle.liberty.features.OHE
import com.sparkydots.spark.dataframe.ReadWrite
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import scala.util.Random
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}


object ModelRank {

   def run(sqlContext: SQLContext, rw: ReadWrite, typedKnown: DataFrame, typedLb: DataFrame): Unit = {

     import sqlContext.implicits._


     val bohe = new OHE(sqlContext, typedKnown, typedLb)

     val Seq(known, lb) = Seq(typedKnown, typedLb).map(bohe.encode(_))

//          val classes = (d: Double) => if (d > 10) 1.0 else 0.0 //if (d > 10) 2.0 else if(d > 2) 1.0 else 0.0
//     val classes = (d: Double) => if (d > 10) 2.0 else if(d > 2) 1.0 else 0.0
//     val classes = (d: Double) => if (d > 2) 1.0 else 0.0
     val classes = (d: Double) => d
//     val classes = (d: Double) => if (d > 10) 3.0 else if(d > 5) 2.0 else if (d > 2) 1.0 else 0.0

     val Array(trainSingles, validateSingles, testSingles) = known.randomSplit(Array(0.50, 0.49, 0.01), 11101L).map(_.cache())

     val Array(train, validate, test, knownLabeledPoints, lbLabeledPoints) = Array(trainSingles, validateSingles, testSingles, known, lb).
       map(_.map { case (id, hz, vs) => LabeledPoint(classes(hz), vs)
     }.cache()) //.map(s => s.map(_._2).cache())



     // ===== Logistic Regression =====

          val lr = new LogisticRegressionWithLBFGS().setNumClasses(2)
          lr.optimizer.setNumIterations(40)
          lr.optimizer.setRegParam(0.0001)

          val model1 = lr.run(train)
          val model2 = lr.run(validate)


          val scoreAndLabelsTrain1 = trainSingles.map { case (id, label, feats) =>
            val prediction = model1.predict(feats)
            (label, prediction)
          }.toDF("label", "pred")

     val scoreAndLabelsValidate1 = validateSingles.map { case (id, label, feats) =>
       val prediction = model1.predict(feats)
       (label, prediction)
     }.toDF("label", "pred")

     val scoreAndLabelsTrain2 = trainSingles.map { case (id, label, feats) =>
       val prediction = model1.predict(feats)
       (label, prediction)
     }.toDF("label", "pred")

     val scoreAndLabelsValidate2 = validateSingles.map { case (id, label, feats) =>
       val prediction = model1.predict(feats)
       (label, prediction)
     }.toDF("label", "pred")



     val errorTrain1 = GiniError.error(scoreAndLabelsTrain1)
     val errorValidate1 = GiniError.error(scoreAndLabelsValidate1)
     val errorTrain2 = GiniError.error(scoreAndLabelsTrain2)
     val errorValidate2 = GiniError.error(scoreAndLabelsValidate2)

     println(s"\n$errorTrain1 -> $errorValidate1\n$errorValidate2 -> $errorTrain2")




    // ========  Linear Regression ======
     val linreg1 = LinearRegressionWithSGD.train(train, 40)
     val linreg2 = LinearRegressionWithSGD.train(validate, 40)

     val linRegScoreAndLabelsTrain1 = trainSingles.map { case (id, label, feats) =>
       val prediction = linreg1.predict(feats)
       (label, prediction)
     }.toDF("label", "pred")

     val linRegScoreAndLabelsValidate1 = validateSingles.map { case (id, label, feats) =>
       val prediction = linreg1.predict(feats)
       (label, prediction)
     }.toDF("label", "pred")

     val linRegScoreAndLabelsTrain2 = trainSingles.map { case (id, label, feats) =>
       val prediction = linreg2.predict(feats)
       (label, prediction)
     }.toDF("label", "pred")

     val linRegScoreAndLabelsValidate2 = validateSingles.map { case (id, label, feats) =>
       val prediction = linreg2.predict(feats)
       (label, prediction)
     }.toDF("label", "pred")



     val linRegErrorTrain1 = GiniError.error(linRegScoreAndLabelsTrain1)
     val linRegErrorValidate1 = GiniError.error(linRegScoreAndLabelsValidate1)
     val linRegErrorTrain2 = GiniError.error(linRegScoreAndLabelsTrain2)
     val linRegErrorValidate2 = GiniError.error(linRegScoreAndLabelsValidate2)

     println(s"\n$linRegErrorTrain1 -> $linRegErrorValidate1\n$linRegErrorTrain2 -> $linRegErrorValidate2")



      // =====  Random Forest Regression  ====
     import org.apache.spark.mllib.tree.RandomForest
     import org.apache.spark.mllib.tree.model.RandomForestModel
     import org.apache.spark.mllib.util.MLUtils

     val numClasses = 2
     val categoricalFeaturesInfo = Map[Int, Int]()
     val numTrees = 400 //200
     val featureSubsetStrategy = "all" // "auto" Let the algorithm choose.
     val impurity = "variance"
     val maxDepth = 11
     val maxBins = 100

     val modelRF1 = RandomForest.trainRegressor(train, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
     val modelRF2 = RandomForest.trainRegressor(validate, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

     val rfScoreAndLabelsTrain1 = trainSingles.map { case (id, label, feats) =>
       val prediction = modelRF1.predict(feats)
       (label, prediction)
     }.toDF("label", "pred")

     val rfScoreAndLabelsValidate1 = validateSingles.map { case (id, label, feats) =>
       val prediction = modelRF1.predict(feats)
       (label, prediction)
     }.toDF("label", "pred")

     val rfScoreAndLabelsTrain2 = trainSingles.map { case (id, label, feats) =>
       val prediction = modelRF2.predict(feats)
       (label, prediction)
     }.toDF("label", "pred")

     val rfScoreAndLabelsValidate2 = validateSingles.map { case (id, label, feats) =>
       val prediction = modelRF2.predict(feats)
       (label, prediction)
     }.toDF("label", "pred")

     val rfErrorTrain1 = GiniError.error(rfScoreAndLabelsTrain1)
     val rfErrorValidate1 = GiniError.error(rfScoreAndLabelsValidate1)
     val rfErrorTrain2 = GiniError.error(rfScoreAndLabelsTrain2)
     val rfErrorValidate2 = GiniError.error(rfScoreAndLabelsValidate2)

     val rfAvgTrain = (rfErrorTrain1 + rfErrorValidate2)/2
     val rfAvgTest = (rfErrorValidate1 + rfErrorTrain2)/2

     println(s"\n$rfErrorTrain1 -> $rfErrorValidate1\n$rfErrorValidate2 -> $rfErrorTrain2")
      println(s"\n$rfAvgTrain -> $rfAvgTest")


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

     rw.writeLibertySubmissionToFile("Id,Hazard", rfSubmTest, "RF4.csv")

   }

  def genpairs(singlesRDD: RDD[(Int, Double, Vector)], times: Int): RDD[LabeledPoint] = {
    val singles = singlesRDD.repartition(1)
    (1 to times).map { t =>
      singles.mapPartitions { valsIter =>
        val vals = valsIter.toList
        Random.shuffle(vals).zip(Random.shuffle(vals)).collect {
          case ((_, l1, v1), (_, l2, v2)) if l1 != l2 =>
            LabeledPoint((math.signum(l1 - l2)+1)/2, Vectors.dense((new BDV(v1.toArray) - new BDV(v2.toArray)).toArray).toSparse)
        }.toIterator
      }.repartition(16)
    }.reduce(_ union _)
  }



}
