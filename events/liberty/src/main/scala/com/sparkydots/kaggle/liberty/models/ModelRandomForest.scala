package com.sparkydots.kaggle.liberty.models

import com.sparkydots.kaggle.liberty.dataset.Columns
import com.sparkydots.kaggle.liberty.error.GiniError
import com.sparkydots.kaggle.liberty.features.CategoricalFeatureEncoder
import com.sparkydots.spark.dataframe.ReadWrite
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.{BoostingStrategy, Strategy}
import org.apache.spark.mllib.tree.impurity.{Gini, Variance}
import org.apache.spark.mllib.tree.loss.{LogLoss, SquaredError}
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.collection.JavaConverters._

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils


object ModelRandomForest {

//
//
//   def run(sqlContext: SQLContext, rw: ReadWrite, typedKnown: DataFrame, typedLb: DataFrame): Unit = {
//     import sqlContext.implicits._


 //
//     val universe = typedKnown.unionAll(typedLb).cache()
//
//     val encoders = Columns.predictors.zipWithIndex.
//       map { case (p, i) => (i, new CategoricalFeatureEncoder(universe, p)) }
//
//     val encoderSizes = encoders.map { case (i, e) => (i, e.size) }.toMap
//
//     val bcEncoders = sqlContext.sparkContext.broadcast(encoders)
//
//     val data = typedKnown.map { r =>
//       val id = r.getInt(0)
//       val hazard = r.getInt(1).toDouble
//       val feats = bcEncoders.value.map { case (i, enc) => enc(r.get(i + 2)).toDouble }.toArray
//       LabeledPoint(hazard, Vectors.dense(feats))
//     }
//
//     val classes = (d: Double) => if (d > 10) 2.0 else if(d > 2) 1.0 else 0.0
//     val Array(trainingData, validateData, testData) = data.randomSplit(Array(0.40, 0.35, 0.25), 101L).map(_.repartition(16).map(lp => lp.copy(label = classes(lp.label))).cache())
//
//
//     val numClasses = 3
//     val numTrees = 200
//     val featureSubsetStrategy = "auto"
//     val impurity = "gini"
//     val maxDepth = 6
//     val maxBins = 100
//
//     val model = RandomForest.trainClassifier(trainingData, numClasses, encoderSizes, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, 101)
//
//
//     val labelsAndPredictionsTraining = trainingData.map { point =>
//       val prediction = model.predict(point.features)
//       (point.label, prediction)
//     }
//     val trainErr = labelsAndPredictionsTraining.filter(r => r._1 != r._2).count.toDouble / labelsAndPredictionsTraining.count()
//
//     val labelsAndPredictionsValidate = validateData.map { point =>
//       val prediction = model.predict(point.features)
//       (point.label, prediction)
//     }
//     val trainValidate = labelsAndPredictionsValidate.filter(r => r._1 != r._2).count.toDouble / labelsAndPredictionsValidate.count()
//
//     val labelsAndPredictionsTest = testData.map { point =>
//       val prediction = model.predict(point.features)
//       (point.label, prediction)
//     }
//     val trainTest = labelsAndPredictionsTest.filter(r => r._1 != r._2).count.toDouble / labelsAndPredictionsTest.count()
//
//     val lapTrain = labelsAndPredictionsTraining.toDF("label", "pred")
//     val lapValidate = labelsAndPredictionsValidate.toDF("label", "pred")
//     val lapTest = labelsAndPredictionsTest.toDF("label", "pred")
//
//     val errorTrain = GiniError.error(lapTrain)
//     val errorValidate = GiniError.error(lapValidate)
//     val errorTest = GiniError.error(lapTest)
//
//     println(f"Train: $errorTrain%1.6f Validate: $errorValidate%1.6f Test: $errorTest%1.6f NumTrees: ${model.trees.size}")
//
//   }
//
//
//
//   def run2(sqlContext: SQLContext, rw: ReadWrite, typedKnown: DataFrame, typedLb: DataFrame): Unit = {
//
//     val universe = typedKnown.unionAll(typedLb).cache()
//
//     val encoders = Columns.predictors.zipWithIndex.
//       map { case (p, i) => (i, new CategoricalFeatureEncoder(universe, p)) }
//
//     val encoderSizes = encoders.map { case (i, e) => (i, e.size) }.toMap
//
//     val bcEncoders = sqlContext.sparkContext.broadcast(encoders)
//
//     val data = typedKnown.map { r =>
//       val id = r.getInt(0)
//       val hazard = r.getInt(1).toDouble
//       val feats = bcEncoders.value.map { case (i, enc) => enc(r.get(i + 2)).toDouble }.toArray
//       LabeledPoint(hazard, Vectors.dense(feats))
//     }
//
//     val Array(trainingData, validateData, testData) = data.randomSplit(Array(0.40, 0.35, 0.25), 101L).map(_.repartition(16).cache())
//
//     val categoricalFeaturesInfo = encoderSizes.toList.map(x => (x._1.asInstanceOf[Integer], x._2.asInstanceOf[Integer])).toMap.asJava
//
//
//
//
//
//
//
//     val classes = (d: Double) => if (d > 10) 2.0 else if(d > 2) 1.0 else 0.0
//
//     val tr2 = trainingData.map(lp => lp.copy(label = classes(lp.label)))
//     val va2 = validateData.map(lp => lp.copy(label = classes(lp.label)))
//     val te2 = testData.map(lp => lp.copy(label = classes(lp.label)))
//
//     val treeStrategy2 = new Strategy(algo = Classification, impurity = Gini, maxDepth = 2, numClasses = 3, maxBins = 100, categoricalFeaturesInfo = encoderSizes, checkpointInterval = 1000)
//     val boostingStrategy2 = new BoostingStrategy(treeStrategy2, LogLoss, numIterations = 300, learningRate = 0.001, validationTol = 1e-5)
//     val m2 = new GradientBoostedTrees(boostingStrategy2).run(tr2)
//
//     val labelAndPreds2 = te2.map { point =>
//       val prediction = m2.predict(point.features)
//       (point.label, prediction)
//     }
//
//
//     labelAndPreds2.filter(r => r._1 == 0.0 && r._2 == 0.0).count.toDouble
//     labelAndPreds2.filter(r => r._1 == 1.0 && r._2 == 1.0).count.toDouble
//     labelAndPreds2.filter(r => r._1 == 0.0 && r._2 == 1.0).count.toDouble
//     labelAndPreds2.filter(r => r._1 == 1.0 && r._2 == 0.0).count.toDouble
//
//     val testErr = labelAndPreds2.filter(r => r._1 != r._2).count.toDouble / te2.count()
//     println("Test Error = " + testErr)
//
//   }
//
//



 }
