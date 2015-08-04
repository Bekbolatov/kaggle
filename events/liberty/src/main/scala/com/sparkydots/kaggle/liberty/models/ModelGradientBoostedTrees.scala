package com.sparkydots.kaggle.liberty.models

import com.sparkydots.kaggle.liberty.dataset.Columns
import com.sparkydots.kaggle.liberty.error.GiniError
import com.sparkydots.kaggle.liberty.features.CategoricalFeatureOneHotEncoder
import com.sparkydots.spark.dataframe.ReadWrite
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.{Algo, BoostingStrategy, Strategy}
import org.apache.spark.sql.{DataFrame, SQLContext}
import scala.collection.JavaConverters._


object ModelGradientBoostedTrees {



  def run(sqlContext: SQLContext, rw: ReadWrite, typedKnown: DataFrame, typedLb: DataFrame): Unit = {
    import sqlContext.implicits._

    val universe = typedKnown.unionAll(typedLb).cache()

    val encoders = Columns.predictors.zipWithIndex.
      map { case (p, i) => (i, new CategoricalFeatureOneHotEncoder(universe, p)) }

    val encoderSizes = encoders.map { case (i, e) => (i, e.size) }.toMap

    val bcEncoders = sqlContext.sparkContext.broadcast(encoders)

    val data = typedKnown.map { r =>
      val id = r.getInt(0)
      val hazard = r.getInt(1).toDouble
      val feats = bcEncoders.value.map { case (i, enc) => enc(r.get(i + 2)).toDouble }.toArray
      LabeledPoint(hazard, Vectors.dense(feats))
    }

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val categoricalFeaturesInfo = encoderSizes.toList.map(x => (x._1.asInstanceOf[Integer], x._2.asInstanceOf[Integer])).toMap.asJava

    val treeStrategy = Strategy.defaultStategy(Algo.Regression)
    treeStrategy.setMaxDepth(5)
    treeStrategy.setMaxBins(100)
    treeStrategy.setCategoricalFeaturesInfo(categoricalFeaturesInfo)

    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.setNumIterations(3) // Note: Use more iterations in practice.
    boostingStrategy.setTreeStrategy(treeStrategy)

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    val labelsAndPredictionsTraining = trainingData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val labelsAndPredictionsTest = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val lapTrain = labelsAndPredictionsTraining.toDF("label", "pred")
    val lapTest = labelsAndPredictionsTest.toDF("label", "pred")

    val errorTrain = GiniError.error(lapTrain)
    val errorTest = GiniError.error(lapTest)

    println(s"Train: $errorTrain%1.6f Test: $errorTest%1.6f")

  }





}
