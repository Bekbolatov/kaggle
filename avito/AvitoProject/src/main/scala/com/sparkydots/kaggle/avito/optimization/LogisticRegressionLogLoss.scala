package com.sparkydots.kaggle.avito.optimization

import com.sparkydots.kaggle.avito.optimization.BLAS.{axpy, dot}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, Row}


class LogisticRegressionLogLoss extends Serializable {

  var optimizer = new LBFGS(new LogisticGradient, new SquaredL2Updater)

  def run(input: RDD[LabeledPoint], numFeatures: Int = 100, initialWeightsMaybe: Option[Vector] = None): LogisticRegressionLogLossModel = {
    val initialWeights = initialWeightsMaybe.getOrElse(Vectors.dense(new Array[Double](numFeatures)))
    val scaler = new StandardScaler(withStd = true, withMean = false).fit(input.map(_.features))
    val data = input.map(lp => (lp.label, scaler.transform(lp.features))).cache()
    var weights = optimizer.optimize(data, initialWeights)
    weights = scaler.transform(weights)

    new LogisticRegressionLogLossModel(weights)
  }

  def fit(dataset: DataFrame, maxIter: Int = 10, regParam: Double = 0.0, numFeatures: Int = 100, initialWeightsMaybe: Option[Vector] = None): LogisticRegressionLogLossModel = {
    val oldDataset = dataset.select("label", "features").map { case Row(label: Double, features: Vector) => LabeledPoint(label, features) }
    optimizer.setRegParam(regParam).setNumIterations(maxIter)
    run(oldDataset, numFeatures, initialWeightsMaybe)
  }


  def setRegL1() = { optimizer.setUpdater(new L1Updater) }
  def setRegL2() = { optimizer.setUpdater(new SquaredL2Updater) }

}

object LogisticRegressionLogLoss {
  def fit(dataset: DataFrame, maxIter: Int = 10, regParam: Double = 0.0, numFeatures: Int = 100): LogisticRegressionLogLossModel = {
    val lr = new LogisticRegressionLogLoss()
    val model = lr.fit(dataset, maxIter, regParam, numFeatures)
    model
  }
}

class LogisticRegressionLogLossModel(val weights: Vector) extends Serializable {
  def col(colName: String): Column = new Column(colName)
  val margin: Vector => Double = (features) => BLAS.dot(features, weights)
  val score: Vector => Double = (features) => 1.0 / (1.0 + math.exp(-margin(features)))
  def transform(dataset: DataFrame): DataFrame = {
    val features2prob = udf { (features: Vector) => score(features): Double }
    dataset.withColumn("probability", features2prob(col("features"))).select("probability", "label")
  }
}
