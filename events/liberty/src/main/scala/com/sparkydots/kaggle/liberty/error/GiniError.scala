package com.sparkydots.kaggle.liberty.error

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object GiniError {

  /**
   * ("id", "label"), ("id", "pred")
   */
  def error(labels: DataFrame, preds: DataFrame): Double = error(labels.join(preds, "id").select("label", "pred"))

  /**
   * (label, pred)
   */
  def error(lap: DataFrame): Double = {

    val best = lap.map(_.getDouble(0)).collect()
    val pred = lap.orderBy(col("pred").desc).map(_.getDouble(0)).collect()

    val sumsBest = best.scanLeft(0.0)(_ + _)
    val sumsPred = pred.scanLeft(0.0)(_ + _)

    val total = sumsBest.last

    val modelBest = sumsBest.sum
    val modelPred = sumsPred.sum

    val modelRandom = total * (best.size.toDouble + 1.0) / 2.0

    (modelPred - modelRandom) / (modelBest - modelRandom)
  }

  /*
  doesn't preserve order in partitions
   */
  def errorRX(labelsAndPredictions: DataFrame): Double = {
    val lap = labelsAndPredictions.cache()

    val best = lap.orderBy(col("label").desc).map(_.getDouble(0)).repartition(64).cache()
    val pred = lap.orderBy(col("pred").desc).map(_.getDouble(0)).repartition(64).cache()

    val sumsBestTotals = lap.sqlContext.sparkContext.broadcast(0.0 +: best.mapPartitionsWithIndex{ case(partition, iter) => Iterator(iter.sum) }.collect)
    val sumsPredTotals = lap.sqlContext.sparkContext.broadcast(0.0 +: pred.mapPartitionsWithIndex{ case(partition, iter) => Iterator(iter.sum) }.collect)

    val sumsBest = best.mapPartitionsWithIndex { case(partition, iter) =>
      val base = sumsPredTotals.value(partition)
      iter.scanLeft(base)(_ + _)
    }.collect()

    val sumsPred = pred.mapPartitionsWithIndex { case(partition, iter) =>
      val base = sumsPredTotals.value(partition)
      iter.scanLeft(base)(_ + _)
    }.collect()

    val total = sumsBest.last

    val modelBest = sumsBest.sum
    val modelPred = sumsPred.sum

    val modelRandom = total * (sumsBest.size.toDouble + 1.0) / 2.0

    labelsAndPredictions.unpersist()
    best.unpersist()
    pred.unpersist()
    sumsBestTotals.unpersist()
    sumsPredTotals.unpersist()

    (modelPred - modelRandom) / (modelBest - modelRandom)
  }


  /*


  val vector = sc.parallelize(1 to 20, 3)

val sums = 0 +: vector.mapPartitionsWithIndex{ case(partition, iter) => Iterator(iter.sum) }.collect

val prefixScan = vector.mapPartitionsWithIndex { case(partition, iter) =>
  val base = sums(partition)
  iter.scanLeft(base)(_+_).drop(1)
}.collect


   */

}
