package com.sparkydots.kaggle.liberty.error

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object GiniError {

  /**
   * ("id", "label"), ("id", "pred")
   */
  def error(labels: DataFrame, preds: DataFrame) = error(labels.join(preds, "id").select("label", "pred"))

  /**
   * (label, pred)
   */
  def error(labelsAndPredictions: DataFrame): Double = {

    val lap = labelsAndPredictions.cache()

    val best = lap.orderBy(col("label").desc).map(_.getDouble(0)).collect()
    val pred = lap.orderBy(col("pred").desc).map(_.getDouble(0)).collect()

    val sumsBest = best.scanLeft(0.0)(_ + _)
    val sumsPred = pred.scanLeft(0.0)(_ + _)

    val total = sumsBest.last

    val modelBest = sumsBest.sum
    val modelPred = sumsPred.sum

    val modelRandom = total * (best.size.toDouble + 1.0) / 2.0

    (modelPred - modelRandom) / (modelBest - modelRandom)
  }
}
