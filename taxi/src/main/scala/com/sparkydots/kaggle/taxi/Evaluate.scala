package com.sparkydots.kaggle.taxi

import com.sparkydots.util.Errors._
import org.apache.spark.rdd.RDD

object Evaluate extends Serializable {

  def error(preds: RDD[(Double, Double)]) = math.sqrt(preds.map(diffLogsSq(_)).sum / preds.count)

}
