package com.sparkydots.kaggle.taxi

import com.sparkydots.util.Errors._
import org.apache.spark.rdd.{PairRDDFunctions, RDD}

object Evaluate extends Serializable {


  def means[K](data: RDD[(K, Double)]): Map[K, Double] = {
    data.asInstanceOf[PairRDDFunctions[K, Double]].groupByKey()
      .asInstanceOf[PairRDDFunctions[K, Iterable[Double]]].mapValues { values =>
      val n = values.size
      val sum = values.sum
      sum / n
    }.collect().toMap
  }

  def error(preds: RDD[(Double, Double)]) = math.sqrt(preds.map(p => pointLogPlusOneErrorSquared(p._1, p._2)).sum / preds.count)

}
