package com.sparkydots.kaggle.liberty.features

import com.sparkydots.kaggle.liberty.dataset.Columns
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SQLContext}

class OHE(sqlContext: SQLContext, typedKnown: DataFrame, typedLb: DataFrame) extends Serializable {

  val universe = typedKnown.unionAll(typedLb).cache()

  val encoders = Columns.predictors.zipWithIndex.
    map { case (p, i) => (i, new CategoricalFeatureEncoder(universe, p)) }

  val encoderSizes = encoders.map { case (i, e) => (i, e.size) }.toMap

  val bcEncoders = sqlContext.sparkContext.broadcast(encoders)

  def encode(df: DataFrame) = {
    df.map { r =>
      val id = r.getInt(0)
      val hazard = r.getInt(1).toDouble
      var offset = 0
      val feats = bcEncoders.value.map {
        case (i, enc) =>
          val idx = enc(r.get(i + 2)) + offset
          offset = offset + enc.size
          (idx, 1.0)
      }.toArray

      (id, hazard, Vectors.sparse(offset, feats))
    }
  }

  def convert(k: DataFrame, l: DataFrame) = Seq(encode(k), encode(l))

}
