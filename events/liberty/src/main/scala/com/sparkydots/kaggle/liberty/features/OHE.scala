package com.sparkydots.kaggle.liberty.features

import com.sparkydots.kaggle.liberty.dataset.Columns
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SQLContext}

class OHE(sqlContext: SQLContext, typedKnown: DataFrame, typedLb: DataFrame) extends Serializable {

  val universe = typedKnown.unionAll(typedLb).cache()

  val encoders = Columns.predictors.zipWithIndex.map {
    case (p, i) if Columns.intPredictors.contains(p) => (i, new IntFeatureEncoder)
    case (p, i) if Columns.letterPredictors.contains(p) => (i, new CategoricalFeatureEncoder(universe, p))
  }

  val encoderSizes = encoders.map { case (i, e) => (i, e.size) }.toMap

  val bcEncoders = sqlContext.sparkContext.broadcast(encoders)

  def encode(df: DataFrame) = {
    df.map { r =>
      val id = r.getInt(0)
      val hazard = r.getInt(1).toDouble
      var offset = 0
      val feats = bcEncoders.value.map {
        case (i, enc: CategoricalFeatureEncoder) =>
          val (localIdx, value) = enc(r.getString(i + 2))
          val idx = localIdx + offset
          offset = offset + enc.size
          (idx, value)
        case (i, enc: IntFeatureEncoder) =>
          val (localIdx, value) = enc(r.getInt(i + 2))
          val idx = localIdx + offset
          offset = offset + enc.size
          (idx, value)

      }.toArray

      (id, hazard, Vectors.sparse(offset, feats))
    }
  }

  def convert(k: DataFrame, l: DataFrame) = Seq(encode(k), encode(l))

}
