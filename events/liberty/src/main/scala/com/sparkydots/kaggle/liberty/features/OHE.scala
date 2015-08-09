package com.sparkydots.kaggle.liberty.features

import com.sparkydots.kaggle.liberty.dataset.Columns
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SQLContext}

class OHE(sqlContext: SQLContext, typedKnown: DataFrame, typedLb: DataFrame) extends Serializable {

  val universe = typedKnown.unionAll(typedLb).cache()

  lazy val encodersOHE = Columns.predictors.zipWithIndex.map {
    case (p, i) if Columns.intPredictors.contains(p) => (i, new IntFeatureEncoder)
    case (p, i) if Columns.letterPredictors.contains(p) => (i, new CategoricalFeatureOHEEncoder(universe, p))
  }

  lazy val encoders = Columns.predictors.zipWithIndex.map {
    case (p, i) if Columns.intPredictors.contains(p) => (i, new IntFeatureEncoder)
    case (p, i) if Columns.letterPredictors.contains(p) => (i, new CategoricalFeatureEncoder(universe, p))
  }

  lazy val encoderSizesOHE = encodersOHE.map { case (i, e) => (i, e.size) }.toMap
  lazy val encoderSizes = encoders.map { case (i, e) => (i, e.size) }.toMap

  lazy val bcEncodersOHE = sqlContext.sparkContext.broadcast(encodersOHE)
  lazy val bcEncoders = sqlContext.sparkContext.broadcast(encoders)

  def encode(df: DataFrame, ohe: Boolean = true, otherBcEnc: Option[Broadcast[Seq[(Int, FeatureEncoder[_ >: Int with String])]]] = None) = {
    val bcEnc = otherBcEnc.getOrElse(if (ohe) bcEncodersOHE else bcEncoders)
    df.map { r =>
      val id = r.getInt(0)
      val hazard = r.getInt(1).toDouble
      var offset = 0
      val feats = bcEnc.value.map { case (i, encoder) =>
        val (localIdx, value) = encoder match {
          case enc: IntFeatureEncoder => enc(r.getInt(i + 2))
          case enc: CategoricalFeatureEncoder => enc(r.getString(i + 2))
          case enc: CategoricalFeatureOHEEncoder => enc(r.getString(i + 2))
        }
        val idx = localIdx + offset
        offset = offset + encoder.size
        (idx, value)
      }.toArray

      (id, hazard, Vectors.sparse(offset, feats))
    }
  }

  def categoricalFeaturesInfo = {
    var offset = 0
    encoders.flatMap {
      case (i, e: CategoricalFeatureEncoder) =>
        val idx = offset
        offset = offset + e.size
        Some((idx, e.vocabSize))
      case (i, e: IntFeatureEncoder) =>
        offset = offset + e.size
        None
    }.toMap
  }

}
