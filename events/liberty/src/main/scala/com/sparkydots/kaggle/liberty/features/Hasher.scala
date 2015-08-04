package com.sparkydots.kaggle.liberty.features

import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.util.hashing.MurmurHash3

class Hasher(bits: Int = 15) extends Serializable {

  var numBuckets: Int = math.pow(2, bits).toInt

  def hashString(item: String, blockSize: Int = numBuckets, offset: Int = 0): Int = math.abs(MurmurHash3.stringHash(item)) % blockSize + offset

  def dedupeFeatures(features: Seq[(Int, Double)]): Seq[(Int, Double)] = features.groupBy(_._1).mapValues(x => x.map(_._2).sum).toSeq.sortBy(_._1)

  def hashFeatures(data: RDD[LabeledPoint]) = {
    data.map { case LabeledPoint(label, features) =>
      LabeledPoint(label, hashFeaturesVector(features))
    }
  }

  def hashFeaturesWithInteractions(data: RDD[LabeledPoint]) = {
    data.map { case LabeledPoint(label, features) =>
      LabeledPoint(label, hashFeaturesVectorWithInteractions(features))
    }
  }

  def secondOrderInteractions(one: Seq[(Int, Double)], other: Seq[(Int, Double)]) = {
    for {
      a <- one
      b <- other
      if a._1 != b._1 && a._2 != 0.0 && b._2 != 0.0
    } yield {
      (hashString(s"i${a._1.toString}_${b._1.toString}"), a._2 * b._2)
    }
  }

  def hashFeaturesVector(origFeatures: Vector) = {
    val sv = origFeatures.toSparse
    val indices = sv.indices
    val values = sv.values

    val newFeatures = dedupeFeatures(indices.zip(values).
      map { case (idx, v) => (hashString("f" + idx.toString), v) }).toArray

    Vectors.sparse(numBuckets, newFeatures)
  }

  def hashFeaturesVectorWithInteractions(origFeatures: Vector) = {
    val sv = origFeatures.toSparse
    val indices = sv.indices
    val values = sv.values

    val hashedFeatures = indices.zip(values).map { case (idx, v) =>
      (hashString("f" + idx.toString), v)
    }
    val interactions = secondOrderInteractions(hashedFeatures, hashedFeatures)
    val newFeatures = dedupeFeatures(hashedFeatures ++ interactions).toArray

    Vectors.sparse(numBuckets, newFeatures)
  }

}
