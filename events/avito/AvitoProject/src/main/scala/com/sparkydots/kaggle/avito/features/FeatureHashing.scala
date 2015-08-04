package com.sparkydots.kaggle.avito.features

import scala.util.hashing.MurmurHash3

class FeatureHashing(bits: Int) extends Serializable {
  var numBuckets: Int = math.pow(2, bits).toInt
  def hashString(blockSize: Int, offset: Int, item: String): Int = math.abs(MurmurHash3.stringHash(item)) % blockSize + offset

  def createFeature(blockSize: Int, offset: Int, feature: String): Int = hashString(blockSize, offset, feature)
  def createFeature(blockSize: Int, offset: Int, feature: String, value: String): Int = hashString(blockSize, offset, feature + value)
  def createFeatures_String(blockSize: Int, offset: Int, feature: String, values: String*): Seq[Int] = values.map(w => createFeature(blockSize, offset, feature, w))
  def createFeatures_Int(blockSize: Int, offset: Int, feature: String, values: Int*): Seq[Int] = values.map(w => createFeature(blockSize, offset, feature, w.toString))

  def setFeaturesValue(features: Seq[Int], value: Double): Seq[(Int, Double)] = features.map(f => (f, value))
  def hashAndSetFeatureValue(blockSize: Int, offset: Int, feature: String, amount: Double): (Int, Double) = (createFeature(blockSize, offset, feature), amount)

  def dedupeFeatures(features: Seq[(Int, Double)]): Seq[(Int, Double)] = features.groupBy(_._1).mapValues(x => x.map(_._2).sum).toSeq.sortBy(_._1)

  def sentenceFeatures(blockSize: Int, offset: Int, feature: String, sentence: String): Seq[(Int, Double)] = {
    val wordFeatures = sentence.split(" ").map(_.trim).filter(_.nonEmpty).map(createFeature(blockSize, offset, feature, _))
    dedupeFeatures(setFeaturesValue(wordFeatures, 1.0))
  }

  def intsFeatures(blockSize: Int, offset: Int, feature: String, ints: Seq[Int]): Seq[(Int, Double)] = {
    val items = ints.map(x => createFeature(blockSize, offset, feature, x.toString))
    dedupeFeatures(setFeaturesValue(items, 1.0))
  }

}
