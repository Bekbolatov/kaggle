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

  /**
   * We know inputs are sorted asc, and non-repeating
   * @param first
   * @param second
   * @return
   */
  def numberOfCommonElements(first: Seq[Int], second: Seq[Int]): Int = {
    if (first.isEmpty || second.isEmpty || first.sliding(2).filter(_.size > 1).exists(s => s(0) >= s(1)) || first.sliding(2).filter(_.size > 1).exists(s => s(0) >= s(1))) {
      0
    } else {
      val it1 = first.iterator
      val it2 = second.iterator
      var c1 = it1.next
      var c2 = it2.next
      var count = 0
      while (it1.hasNext && it2.hasNext) {
        if (c1 < c2) {
          c1 = it1.next
        } else if (c1 > c2) {
          c2 = it2.next
        } else {
          count = count + 1
          c1 = it1.next
          c2 = it2.next
        }
      }
      if (c1 == c2) count = count + 1
      count
    }
  }
}
