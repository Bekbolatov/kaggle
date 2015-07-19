package com.sparkydots.kaggle.avito.features

import org.apache.spark.sql.SQLContext

import scala.math._
import scala.util.hashing.MurmurHash3

object FeatureHashing {

  val numBuckets: Int = math.pow(2, 15).toInt //32,768
  def _hash(item: String): Int = abs(MurmurHash3.stringHash(item)) % numBuckets

  def hashFeature(feature: String, value: String): Int = _hash(feature + value)
  def hashStringValues(feature: String, values: String*): Seq[Int] = values.map(w => _hash(feature + w))
  def hashValues(feature: String, values: Int*): Seq[Int] = values.map(w => _hash(feature + w.toString))
//  def hashValues(feature: String, values: Seq[Int]): Seq[Int] = values.map(w => _hash(feature + w.toString))

  def combine(features: Seq[Int]): Seq[(Int, Double)] = features.map(_ -> 1).groupBy(_._1).mapValues(x => x.map(_._2).sum.toDouble).toSeq.sortBy(_._1)
  def combinePairs(features: Seq[(Int, Double)]): Seq[(Int, Double)] = features.groupBy(_._1).mapValues(x => x.map(_._2).sum).toSeq.sortBy(_._1)

  def hashFeatureAmount(feature: String, amount: Double): (Int, Double) = (_hash(feature), amount)


  def sentenceFeatures(feature: String, sentence: String): Seq[(Int, Double)] =
    combine(sentence.split(" ").map(_.trim).filter(_.nonEmpty).map(hashFeature(feature, _)))

  def intsFeatures(feature: String, ints: Seq[Int]): Seq[(Int, Double)] =
    combine(ints.map(x => hashFeature(feature, x.toString)))


  /**
   * We know inputs are sorted asc, and non-repeating
   * @param first
   * @param second
   * @return
   */
  def paramOverlap(first: Seq[Int], second: Seq[Int]): Int = {
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

  def calcError(sqlContext: SQLContext) = {


    sqlContext.sql("select count(1), avg(errf(s.histctr, s.isClick)) from searchStream s join searchInfo i on (i.id = s.searchId) where s.type = 3 and i.eventTime > 1900800").show()

    sqlContext.sql(
      """
        |select i.user, avg(errf(s.histctr, s.isClick)
        |from searchStream s
        |join searchInfo i on (i.id = s.searchId)
        |where s.type = 3 and i.eventTime > 1900800
        |group by
      """.stripMargin).show()

  }

}
