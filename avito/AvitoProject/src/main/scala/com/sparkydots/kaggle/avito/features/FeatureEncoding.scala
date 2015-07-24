package com.sparkydots.kaggle.avito.features

object FeatureEncoding {

  def booleanFeature(value: Boolean): Seq[(Int, Seq[Int])] = if (value) Seq((1, Seq(0))) else Seq((1, Seq()))
  def intFeature(value: Int, blockSize: Int = 1): Seq[(Int, Seq[Int])] = Seq((blockSize, Seq(value)))
  def indicatorFeatures(features: Seq[Int], blockSize: Int): Seq[(Int, Seq[Int])] = Seq((blockSize, features))

  def feature_category(category: Int): Seq[(Int, Int)] = {

          val trueCategory = if (category > 250000) {
            category - 250000 + 60
          } else if (category  == 500001) {
            category - 500000 + 6
          } else {
            category
          }
    Seq((12, trueCategory))
  }

  def dedupeFeatures(features: Seq[(Int, Double)]): Seq[(Int, Double)] = features.groupBy(_._1).mapValues(x => x.map(_._2).sum).toSeq.sortBy(_._1)

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

//          val trueSearchLoc = if (searchLoc > 1250000) {
//            searchLoc - 1250000 + 4666
//          } else if (searchLoc > 1000000) {
//            searchLoc - 1000000 + 4629
//          } else if (searchLoc > 750000) {
//            searchLoc - 750000 + 4592
//          } else if (searchLoc > 250000) {
//            4592
//          } else {
//            searchLoc
//          }
//
//          val trueSearchCat = if (searchCat > 250000) {
//            searchCat - 250000 + 60
//          } else if (searchCat  == 500001) {
//            searchCat - 500000 + 6
//          } else {
//            searchCat
//          }
//
//          val trueCategory = if (category > 250000) {
//            category - 250000 + 60
//          } else if (category  == 500001) {
//            category - 500000 + 6
//          } else {
//            category
//          }

