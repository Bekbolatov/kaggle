package com.sparkydots.kaggle.avito.features

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import FeatureHashing._


import com.sparkydots.kaggle.avito.functions.Functions._

object FeatureGeneration {

  val numFeatures = FeatureHashing.numBuckets

  //  "isClick",
  //  "os", "uafam",
  //  "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
  //  "position", "histctr",
  //  "category", "params", "price", "title"
  def featurize(data: DataFrame): RDD[LabeledPoint] = {
    data.map { r =>

      // get source values
      val isClick = r.getInt(0).toDouble
      val os = r.getInt(1)
      val uafam = r.getInt(2)
      val loggedIn = r.getInt(8)

      val searchTime = r.getInt(3)
      val searchTime_hour = _hourOfDay(searchTime)

      val searchQuery = r.getString(4)
      val searchLoc = r.getInt(5)
      val searchCat = r.getInt(6)
      val searchParams = r.getSeq[Int](7)

      val position = r.getInt(9)
      val histctr = r.getDouble(10)

      val category = r.getInt(11)
      val params = r.getSeq[Int](12)
      val price = r.getDouble(13)
      val title = r.getString(14)

      // categorize and hash values into features
      val featureIndices =
        hashValues("OS", os) ++
        hashValues("UA", uafam) ++
        hashValues("LoggedIn", loggedIn) ++
        hashValues("Weekend", _weekend(searchTime)) ++
        hashValues("TimeMorning", _time_morning(searchTime_hour) ) ++
        hashValues("TimeNoon", _time_noon (searchTime_hour) ) ++
        hashValues("TimeAfter", _time_afternoon (searchTime_hour) ) ++
        hashValues("TimeEvening", _time_evening (searchTime_hour) ) ++
        hashValues("TimeLate", _time_late_evening(searchTime_hour) ) ++
        hashValues("TimeNight", _time_night(searchTime_hour) ) ++
        hashValues("QueryLen", _length(searchQuery)) ++
        hashValues("QLoc", searchLoc) ++
        hashValues("QCat", searchCat) ++
        hashValues("QParams", searchParams:_*) ++
        hashValues("Pos", position) ++
        hashValues("Cat", category) ++
        hashValues("adpars", params:_*) ++
        hashValues("titleLen", _length(title)) ++
        hashValues("PriceHigh", _price_high(price)) ++
        hashValues("PriceMedHi", _price_medhigh(price)) ++
        hashValues("PriceMed", _price_med(price)) ++
        hashValues("PriceMedLo", _price_medlow(price)) ++
        hashValues("PriceLow", _price_low(price)) ++
        hashValues("Price99", _price_99(price)) ++
        hashValues("PriceMiss", _price_miss(price))

      val features = combine(featureIndices) ++
        Seq(hashFeatureAmount("HistCTR", math.max(histctr, 0.0))) ++
      sentenceFeatures("title", title) ++
      sentenceFeatures("query", searchQuery) ++
      Seq(hashFeatureAmount("paramMatch", paramOverlap(searchParams, params).toDouble))

      val finalFeatures = combinePairs(features)
      // Later will add user's past behavior (number of searches, views/clicks on ctx ads, visits/phonerequests)

      LabeledPoint(isClick, Vectors.sparse(numFeatures, finalFeatures))
    }
  }

}
