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
  //  "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount",
  //  "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
  //  "position", "histctr",
  //  "category", "params", "price", "title", "adImpCount", "adClickCount"
  def featurize(data: DataFrame): RDD[LabeledPoint] = {
    data.map { r =>

      // get source values
      val isClick = r.getInt(0).toDouble
      val os = r.getInt(1)
      val uafam = r.getInt(2)

      val visitCount = r.getLong(3)
      val phoneCount = r.getLong(4)

      val impCount = r.getLong(5)
      val clickCount = r.getLong(6)

      val searchTime = r.getInt(7)
      val searchTime_hour = _hourOfDay(searchTime)

      val searchQuery = r.getString(8)
      val searchLoc = r.getInt(9)
      val searchCat = r.getInt(10)
      val searchParams = r.getSeq[Int](11)

      val loggedIn = r.getInt(12)

      val position = r.getInt(13)
      val histctr = r.getDouble(14)

      val category = r.getInt(15)
      val params = r.getSeq[Int](16)
      val price = r.getDouble(17)

      val title = r.getString(18)

      val adImpCount = r.getLong(19)
      val adClickCount = r.getLong(20)

      val ctr = if (impCount > 0)
        clickCount * 1.0 / impCount
      else
        0.0

      val adCtr = if (adImpCount > 0)
        adClickCount * 1.0 / adImpCount
      else
        0.0

      val hiCtr = if (impCount > 2 && ctr >= 0.049067) 1 else 0
      val hiAdCtr = if (adImpCount > 10 && adCtr >= 0.009775) 1 else 0

      val spammer = if (impCount > 2000 || visitCount > 2000) 1 else 0

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
        //hashValues("PriceHigh", if (price > 0 && price < 21000) 1 else 0) ++
        //hashValues("PriceLow", if (price > 21000) 1 else 0) ++
        //hashValues("Price99", if( (price*100).toInt % 100 == 99) 1 else 0) ++
        hashValues("PriceNo", if (price < 0) 1 else 0)
        //hashValues("HiCTR", hiCtr)
        //hashValues("HiAdCTR", hiAdCtr) ++
        //hashValues("spammer", spammer) ++
        //hashValues("HiVisit", if (visitCount > 240) 1 else 0 ) ++
        //hashValues("HiImp", if(impCount > 197) 1 else 0) ++
        //hashValues("HiPhone", if(phoneCount > 11) 1 else 0)

      val features = combine(featureIndices) ++
        Seq(hashFeatureAmount("HistCTR", math.max(histctr, 0.0))) ++
        sentenceFeatures("title", title) ++
        sentenceFeatures("query", searchQuery) ++
        Seq(hashFeatureAmount("paramMatch", paramOverlap(searchParams, params).toDouble))
      val singleFeatures = combinePairs(features)
      //val finalFeatures = combinePairs(singleFeatures ++ interactions(singleFeatures))

      LabeledPoint(isClick, Vectors.sparse(numFeatures, singleFeatures))
    }
  }

  def interactions(singleFeatures: Seq[(Int, Double)]): Seq[(Int, Double)] = {
    (for {
      a <- singleFeatures
      b <- singleFeatures
    } yield {
      if (a._1 != b._1 && a._2 != 0.0 && b._2 != 0.0) {
        Some(hashFeatureAmount(s"Interact_${a._1}_${b._1}_", a._2 * b._2))
      } else {
        None
      }
    }).flatten
  }

}
