package com.sparkydots.kaggle.avito.features

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, DataFrame}

import com.sparkydots.kaggle.avito.functions.Functions._

import scala.util.Try

class FeatureGeneration(sqlContext: SQLContext, bits: Int = 15, addInteractions: Boolean = true) extends Serializable  {
  var numFeatures = math.pow(2, bits).toInt
  var hasher = new FeatureHashing(bits)

  /**
   *   //  "isClick",
  //  "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount",
  //  "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
  //  "position", "histctr",
  //  "category", "params", "price", "title", "adImpCount", "adClickCount"
   // "searchLocLevel", "searchLocPar", "searchCatLevel", "searchCatPar", "adCatLevel", "adCatPar"
   * @param data
   * @param addInteractions
   * @return
   */
  def featurize(data: DataFrame, sqlContext: SQLContext): DataFrame = {
    import sqlContext.implicits._

    data.map { r =>
      val isClick = r.getInt(0).toDouble
      val os = Try(r.getInt(1)).getOrElse(-1)
      val uafam = Try(r.getInt(2)).getOrElse(-1)

      val visitCount = Try(r.getLong(3).toInt).getOrElse(0)
      val phoneCount = Try(r.getLong(4).toInt).getOrElse(0)
      val impCount = Try(r.getLong(5).toInt).getOrElse(0)
      val clickCount = Try(r.getLong(6).toInt).getOrElse(0)

      val searchTime = Try(r.getInt(7)).getOrElse(0)
      val searchTime_hour = hourOfDay(searchTime)
      val searchQuery = Try(r.getString(8).toLowerCase).getOrElse("")
      val searchLoc = Try(r.getInt(9)).getOrElse(-1)
      val searchCat = Try(r.getInt(10)).getOrElse(-1)
      val searchParams = Try ({
        val  l = r.getSeq[Int](11)
        if( l == null) {
          Seq.empty
        } else {
          l
        }
      }).getOrElse(Seq.empty)

      val loggedIn = Try(r.getInt(12)).getOrElse(-1)
      val position = r.getInt(13)
      val histctr = r.getDouble(14)
      val category = Try(r.getInt(15)).getOrElse(-1)
      val params = Try({
        val  l = r.getSeq[Int](16)
        if( l == null) {
          Seq.empty
        } else {
          l
        }
      }).getOrElse(Seq.empty)
      val price = Try(r.getDouble(17)).getOrElse(-1.0)
      val title = Try(r.getString(18).toLowerCase).getOrElse("")

      val adImpCount =  Try(r.getLong(19).toInt).getOrElse(0)
      val adClickCount =  Try(r.getLong(20).toInt).getOrElse(0)

      //"searchLocLevel", "searchLocPar", "searchCatLevel", "searchCatPar", "adCatLevel", "adCatPar"
      val searchLocLevel =  Try(r.getInt(21)).getOrElse(-1)
      val searchLocPar =  Try(r.getInt(22)).getOrElse(-1)
      val searchCatLevel =  Try(r.getInt(23)).getOrElse(-1)
      val searchCatPar =  Try(r.getInt(24)).getOrElse(-1)
      val adCatLevel =  Try(r.getInt(25)).getOrElse(-1)
      val adCatPar =  Try(r.getInt(26)).getOrElse(-1)

      val ctr = if (impCount > 50) clickCount * 1.0 / impCount else histctr
      val adCtr = if (adImpCount > 10000) adClickCount * 1.0 / adImpCount else 0.007450876279364931

      val trueSearchLoc = if (searchLoc > 1250000) {
        searchLoc - 1250000 + 4666
      } else if (searchLoc > 1000000) {
        searchLoc - 1000000 + 4629
      } else if (searchLoc > 750000) {
        searchLoc - 750000 + 4592
      } else if (searchLoc > 250000) {
        4592
      } else {
        searchLoc
      }

      val trueSearchCat = if (searchCat > 250000) {
        searchCat - 250000 + 60
      } else if (searchCat  == 500001) {
        searchCat - 500000 + 6
      } else {
        searchCat
      }

      val trueCategory = if (category > 250000) {
        category - 250000 + 60
      } else if (searchCat  == 500001) {
        category - 500000 + 6
      } else {
        category
      }

      //<=38
      //y = 3E-05x2 + 0.0004x + 0.0103
      //0.045871559633027525
//      val lenQuery = length(searchQuery)
//      val lenthBased = if ( lenQuery < 38) {
//        math.min(0.0, 3E-05*lenQuery*lenQuery + 0.0004*lenQuery + 0.0103)
//      } else {
//        0.04587155963302
//      }

      val qParamBlockSize = (numFeatures - 4964)/2
      val aParamBlockSize = qParamBlockSize //math.pow(2, 14).toInt

      val qParamOffset = 4964
      val aParamOffset = qParamOffset + qParamBlockSize //qParamOffset + qParamBlockSize

      val featureIndices =
        //hasher.createFeatures_Int("OS", os) ++
//          hasher.createFeatures_Int("UA", uafam) ++
          Seq(math.abs(loggedIn) % numFeatures) ++ //hasher.createFeatures_Int(freeBlockSize, freeBlockOffset, "LoggedIn", loggedIn) ++
          Seq(math.abs(dayOfWeek(searchTime) + 2) % numFeatures) ++  // hasher.  createFeatures_Int(freeBlockSize, freeBlockOffset, "DayOfWeek", dayOfWeek(searchTime)) ++
          Seq(math.abs(searchTime_hour + 9) % numFeatures) ++ //hasher.  createFeatures_Int(freeBlockSize, freeBlockOffset, "Hour", searchTime_hour ) ++
//          hasher.  createFeatures_Int("TimeMorning", time_morning(searchTime_hour) ) ++
//          hasher.   createFeatures_Int("TimeNoon", time_noon (searchTime_hour) ) ++
//          hasher.    createFeatures_Int("TimeAfter", time_afternoon (searchTime_hour) ) ++
//          hasher.   createFeatures_Int("TimeEvening", time_evening (searchTime_hour) ) ++
//          hasher.   createFeatures_Int("TimeLate", time_late_evening(searchTime_hour) ) ++
//          hasher.    createFeatures_Int("TimeNight", time_night(searchTime_hour) ) ++
          Seq(math.min(math.abs(trueSearchLoc + 33) % numFeatures, 4700)) ++ // hasher.   createFeatures_Int(freeBlockSize, freeBlockOffset, "QLoc", searchLoc) ++
          Seq(math.abs(trueSearchCat + 4701) % numFeatures) ++ //hasher.     createFeatures_Int(freeBlockSize, freeBlockOffset, "QCat", searchCat) ++
            Seq(if (position > 4) 4770 else  4769) ++ // hasher.    createFeatures_Int(freeBlockSize, freeBlockOffset, "Pos", position) ++
            Seq(math.abs(searchLocLevel + 4772) % numFeatures) ++ //hasher.   createFeatures_Int(freeBlockSize, freeBlockOffset, "searchLocLevel", searchLocLevel) ++
            Seq(math.abs(searchLocPar + 4776) % numFeatures) ++ //hasher.   createFeatures_Int(freeBlockSize, freeBlockOffset, "searchLocPar", searchLocPar) ++
            Seq(math.abs(searchCatLevel + 4862) % numFeatures) ++ //hasher.   createFeatures_Int(freeBlockSize, freeBlockOffset, "searchCatLevel", searchCatLevel) ++
            Seq(math.abs(searchCatPar + 4865) % numFeatures) ++ //hasher.   createFeatures_Int(freeBlockSize, freeBlockOffset, "searchCatPar", searchCatPar) ++
            Seq(math.abs(adCatLevel + 4878) % numFeatures) ++ //hasher.   createFeatures_Int(freeBlockSize, freeBlockOffset, "adCatLevel", adCatLevel) ++
            Seq(math.abs(adCatPar + 4881) % numFeatures) ++ //hasher.   createFeatures_Int(freeBlockSize, freeBlockOffset, "adCatPar", adCatPar) ++
            Seq(math.abs(trueCategory + 4884) % numFeatures) ++  // hasher.  createFeatures_Int(freeBlockSize, freeBlockOffset, "Cat", category) ++
            Seq((if (price <= 0) 1 else 0) + 4953) ++   //hasher.  createFeatures_Int(freeBlockSize, freeBlockOffset, "PriceMiss", if (price <= 0) 1 else 0) ++
            Seq ((if (impCount < 15) 1 else 0) + 4955) ++ //hasher.  createFeatures_Int(freeBlockSize, freeBlockOffset, "NewVisitor", if (impCount < 15) 1 else 0) ++
            Seq( (if (phoneCount > 1) 1 else 0) + 4957) ++ //hasher.  createFeatures_Int(freeBlockSize, freeBlockOffset, "CalledBefore", if (phoneCount > 1) 1 else 0) ++
            Seq( (if (visitCount > 0) 1 else 0) + 4959)++ //hasher.  createFeatures_Int(freeBlockSize, freeBlockOffset, "VisitedBefore", if (visitCount > 1) 1 else 0)
//          hasher.     createFeatures_Int(aParamBlockSize, aParamOffset, "QueryLen", length(searchQuery))) ++
          hasher.   createFeatures_Int(qParamBlockSize, qParamOffset, "QParams", searchParams:_*) ++
          hasher. createFeatures_Int(aParamBlockSize, aParamOffset, "adpars", params:_*)
//          hasher.  createFeatures_Int("titleLen", length(title))
//                hasher.  createFeatures_Int(freeBlockSize, freeBlockOffset, "PriceHigh", if (price > 30000) 1 else 0) ++
//                hasher.  createFeatures_Int(freeBlockSize, freeBlockOffset, "PriceLow",if (price > 0 & price < 30000) 1 else 0) ++
//      hasher.  createFeatures_Int("FrequentVisitor", if (impCount > 2000) 1 else 0) ++

      val singleCatFeatures = hasher.setFeaturesValue(featureIndices.toSet.toSeq, 1.0)

      val singleContFeatures =
        Seq((4963, ctr), (4964, adCtr), (4965, hasher.numberOfCommonElements(searchParams, params).toDouble), (4961, 1.0*length(searchQuery))) ++
        hasher.sentenceFeatures(qParamBlockSize, qParamOffset, "title", title) ++
        hasher.sentenceFeatures(aParamBlockSize, aParamOffset, "query", searchQuery)
//        Seq(hasher.hashAndSetFeatureValue(freeBlockSize, freeBlockOffset, "paramMatch", hasher.numberOfCommonElements(searchParams, params).toDouble))

      val features = singleCatFeatures ++ singleContFeatures

      LabeledPoint(isClick, Vectors.sparse(numFeatures, hasher.dedupeFeatures(features)))
    }.toDF()

  }






      //      val immediateCategoricalFeatures =
//        hasher.createFeatures_Int("OS", os) ++
//        hasher.createFeatures_Int("UA", uafam) ++
//        hasher.createFeatures_Int("LoggedIn", loggedIn) ++
//        hasher.createFeatures_Int("Monday", if(_dayOfWeek(searchTime) == 0) 1 else 0) ++
//        hasher.createFeatures_Int("Weekend", _weekend(searchTime)) ++
//        hasher.createFeatures_Int("TimeMorning", _time_morning(searchTime_hour) ) ++
//        hasher.createFeatures_Int("TimeAfter", _time_afternoon (searchTime_hour) ) ++
//        hasher.createFeatures_Int("TimeEvening", _time_evening (searchTime_hour) ) ++
//        hasher.createFeatures_Int("QueryEmpty", if(_length(searchQuery) < 1) 1 else 0) ++
//        hasher.createFeatures_Int("QLoc", searchLoc) ++
//        hasher.createFeatures_Int("QCat", searchCat) ++
//        hasher.createFeatures_Int("QParams", searchParams:_*) ++
//        hasher.createFeatures_Int("Pos", position) ++
//        hasher.createFeatures_Int("Cat", category) ++
//        hasher.createFeatures_Int("adpars", params:_*) ++
//        hasher.createFeatures_Int("shortTitle", if (_length(title) < 100) 1 else 0) ++
//        hasher.createFeatures_Int("PriceMid", if (price >= 3000 && price < 21000) 1 else 0) ++
//        hasher.createFeatures_Int("PriceLow", if (price < 3000) 1 else 0) ++
//        hasher.createFeatures_Int("PriceNo", if (price < 0) 1 else 0)
//
//      val categoricalFeatures = hasher.combinePairs(
//        hasher.combine(immediateCategoricalFeatures) ++
//        hasher.sentenceFeatures("title", title) ++
//        hasher.sentenceFeatures("query", searchQuery))
//
//      val contFeatures = hasher.combinePairs(Seq(
//        hasher.hashFeatureAmount("HistCTR", math.max(histctr, 0.0)),
//        hasher.hashFeatureAmount("paramMatch", hasher.paramOverlap(searchParams, params).toDouble),
//        //hasher.hashFeatureAmount("visitCount", visitCount.toDouble),
//        //hasher.hashFeatureAmount("phoneCount", phoneCount.toDouble),
//        //hasher.hashFeatureAmount("impCount", impCount.toDouble),
//        //hasher.hashFeatureAmount("adClickCount", adClickCount),
//        //hasher.hashFeatureAmount("adImpCount", adImpCount),
//        //hasher.hashFeatureAmount("ctr", ctr),
//        hasher.hashFeatureAmount("adctr", adCtr)))
//
//      val finalFeatures = if (addInteractions) {
//        hasher.combinePairs(otherInteractions(categoricalFeatures) ++ crossInteractions(contFeatures, contFeatures) ++ crossInteractions(categoricalFeatures, contFeatures))
//      } else {
//        hasher.combinePairs(categoricalFeatures ++ contFeatures)
//      }
//
//      LabeledPoint(isClick, Vectors.sparse(numFeatures, finalFeatures))
//
//






//  def otherInteractions(singleFeatures: Seq[(Int, Double)]): Seq[(Int, Double)] = {
//    (for {
//      a <- singleFeatures
//      b <- singleFeatures
//    } yield {
//      if (a._1 != b._1 && a._2 != 0.0 && b._2 != 0.0) {
//        Some(hasher.hashAndSetFeatureValue(s"Interact_${a._1}_${b._1}_", a._2 * b._2))
//      } else {
//        None
//      }
//    }).flatten
//  }
//
//  def crossInteractions(singleFeatures: Seq[(Int, Double)], otherFeatures: Seq[(Int, Double)]): Seq[(Int, Double)] = {
//    (for {
//      a <- singleFeatures
//      b <- otherFeatures
//    } yield {
//        if (a._2 != 0.0 && b._2 != 0.0) {
//          Some(hasher.hashAndSetFeatureValue(s"Interact_${a._1}_${b._1}_", a._2 * b._2))
//        } else {
//          None
//        }
//      }).flatten
//  }

}
