package com.sparkydots.kaggle.avito

import org.apache.spark.SparkContext
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/*
spark-shell --jars AvitoProject-assembly-1.0.jar
val (sqlContext, userRDD, ctxAdRDD, searchRDD, ctxImpressionsRDD, ctxToFindRDD, dfVisit, dfPhone, trainSet, validateSet, testSet, validHistorySet) = com.sparkydots.kaggle.avito.ReadData.ingest(sc)
 */

object ReadData {

  def ingest(sc: SparkContext) = {
    Logger.getLogger("amazon.emr.metrics").setLevel(Level.OFF)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    import Functions._
    import UdfFunctions._

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    sqlContext.udf.register("strLen", (s: String) => s.length())
    sqlContext.udf.register("errf", _error)

    def loadDF(filename: String) = sqlContext.load("com.databricks.spark.csv", Map("header" -> "true", "delimiter" -> "\t", "path" -> s"s3n://sparkydotsdata/kaggle/avito/${filename}.tsv"))

    //User
    val _dfUser = loadDF("UserInfo")
    val userRDD = _dfUser
      .withColumn("id", toInt(_dfUser("UserID")))
      .withColumn("uaOsId", toInt(_dfUser("UserAgentOSID")))
      .withColumn("uaFamilyId", toInt(_dfUser("UserAgentFamilyID")))
      .select("id", "uaOsId", "uaFamilyId").rdd
      .map(r => (r.getInt(0), (r.getInt(1), r.getInt(2))))

    //Ad: 5G
    val _dfAd = loadDF("AdsInfo")
    val ctxAdRDD = _dfAd  // (adId, (catId, params, price, title) )
      .withColumn("id", toInt(_dfAd("AdID")))
      .withColumn("categoryId", toIntOrMinus(_dfAd("CategoryID")))
      .withColumn("params", parseParams(_dfAd("Params")))
      .withColumn("price", toDoubleOrMinus(_dfAd("Price")))
      .withColumn("title", toLower(_dfAd("Title")))
      .withColumn("isContext", toInt(_dfAd("IsContext")))
      .select("id", "categoryId", "params", "price", "title", "isContext")
      .filter("isContext = 1").rdd
      .map(r => (r.getInt(0), (r.getInt(1), r.getString(2).split(",").filter(_.nonEmpty).map(_.toInt).toSeq, r.getDouble(3), r.getString(4))))

    // SearchInfo : 8.8G
    val _dfSearchInfo = loadDF("SearchInfo")
    val searchRDD = _dfSearchInfo // (searchId, (eventTime, userId, userLoggedIn?,   query, loc, cat, params) )
      .withColumn("id", toInt(_dfSearchInfo("SearchID")))
      .withColumn("eventTime", parseTime(_dfSearchInfo("SearchDate")))
      .withColumn("userId", toInt(_dfSearchInfo("UserID")))
      .withColumn("userLogged", toInt(_dfSearchInfo("IsUserLoggedOn")))
      .withColumn("searchQuery", toLower(_dfSearchInfo("SearchQuery")))
      .withColumn("locationId", toIntOrMinus(_dfSearchInfo("LocationID")))
      .withColumn("categoryId", toIntOrMinus(_dfSearchInfo("CategoryID")))
      .withColumn("params", parseParams(_dfSearchInfo("SearchParams")))
      .select("id", "eventTime", "userId", "userLogged", "searchQuery", "locationId", "categoryId", "params").rdd
      .map(r => (r.getInt(0), (r.getInt(1), r.getInt(2), r.getInt(3), r.getString(4), r.getInt(5), r.getInt(6), r.getString(7).split(",").filter(_.nonEmpty).map(_.toInt).toSeq)))
      .cache()



    // SearchStream : 10G  -- trainSearchStream.tsv  (5.9G)
    val _dfSearchStream = loadDF("trainSearchStream")
    val ctxImpressionsRDD = _dfSearchStream // (mid, (searchId, adId, position, histCTR, isClick))
      .withColumn("mid", toMid(_dfSearchStream("SearchID"), _dfSearchStream("AdID")))
      .withColumn("searchId", toInt(_dfSearchStream("SearchID")))
      .withColumn("adId", toInt(_dfSearchStream("AdID")))
      .withColumn("position", toInt(_dfSearchStream("Position")))
      .withColumn("type", toInt(_dfSearchStream("ObjectType"))) //1, 2, 3
      .withColumn("histctr", toDoubleOrMinus(_dfSearchStream("HistCTR"))) // only for ObjectType 3
      .withColumn("isClick", toIntOrMinus(_dfSearchStream("IsClick"))) // now: -1, 0, 1
      .select("mid", "searchId", "adId", "position", "type", "histctr", "isClick")
      .filter("type = 3").select("mid", "searchId", "adId", "position", "histctr", "isClick").rdd
      .map(r => (r.getLong(0), (r.getInt(1), r.getInt(2), r.getInt(3), r.getDouble(4), r.getInt(5))))
      .cache()

    // SearchStream : 0.5G  -- testSearchStream.tsv
    val _dfSearchStreamToFind = loadDF("testSearchStream")
    val  ctxToFindRDD = _dfSearchStreamToFind.
      withColumn("id", toInt(_dfSearchStreamToFind("ID"))).
      withColumn("searchId", toInt(_dfSearchStreamToFind("SearchID"))).
      withColumn("adId", toInt(_dfSearchStreamToFind("AdID"))).
      withColumn("position", toInt(_dfSearchStreamToFind("Position"))).
      withColumn("type", toInt(_dfSearchStreamToFind("ObjectType"))). //1, 2, 3
      withColumn("histctr", toDoubleOrMinus(_dfSearchStreamToFind("HistCTR"))). // only for ObjectType 3
      select("id", "searchId", "adId", "position", "type", "histctr")
      .filter("type = 3")
      .rdd


    ////////////////////////////////////////////////////////////////////////
    // Visit : 12G (3.2G) AND Phone Request: 0.6G
    val _dfVisit = loadDF("VisitsStream")
    val dfVisit = _dfVisit.
      withColumn("userId", toInt(_dfVisit("UserID"))).
      withColumn("ipId", toInt(_dfVisit("IPID"))).
      withColumn("adId", toInt(_dfVisit("AdID"))).
      withColumn("eventTime", parseTime(_dfVisit("ViewDate"))).
      select("userId", "ipId", "adId", "eventTime")

    val _dfPhone = loadDF("PhoneRequestsStream")
    val dfPhone = _dfPhone.
      withColumn("userId", toInt(_dfPhone("UserID"))).
      withColumn("ipId", toInt(_dfPhone("IPID"))).
      withColumn("adId", toInt(_dfPhone("AdID"))).
      withColumn("eventTime", parseTime(_dfPhone("PhoneRequestDate"))).
      select("userId", "ipId", "adId", "eventTime")
    ////////////////////////////////////////////////////////////////////////

    val ctxImpressionsBySearchRDD = ctxImpressionsRDD
      .map({  case (mid, (searchId, adId, position, histCTR, isClick)) => (searchId, (histCTR, isClick, mid)) }).cache()

    val evaluateSet: RDD[(Int, (Int, Iterable[Long]))] =  // [  (userId, (cutoffTime, [mid, mid, ...])), (userId, (cutoffTime, [mid, mid, ...])), ... ]
      ctxImpressionsBySearchRDD
      .join(searchRDD.filter(_._2._1 > 1900800).mapValues({ case (eventTime, userId, userLoggedIn, query, loc, cat, params) => (userId, eventTime)}))
      .map({ case (searchId, ((histCTR, isClick, mid), (userId, eventTime))) => (userId, (eventTime, histCTR, isClick, mid))})
      .groupByKey()
      .map({ case (userId, impressions) =>
        val els = impressions.groupBy(imp => imp._1 / 60).toSeq.map(_._2)
        val el = els(41*userId % els.size)
        (userId, (el.map(_._1).min, el.map(_._4)))
      })

    val Array(_trainSet, _validateSet, _testSet) = evaluateSet.randomSplit(Array(0.65, 0.25, 0.10))

    val trainSet = _trainSet.flatMap(_._2._2).setName("trainSet").cache()
    val validateSet = _validateSet.flatMap(_._2._2).setName("validateSet").cache()
    val testSet = _testSet.flatMap(_._2._2).setName("testSet").cache()

    val validHistorySet = ctxImpressionsBySearchRDD
      .join(searchRDD.mapValues({ case (eventTime, userId, userLoggedIn, query, loc, cat, params) => (userId, eventTime)}))
      .map({ case (searchId, ((histCTR, isClick, mid), (userId, eventTime))) => (userId, (eventTime, mid)) })
      .join(evaluateSet)
      .flatMap({
        case (userId, ((eventTime, mid), (cutoffTime, _))) if eventTime < cutoffTime => Some(mid)
        case _ => None
      }).setName("trainSet").cache()


    println("calculating errors")
    val trainError = ctxImpressionsRDD.join(trainSet.map((_, 1))).map({ case (_,( (searchId, adId, position, histCTR, isClick) ,_)) => _error(histCTR, isClick) }).mean()
    val validateError = ctxImpressionsRDD.join(validateSet.map((_, 1))).map({ case (_,( (searchId, adId, position, histCTR, isClick) ,_)) => _error(histCTR, isClick) }).mean()
    val testError = ctxImpressionsRDD.join(testSet.map((_, 1))).map({ case (_,( (searchId, adId, position, histCTR, isClick) ,_)) => _error(histCTR, isClick) }).mean()

    println(s"Errors:\nTrain\tValidate\tTest\n${trainError}\t${validateError}\t${testError}")

    (sqlContext, userRDD, ctxAdRDD, searchRDD, ctxImpressionsRDD, ctxToFindRDD, dfVisit, dfPhone, trainSet, validateSet, testSet, validHistorySet)
  }
}
