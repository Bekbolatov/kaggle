package com.sparkydots.kaggle.avito

import com.sparkydots.kaggle.avito.functions.UdfFunctions._
import org.apache.spark.sql.{SQLContext, DataFrame}

object LoadSave {

  def loadOrigDF(sqlContext: SQLContext, filename: String) = sqlContext.load("com.databricks.spark.csv", Map("header" -> "true", "delimiter" -> "\t", "path" -> s"s3n://sparkydotsdata/kaggle/avito/${filename}.tsv"))
  def loadDF(sqlContext: SQLContext, filename: String) = sqlContext.load(s"s3n://sparkydotsdata/kaggle/avito/parsed/${filename}.parquet")
  def saveDF(sqlContext: SQLContext, df: DataFrame, filename: String) = df.saveAsParquetFile(s"s3n://sparkydotsdata/kaggle/avito/parsed/${filename}.parquet")

  def origLoad(sqlContext: SQLContext) = {

    val _users = loadOrigDF(sqlContext, "UserInfo")
    val users = _users
      .withColumn("id", toInt(_users("UserID")))
      .withColumn("os", toInt(_users("UserAgentOSID")))
      .withColumn("uafam", toInt(_users("UserAgentFamilyID")))
      .select("id", "os", "uafam")
      .cache()
    saveDF(sqlContext, users, "users")

    val _ads = loadOrigDF(sqlContext, "AdsInfo")
    val ads = _ads
      .withColumn("id", toInt(_ads("AdID")))
      .withColumn("category", toIntOrMinus(_ads("CategoryID")))
      .withColumn("params", parseParams(_ads("Params")))
      .withColumn("price", toDoubleOrMinus(_ads("Price")))
      .withColumn("title", toLower(_ads("Title")))
      .withColumn("isContext", toInt(_ads("IsContext")))
      .select("id", "category", "params", "price", "title", "isContext")
      .filter("isContext = 1")
      .cache()
    saveDF(sqlContext, ads, "ads")

    val ctxAds = ads.filter("isContext = 1").cache()
    saveDF(sqlContext, ctxAds, "ctxAds")

    val nonCtxAds = ads.filter("isContext != 1").cache()
    saveDF(sqlContext, nonCtxAds, "nonCtxAds")

    val _searches = loadOrigDF(sqlContext, "SearchInfo")
    val searches = _searches.
      withColumn("id", toInt(_searches("SearchID"))).
      withColumn("searchTime", parseTime(_searches("SearchDate"))).
      withColumn("userId", toInt(_searches("UserID"))).
      withColumn("loggedIn", toInt(_searches("IsUserLoggedOn"))).
      withColumn("searchQuery", toLower(_searches("SearchQuery"))).
      withColumn("searchLoc", toIntOrMinus(_searches("LocationID"))).
      withColumn("searchCat", toIntOrMinus(_searches("CategoryID"))).
      withColumn("searchParams", parseParams(_searches("SearchParams"))).
      select("id", "searchTime", "userId", "loggedIn", "searchQuery", "searchLoc", "searchCat", "searchParams").
      cache()
    saveDF(sqlContext, searches, "searches")

    val _adImpressions = loadOrigDF(sqlContext, "trainSearchStream")
    val ctxAdImpressions = _adImpressions. // (mid, (searchId, adId, position, histCTR, isClick))
      withColumn("mid", toMid(_adImpressions("SearchID"), _adImpressions("AdID"))).
      withColumn("searchId", toInt(_adImpressions("SearchID"))).
      withColumn("adId", toInt(_adImpressions("AdID"))).
      withColumn("position", toInt(_adImpressions("Position"))).
      withColumn("type", toInt(_adImpressions("ObjectType"))). //1, 2, 3
      withColumn("histctr", toDoubleOrMinus(_adImpressions("HistCTR"))). // only for ObjectType 3
      withColumn("isClick", toIntOrMinus(_adImpressions("IsClick"))). // now: -1, 0, 1
      select("mid", "searchId", "adId", "position", "type", "histctr", "isClick").
      filter("type = 3").select("mid", "searchId", "adId", "position", "histctr", "isClick").
      cache()
    saveDF(sqlContext, ctxAdImpressions, "ctxAdImpressions")

    val _adImpressionsToFind = loadOrigDF(sqlContext, "testSearchStream")
    val ctxAdImpressionsToFind = _adImpressionsToFind
      .withColumn("id", toInt(_adImpressionsToFind("ID")))
      .withColumn("searchId", toInt(_adImpressionsToFind("SearchID")))
      .withColumn("adId", toInt(_adImpressionsToFind("AdID")))
      .withColumn("position", toInt(_adImpressionsToFind("Position")))
      .withColumn("type", toInt(_adImpressionsToFind("ObjectType"))) //1, 2, 3
      .withColumn("histctr", toDoubleOrMinus(_adImpressionsToFind("HistCTR"))) // only for ObjectType 3
      .select("id", "searchId", "adId", "position", "type", "histctr")
      .filter("type = 3")
      .cache()
    saveDF(sqlContext, ctxAdImpressionsToFind, "ctxAdImpressionsToFind")

    val _visits = loadOrigDF(sqlContext, "VisitsStream")
    val visits = _visits.
      withColumn("userId", toInt(_visits("UserID"))).
      withColumn("ad", toInt(_visits("AdID"))).
      withColumn("visitTime", parseTime(_visits("ViewDate"))).
      select("userId", "ad", "visitTime").
      cache()
    saveDF(sqlContext, visits, "visits")

    val _dfPhone = loadOrigDF(sqlContext, "PhoneRequestsStream")
    val phoneRequests = _dfPhone.
      withColumn("userId", toInt(_dfPhone("UserID"))).
      withColumn("ad", toInt(_dfPhone("AdID"))).
      withColumn("phoneTime", parseTime(_dfPhone("PhoneRequestDate"))).
      select("userId", "ad", "phoneTime").
      cache()
    saveDF(sqlContext, phoneRequests, "phoneRequests")

    (users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests)
  }

  def load(sqlContext: SQLContext) = {

    val users = loadDF(sqlContext, "users")
    val ads = loadDF(sqlContext, "ads").cache()
    val ctxAds = loadDF(sqlContext, "ctxAds").cache()
    val nonCtxAds = loadDF(sqlContext, "nonCtxAds").cache()
    val searches = loadDF(sqlContext, "searches").cache()
    val ctxAdImpressions = loadDF(sqlContext, "ctxAdImpressions").cache()
    val ctxAdImpressionsToFind = loadDF(sqlContext, "ctxAdImpressionsToFind").cache()
    val visits = loadDF(sqlContext, "visits")
    val phoneRequests = loadDF(sqlContext, "phoneRequests")

    (users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests)
  }

}
