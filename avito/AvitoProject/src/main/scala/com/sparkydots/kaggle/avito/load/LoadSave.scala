package com.sparkydots.kaggle.avito.load

import com.sparkydots.kaggle.avito.functions.UdfFunctions._
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}

object LoadSave {

  val processedDir = "banana"

  def loadOrigDF(sqlContext: SQLContext, filename: String) = sqlContext.load("com.databricks.spark.csv", Map("header" -> "true", "delimiter" -> "\t", "path" -> s"s3n://sparkydotsdata/kaggle/avito/${filename}.tsv"))
  def loadDF(sqlContext: SQLContext, filename: String) = sqlContext.load(s"s3n://sparkydotsdata/kaggle/avito/${processedDir}/${filename}.parquet")
  def saveDF(sqlContext: SQLContext, df: DataFrame, filename: String) = df.saveAsParquetFile(s"s3n://sparkydotsdata/kaggle/avito/${processedDir}/${filename}.parquet")

  /**
   * Load data from original Kaggle data files.
   * Save processed dataframes.
   * @param sqlContext
   * @return
   */
  def loadOrig(sqlContext: SQLContext) = {

    val _users = loadOrigDF(sqlContext, "UserInfo")
    val users = _users
      .withColumn("id", udf_toInt(_users("UserID")))
      .withColumn("os", udf_toInt(_users("UserAgentOSID")))
      .withColumn("uafam", udf_toInt(_users("UserAgentFamilyID")))
      .select("id", "os", "uafam")
      .repartition(1)
      .cache()
    saveDF(sqlContext, users, "users")

    val _ads = loadOrigDF(sqlContext, "AdsInfo")
    val ads = _ads.
      withColumn("id", udf_toInt(_ads("AdID"))).
      withColumn("category", udf_toIntOrMinus(_ads("CategoryID"))).
      withColumn("params", udf_parseParams(_ads("Params"))).
      withColumn("price", udf_toDoubleOrMinus(_ads("Price"))).
      withColumn("title", udf_toLower(_ads("Title"))).
      withColumn("isContext", udf_toInt(_ads("IsContext"))).
      select("id", "category", "params", "price", "title", "isContext").
      filter("isContext = 1").
      repartition(8).
      cache()
    saveDF(sqlContext, ads, "ads")

    val ctxAds = ads.filter("isContext = 1").repartition(2).cache()
    saveDF(sqlContext, ctxAds, "ctxAds")

    val nonCtxAds = ads.filter("isContext != 1").cache()
    saveDF(sqlContext, nonCtxAds, "nonCtxAds")

    val _searches = loadOrigDF(sqlContext, "SearchInfo")
    val searches = _searches.
      withColumn("id", udf_toInt(_searches("SearchID"))).
      withColumn("searchTime", udf_parseTime(_searches("SearchDate"))).
      withColumn("userId", udf_toInt(_searches("UserID"))).
      withColumn("loggedIn", udf_toInt(_searches("IsUserLoggedOn"))).
      withColumn("searchQuery", udf_toLower(_searches("SearchQuery"))).
      withColumn("searchLoc", udf_toIntOrMinus(_searches("LocationID"))).
      withColumn("searchCat", udf_toIntOrMinus(_searches("CategoryID"))).
      withColumn("searchParams", udf_parseParams(_searches("SearchParams"))).
      select("id", "searchTime", "userId", "loggedIn", "searchQuery", "searchLoc", "searchCat", "searchParams").
      repartition(24).
      cache()
    saveDF(sqlContext, searches, "searches")

    val _adImpressions = loadOrigDF(sqlContext, "trainSearchStream")
    val adImpressions = _adImpressions. // (mid, (searchId, adId, position, histCTR, isClick))
      withColumn("mid", udf_toMid(_adImpressions("SearchID"), _adImpressions("AdID"))).
      withColumn("searchId", udf_toInt(_adImpressions("SearchID"))).
      withColumn("adId", udf_toInt(_adImpressions("AdID"))).
      withColumn("position", udf_toInt(_adImpressions("Position"))).
      withColumn("type", udf_toInt(_adImpressions("ObjectType"))). //1, 2, 3
      withColumn("histctr", udf_toDoubleOrMinus(_adImpressions("HistCTR"))). // only for ObjectType 3
      withColumn("isClick", udf_toIntOrMinus(_adImpressions("IsClick"))). // now: -1, 0, 1
      select("mid", "searchId", "adId", "position", "type", "histctr", "isClick").
      repartition(24).
      cache()

    val nonCtxAdImpressions =  adImpressions.
      filter("type != 3").
      select("mid", "searchId", "adId", "position", "histctr", "isClick")
    saveDF(sqlContext, nonCtxAdImpressions, "nonCtxAdImpressions")

    val ctxAdImpressions =  adImpressions.
      filter("type = 3").
      select("mid", "searchId", "adId", "position", "histctr", "isClick")
    saveDF(sqlContext, ctxAdImpressions, "ctxAdImpressions")

    val _adImpressionsToFind = loadOrigDF(sqlContext, "testSearchStream")
    val adImpressionsToFind = _adImpressionsToFind.
      withColumn("id", udf_toInt(_adImpressionsToFind("ID"))).
      withColumn("searchId", udf_toInt(_adImpressionsToFind("SearchID"))).
      withColumn("adId", udf_toInt(_adImpressionsToFind("AdID"))).
      withColumn("position", udf_toInt(_adImpressionsToFind("Position"))).
      withColumn("type", udf_toInt(_adImpressionsToFind("ObjectType"))). //1, 2, 3
      withColumn("histctr", udf_toDoubleOrMinus(_adImpressionsToFind("HistCTR"))). // only for ObjectType 3
      select("id", "searchId", "adId", "position", "type", "histctr").
      repartition(24).
      cache()

    val ctxAdImpressionsToFind = adImpressionsToFind
      .filter("type = 3")
      .select("id", "searchId", "adId", "position", "histctr")

    saveDF(sqlContext, ctxAdImpressionsToFind, "ctxAdImpressionsToFind")

    val nonCtxAdImpressionsToFind = adImpressionsToFind.
      filter("type != 3").
      select("id", "searchId", "adId", "position", "histctr")

    saveDF(sqlContext, nonCtxAdImpressionsToFind, "nonCtxAdImpressionsToFind")

    val _visits = loadOrigDF(sqlContext, "VisitsStream")
    val visits = _visits.
      withColumn("userId", udf_toInt(_visits("UserID"))).
      withColumn("ad", udf_toInt(_visits("AdID"))).
      withColumn("visitTime", udf_parseTime(_visits("ViewDate"))).
      select("userId", "ad", "visitTime").
      repartition(32).
      cache()
    saveDF(sqlContext, visits, "visits")

    val _dfPhone = loadOrigDF(sqlContext, "PhoneRequestsStream")
    val phoneRequests = _dfPhone.
      withColumn("userId", udf_toInt(_dfPhone("UserID"))).
      withColumn("ad", udf_toInt(_dfPhone("AdID"))).
      withColumn("phoneTime", udf_parseTime(_dfPhone("PhoneRequestDate"))).
      select("userId", "ad", "phoneTime").
      repartition(1).
      cache()
    saveDF(sqlContext, phoneRequests, "phoneRequests")

    // LocationID	Level	RegionID
    val _dfLoc = loadOrigDF(sqlContext, "Location")
    val locations = _dfLoc.
      withColumn("id", udf_toInt(_dfLoc("LocationID"))).
      withColumn("level", udf_toInt(_dfLoc("Level"))).
      withColumn("regionId", udf_toIntOrMinus(_dfLoc("RegionID"))).
      select("id", "level", "regionId").
      repartition(1).
      cache()
    saveDF(sqlContext, locations, "locations")

    // CategoryID	Level	ParentCategoryID	SubcategoryID
    val _dfCat = loadOrigDF(sqlContext, "Category")
    val categories = _dfCat.
      withColumn("id", udf_toInt(_dfCat("CategoryID"))).
      withColumn("level", udf_toInt(_dfCat("Level"))).
      withColumn("catId", udf_toIntOrMinus(_dfCat("ParentCategoryID"))).
      select("id", "level", "catId").
      repartition(1).
      cache()
    saveDF(sqlContext, categories, "categories")

    (users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, nonCtxAdImpressions, ctxAdImpressionsToFind, nonCtxAdImpressionsToFind, visits, phoneRequests, locations, categories)
  }

  /**
   * Load processed individual table data.
   * @param sqlContext
   * @return
   */
  def loadOrigCached(sqlContext: SQLContext) = {

    val users = loadDF(sqlContext, "users")
    val ads = loadDF(sqlContext, "ads").cache()
    val ctxAds = loadDF(sqlContext, "ctxAds").cache()
    val nonCtxAds = loadDF(sqlContext, "nonCtxAds").cache()
    val searches = loadDF(sqlContext, "searches").cache()
    val ctxAdImpressions = loadDF(sqlContext, "ctxAdImpressions").cache()
    val nonCtxAdImpressions = loadDF(sqlContext, "nonCtxAdImpressions").cache()
    val ctxAdImpressionsToFind = loadDF(sqlContext, "ctxAdImpressionsToFind").cache()
    val nonCtxAdImpressionsToFind = loadDF(sqlContext, "nonCtxAdImpressionsToFind").cache()
    val visits = loadDF(sqlContext, "visits")
    val phoneRequests = loadDF(sqlContext, "phoneRequests")
    val locations = loadDF(sqlContext, "locations")
    val categories = loadDF(sqlContext, "categories")

    (users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, nonCtxAdImpressions, ctxAdImpressionsToFind, nonCtxAdImpressionsToFind, visits, phoneRequests, locations, categories)
  }

  /**
   * Load processed individual table data, either from orig Kaggle data files or from saved processed data.
   * Prepare training data:
   *  1. Perform joins and bring all data along context ad impressions.
   *  2. Enrich data with basic count data (ad impressions, clicks, visits, phone requests) where possible.
   *  3. Split data into train/validate and also eval/small.
   * Save processed, enriched and split files.
   * @param sc
   * @param sqlContext
   * @param orig whether to load from original Kaggle source data files
   * @return
   */
  def splitData(sc: SparkContext, sqlContext: SQLContext, prefix: String, orig: Boolean = false) = {
    val (users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, nonCtxAdImpressions, ctxAdImpressionsToFind, nonCtxAdImpressionsToFind, visits, phoneRequests, locations, categories) =
      if (orig)
        LoadSave.loadOrig(sqlContext)
      else
        LoadSave.loadOrigCached(sqlContext)

    val (evalData, trainData, validateData, smallData) =
      TrainingData.split(sqlContext, users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, nonCtxAdImpressions, ctxAdImpressionsToFind, nonCtxAdImpressionsToFind, visits, phoneRequests, locations, categories)
    LoadSave.saveDF(sqlContext, trainData, s"${prefix}TRAIN")
    LoadSave.saveDF(sqlContext, validateData, s"${prefix}VALIDATE")
    LoadSave.saveDF(sqlContext, evalData, s"${prefix}EVAL")
    LoadSave.saveDF(sqlContext, smallData, s"${prefix}SMALL")

    (users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, nonCtxAdImpressions, ctxAdImpressionsToFind, nonCtxAdImpressionsToFind, visits, phoneRequests, locations, categories, evalData, trainData, validateData, smallData)
  }

  /**
   * Load processed, enriched and split files.
   * @param sc
   * @param sqlContext
   * @return
   */
  def loadDatasets(sc: SparkContext, sqlContext: SQLContext, prefix: String) = {

    val rawTrain = LoadSave.loadDF(sqlContext, s"${prefix}TRAIN").cache()
    val rawValidate = LoadSave.loadDF(sqlContext, s"${prefix}VALIDATE").cache()
    val rawEval = LoadSave.loadDF(sqlContext, s"${prefix}EVAL").cache()
    val rawSmall = LoadSave.loadDF(sqlContext, s"${prefix}SMALL").cache()

    (rawTrain, rawValidate, rawEval, rawSmall)
  }

}
