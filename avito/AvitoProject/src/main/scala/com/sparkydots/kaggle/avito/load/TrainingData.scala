package com.sparkydots.kaggle.avito.load

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

object TrainingData {

  def split(sqlContext: SQLContext,
            users: DataFrame,
            ads: DataFrame, ctxAds: DataFrame, nonCtxAds: DataFrame,
            searches: DataFrame,
            ctxAdImpressions: DataFrame,
            nonCtxAdImpressions: DataFrame,
            ctxAdImpressionsToFind: DataFrame,
            nonCtxAdImpressionsToFind: DataFrame,
            visits: DataFrame, phoneRequests: DataFrame, locations: DataFrame, categories: DataFrame,
            splitFracs: Array[Double] = Array(0.7, 0.3), seed: Long = 101L) = {

    import sqlContext.implicits._

    val evaluateSet = //: RDD[(Int, Int, Iterable[Long])] =  // [  (userId, cutoffTime, [mid, mid, ...]), (userId, cutoffTime, [mid, mid, ...]), ... ]
      ctxAdImpressions.
        join(searches.filter("searchTime > 1900800"), ctxAdImpressions("searchId") === searches("id")).
        select(searches("userId"), searches("searchTime"), ctxAdImpressions("mid")).
        map(r => (r.getInt(0), (r.getInt(1), r.getLong(2)))).
        groupByKey().
        map({ case (userId, impressions) =>
          val els = impressions.groupBy(imp => imp._1 / 60).toSeq.map(_._2)
          val el = els(41 * userId % els.size)
          (userId, el.map(_._1).min, el.map(_._2).toSeq)
      })

    val Array(_trainSet, _validateSet) = evaluateSet.randomSplit(splitFracs, seed)

    val evalSet = evaluateSet.flatMap(_._3).repartition(2).toDF("mid").cache()
    val trainSet = _trainSet.flatMap(_._3).repartition(1).toDF("mid").cache()
    val validateSet = _validateSet.flatMap(_._3).repartition(1).toDF("mid").cache()

    // trim historical data about impressions and clicks - as if we ran a batch job on May 12th
    val histCtxAdImpressions =
      ctxAdImpressions.
        join(searches, ctxAdImpressions("searchId") === searches("id")).
        select(searches("userId"), searches("searchTime"), ctxAdImpressions("mid"), ctxAdImpressions("isClick"), ctxAdImpressions("adId")).
        filter("searchTime <= 1900800").
        select(searches("userId"), ctxAdImpressions("mid"), ctxAdImpressions("isClick"), ctxAdImpressions("adId")).
        cache()

    // about Ads
    val ad_imp = histCtxAdImpressions.groupBy("adId").count().
      withColumnRenamed("count", "adImpCount").
      withColumnRenamed("adId", "adImpAdId")

    val ad_click = histCtxAdImpressions.filter("isClick > 0").groupBy("adId").count().
      withColumnRenamed("count", "adClickCount").
      withColumnRenamed("adId", "adClickAdId")

    val ad_imp_click = ad_imp.join(ad_click, ad_imp("adImpAdId") === ad_click("adClickAdId"), "left_outer").
      select("adImpAdId", "adImpCount", "adClickCount")

    val ads_imp_click = ads.join(ad_imp_click, ad_imp_click("adImpAdId") === ads("id"), "left_outer").
      select("id", "category", "params", "price", "title", "adImpCount", "adClickCount")

    //  about Users
    val visitCounts = visits.groupBy("userId").count().
      withColumnRenamed("count", "visitCount").
      withColumnRenamed("userId", "visitUserId")

    val phoneRequestCounts = phoneRequests.groupBy("userId").count().
      withColumnRenamed("count", "phoneCount").
      withColumnRenamed("userId", "phoneUserId")

    val histImpressions = histCtxAdImpressions.groupBy("userId").count().
      withColumnRenamed("count", "impCount").
      withColumnRenamed("userId", "impUserId")

    val histClicks = histCtxAdImpressions.filter("isClick > 0").groupBy("userId").count().
      withColumnRenamed("count", "clickCount").
      withColumnRenamed("userId", "clickUserId")

    val users_visit = users.join(visitCounts, visitCounts("visitUserId") === users("id"), "left_outer").
      select("id", "os", "uafam", "visitCount")

    val users_visit_phone = users_visit.join(phoneRequestCounts, phoneRequestCounts("phoneUserId") === users_visit("id"), "left_outer").
      select("id", "os", "uafam", "visitCount", "phoneCount")

    val users_visit_phone_imp = users_visit_phone.join(histImpressions, histImpressions("impUserId") === users_visit_phone("id"), "left_outer").
      select("id", "os", "uafam", "visitCount", "phoneCount", "impCount")

    val users_visit_phone_imp_click = users_visit_phone_imp.join(histClicks, histClicks("clickUserId") === users_visit_phone_imp("id"), "left_outer").
      select("id", "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount")


    // Search events expanded to include user information
    val searches_users = searches.join(users_visit_phone_imp_click, users_visit_phone_imp_click("id") === searches("userId")).
    select(searches("id"), searches("searchTime"), searches("searchQuery"), searches("searchLoc"), searches("searchCat"), searches("searchParams"),
        searches("loggedIn"), searches("userId"), users_visit_phone_imp_click("os"), users_visit_phone_imp_click("uafam"),
        users_visit_phone_imp_click("visitCount"), users_visit_phone_imp_click("phoneCount"),
        users_visit_phone_imp_click("impCount"), users_visit_phone_imp_click("clickCount")
      ).cache()

    // Eval set
    val ctxAdImpressions_ads_eval = ctxAdImpressions.join(evalSet, evalSet("mid") === ctxAdImpressions("mid")).
      join(ads_imp_click, ads_imp_click("id") === ctxAdImpressions("adId"), "left_outer").
      select("searchId", "adId","position", "histctr", "isClick", "category", "params", "price", "title", "adImpCount", "adClickCount").
      cache()

    val ctxAdImpressions_ads_users_eval = ctxAdImpressions_ads_eval.join(searches_users, searches_users("id") === ctxAdImpressions_ads_eval("searchId"), "left_outer").
      select("isClick",
        "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount",
        "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
        "position", "histctr",
        "category", "params", "price", "title", "adImpCount", "adClickCount" , "searchId", "adId", "userId").
      cache()

    // Small Set
    val ctxAdImpressions_ads_small = ctxAdImpressionsToFind.withColumnRenamed("id", "submid").
      join(ads_imp_click, ads_imp_click("id") === ctxAdImpressionsToFind("adId"), "left_outer").
      select("submid", "searchId", "adId","position", "histctr", "category", "params", "price", "title", "adImpCount", "adClickCount").
      cache()

    val ctxAdImpressions_ads_users_small = ctxAdImpressions_ads_small.join(searches_users, searches_users("id") === ctxAdImpressions_ads_small("searchId"), "left_outer").
      withColumn("isClick", ctxAdImpressions_ads_small("submid")).
      select("isClick",
        "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount",
        "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
        "position", "histctr",
        "category", "params", "price", "title", "adImpCount", "adClickCount", "searchId", "adId", "userId").
    cache()

    //  Train Set
    val ctxAdImpressions_ads_train = ctxAdImpressions.join(trainSet, trainSet("mid") === ctxAdImpressions("mid")).
      join(ads_imp_click, ads_imp_click("id") === ctxAdImpressions("adId"), "left_outer").
      select("searchId", "adId","position", "histctr", "isClick", "category", "params", "price", "title", "adImpCount", "adClickCount").
      cache()

    val ctxAdImpressions_ads_users_train = ctxAdImpressions_ads_train.join(searches_users, searches_users("id") === ctxAdImpressions_ads_train("searchId"), "left_outer")
      .select("isClick",
        "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount",
        "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
        "position", "histctr",
        "category", "params", "price", "title", "adImpCount", "adClickCount", "searchId", "adId", "userId")
    .cache()

    // Validate Set
    val ctxAdImpressions_ads_validate = ctxAdImpressions.join(validateSet, validateSet("mid") === ctxAdImpressions("mid")).
      join(ads_imp_click, ads_imp_click("id") === ctxAdImpressions("adId"), "left_outer").
      select("searchId", "adId","position", "histctr", "isClick", "category", "params", "price", "title", "adImpCount", "adClickCount").
      cache()

    val ctxAdImpressions_ads_users_validate = ctxAdImpressions_ads_validate.join(searches_users, searches_users("id") === ctxAdImpressions_ads_validate("searchId"), "left_outer")
      .select("isClick",
        "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount",
        "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
        "position", "histctr",
        "category", "params", "price", "title", "adImpCount", "adClickCount", "searchId", "adId", "userId")
      .cache()


    val locationsMap = locations.flatMap({
      case Row(id: Int, level: Int, par: Int) => Some((id, (level, par)))
      case _ => None
    }).collectAsMap()
    val bcLocationsMap = sqlContext.sparkContext.broadcast(locationsMap)
    val udf_getLocLevel = udf[Int, Int]( (id: Int) => bcLocationsMap.value.getOrElse(id, (1, -1))._1 )
    val udf_getLocPar = udf[Int, Int]( (id: Int) => bcLocationsMap.value.getOrElse(id, (1, -1))._2 )

    val categoriesMap = categories.flatMap({
      case Row(id: Int, level: Int, par: Int) => Some((id, (level, par)))
      case _ => None
    }).collectAsMap()
    val bcCategoriesMap = sqlContext.sparkContext.broadcast(categoriesMap)
    val udf_getCatLevel = udf[Int, Int]( (id: Int) => bcCategoriesMap.value.getOrElse(id, (1, -1))._1 )
    val udf_getCatPar = udf[Int, Int]( (id: Int) => bcCategoriesMap.value.getOrElse(id, (1, -1))._2 )


    val dataEval = ctxAdImpressions_ads_users_eval.
      withColumn("searchLocLevel", udf_getLocLevel(ctxAdImpressions_ads_users_eval("searchLoc"))).
      withColumn("searchLocPar", udf_getLocPar(ctxAdImpressions_ads_users_eval("searchLoc"))).
      withColumn("searchCatLevel", udf_getCatLevel(ctxAdImpressions_ads_users_eval("searchCat"))).
      withColumn("searchCatPar", udf_getCatPar(ctxAdImpressions_ads_users_eval("searchCat"))).
      withColumn("adCatLevel", udf_getCatLevel(ctxAdImpressions_ads_users_eval("category"))).
      withColumn("adCatPar", udf_getCatPar(ctxAdImpressions_ads_users_eval("category")))

    val dataTrain = ctxAdImpressions_ads_users_train.
    withColumn("searchLocLevel", udf_getLocLevel(ctxAdImpressions_ads_users_train("searchLoc"))).
      withColumn("searchLocPar", udf_getLocPar(ctxAdImpressions_ads_users_train("searchLoc"))).
      withColumn("searchCatLevel", udf_getCatLevel(ctxAdImpressions_ads_users_train("searchCat"))).
      withColumn("searchCatPar", udf_getCatPar(ctxAdImpressions_ads_users_train("searchCat"))).
      withColumn("adCatLevel", udf_getCatLevel(ctxAdImpressions_ads_users_train("category"))).
      withColumn("adCatPar", udf_getCatPar(ctxAdImpressions_ads_users_train("category")))

    val dataValidate = ctxAdImpressions_ads_users_validate.
    withColumn("searchLocLevel", udf_getLocLevel(ctxAdImpressions_ads_users_validate("searchLoc"))).
      withColumn("searchLocPar", udf_getLocPar(ctxAdImpressions_ads_users_validate("searchLoc"))).
      withColumn("searchCatLevel", udf_getCatLevel(ctxAdImpressions_ads_users_validate("searchCat"))).
      withColumn("searchCatPar", udf_getCatPar(ctxAdImpressions_ads_users_validate("searchCat"))).
      withColumn("adCatLevel", udf_getCatLevel(ctxAdImpressions_ads_users_validate("category"))).
      withColumn("adCatPar", udf_getCatPar(ctxAdImpressions_ads_users_validate("category")))

    val dataSmall = ctxAdImpressions_ads_users_small.
    withColumn("searchLocLevel", udf_getLocLevel(ctxAdImpressions_ads_users_small("searchLoc"))).
      withColumn("searchLocPar", udf_getLocPar(ctxAdImpressions_ads_users_small("searchLoc"))).
      withColumn("searchCatLevel", udf_getCatLevel(ctxAdImpressions_ads_users_small("searchCat"))).
      withColumn("searchCatPar", udf_getCatPar(ctxAdImpressions_ads_users_small("searchCat"))).
      withColumn("adCatLevel", udf_getCatLevel(ctxAdImpressions_ads_users_small("category"))).
      withColumn("adCatPar", udf_getCatPar(ctxAdImpressions_ads_users_small("category")))

    (dataEval, dataTrain, dataValidate, dataSmall)
  }

}
