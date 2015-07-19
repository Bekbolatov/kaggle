package com.sparkydots.kaggle.avito

import org.apache.spark.sql.{DataFrame, SQLContext}
import com.sparkydots.kaggle.avito.LoadSave.saveDF

object TrainingData {

  def split(sqlContext: SQLContext,
            users: DataFrame,
            ads: DataFrame, ctxAds: DataFrame, nonCtxAds: DataFrame,
            searches: DataFrame,
            ctxAdImpressions: DataFrame, ctxAdImpressionsToFind: DataFrame,
            visits: DataFrame, phoneRequests: DataFrame,
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

    // trim historical data
    val cutoffTimes = evaluateSet.map(r => (r._1, r._2)).toDF("userId", "cutoffTime")

    val histCtxAdImpressions =
      ctxAdImpressions.
        join(searches, ctxAdImpressions("searchId") === searches("id")).
        select(searches("userId"), searches("searchTime"), ctxAdImpressions("mid"), ctxAdImpressions("isClick"), ctxAdImpressions("adId")).
        join(cutoffTimes, cutoffTimes("userId") === searches("userId")).
        filter(searches("searchTime") < cutoffTimes("cutoffTime")).
        select(searches("userId"), ctxAdImpressions("mid"), ctxAdImpressions("isClick"), ctxAdImpressions("adId")).
        repartition(20).
        cache()

    val ad_imp = histCtxAdImpressions.groupBy("adId").count().
      withColumnRenamed("count", "adImpCount").
      withColumnRenamed("adId", "adImpAdId").
      repartition(20)

    val ad_click = histCtxAdImpressions.filter("isClick > 0").groupBy("adId").count().
      withColumnRenamed("count", "adClickCount").
      withColumnRenamed("adId", "adClickAdId").
      repartition(20)

    val ad_imp_click = ad_imp.join(ad_click, ad_imp("adImpAdId") === ad_click("adClickAdId"), "left_outer").
      select("adImpAdId", "adImpCount", "adClickCount").
      repartition(20)

    val ads_imp_click = ads.join(ad_imp_click, ad_imp_click("adImpAdId") === ads("id"), "left_outer").
    select("id", "category", "params", "price", "title", "adImpCount", "adClickCount")


    //  Prepate data sets

    val visitCounts = visits.groupBy("userId").count().
      withColumnRenamed("count", "visitCount").
      withColumnRenamed("userId", "visitUserId").
      repartition(24)

    val phoneRequestCounts = phoneRequests.groupBy("userId").count().
      withColumnRenamed("count", "phoneCount").
      withColumnRenamed("userId", "phoneUserId").
      repartition(24)

    val histImpressions = histCtxAdImpressions.groupBy("userId").count().
      withColumnRenamed("count", "impCount").
      withColumnRenamed("userId", "impUserId").
      repartition(20)

    val histClicks = histCtxAdImpressions.filter("isClick > 0").groupBy("userId").count().
      withColumnRenamed("count", "clickCount").
      withColumnRenamed("userId", "clickUserId").
      repartition(20)

    val users_visit = users.join(visitCounts, visitCounts("visitUserId") === users("id"), "left_outer").
      select("id", "os", "uafam", "visitCount").
      repartition(24)

    val users_visit_phone = users_visit.join(phoneRequestCounts, phoneRequestCounts("phoneUserId") === users_visit("id"), "left_outer").
      select("id", "os", "uafam", "visitCount", "phoneCount").
      repartition(24)

    val users_visit_phone_imp = users_visit_phone.join(histImpressions, histImpressions("impUserId") === users_visit_phone("id"), "left_outer").
      select("id", "os", "uafam", "visitCount", "phoneCount", "impCount").
      repartition(24)

    val users_visit_phone_imp_click = users_visit_phone_imp.join(histClicks, histClicks("clickUserId") === users_visit_phone_imp("id"), "left_outer").
      select("id", "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount").
      repartition(24)




    val searches_users = searches.join(users_visit_phone_imp_click, users_visit_phone_imp_click("id") === searches("userId")).
    select(searches("id"), searches("searchTime"), searches("searchQuery"), searches("searchLoc"), searches("searchCat"), searches("searchParams"),
        searches("loggedIn"), users_visit_phone_imp_click("os"), users_visit_phone_imp_click("uafam"),
        users_visit_phone_imp_click("visitCount"), users_visit_phone_imp_click("phoneCount"),
        users_visit_phone_imp_click("impCount"), users_visit_phone_imp_click("clickCount")
      ).repartition(24)

    // Eval set
    val ctxAdImpressions_ads_eval = ctxAdImpressions.join(evalSet, evalSet("mid") === ctxAdImpressions("mid"))
      .join(ads_imp_click, ads_imp_click("id") === ctxAdImpressions("adId"), "left_outer")
      .select("searchId", "adId","position", "histctr", "isClick", "category", "params", "price", "title", "adImpCount", "adClickCount")
      .repartition(24)

    val ctxAdImpressions_ads_users_eval = ctxAdImpressions_ads_eval.join(searches_users, searches_users("id") === ctxAdImpressions_ads_eval("searchId"), "left_outer")
      .select("isClick",
        "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount",
        "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
        "position", "histctr",
        "category", "params", "price", "title", "adImpCount", "adClickCount")
      .cache()

    // Small Set
    val ctxAdImpressions_ads_small = ctxAdImpressionsToFind.withColumnRenamed("id", "submid").
      join(ads_imp_click, ads_imp_click("id") === ctxAdImpressionsToFind("adId"), "left_outer").
      select("submid", "searchId", "adId","position", "histctr", "category", "params", "price", "title", "adImpCount", "adClickCount").
      repartition(24)

    val ctxAdImpressions_ads_users_small = ctxAdImpressions_ads_small.join(searches_users, searches_users("id") === ctxAdImpressions_ads_small("searchId"), "left_outer").
      withColumn("isClick", ctxAdImpressions_ads_small("submid")).
      select("isClick",
        "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount",
        "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
        "position", "histctr",
        "category", "params", "price", "title", "adImpCount", "adClickCount").cache()

    //  Train Set
    val ctxAdImpressions_ads_train = ctxAdImpressions.join(trainSet, trainSet("mid") === ctxAdImpressions("mid"))
      .join(ads_imp_click, ads_imp_click("id") === ctxAdImpressions("adId"), "left_outer")
      .select("searchId", "adId","position", "histctr", "isClick", "category", "params", "price", "title", "adImpCount", "adClickCount")
      .repartition(24)

    val ctxAdImpressions_ads_users_train = ctxAdImpressions_ads_train.join(searches_users, searches_users("id") === ctxAdImpressions_ads_train("searchId"), "left_outer")
      .select("isClick",
        "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount",
        "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
        "position", "histctr",
        "category", "params", "price", "title", "adImpCount", "adClickCount")
    .cache()

    // Validate Set
    val ctxAdImpressions_ads_validate = ctxAdImpressions.join(validateSet, validateSet("mid") === ctxAdImpressions("mid"))
      .join(ads_imp_click, ads_imp_click("id") === ctxAdImpressions("adId"), "left_outer")
      .select("searchId", "adId","position", "histctr", "isClick", "category", "params", "price", "title", "adImpCount", "adClickCount")
      .repartition(24).cache()

    val ctxAdImpressions_ads_users_validate = ctxAdImpressions_ads_validate.join(searches_users, searches_users("id") === ctxAdImpressions_ads_validate("searchId"), "left_outer")
      .select("isClick",
        "os", "uafam", "visitCount", "phoneCount", "impCount", "clickCount",
        "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
        "position", "histctr",
        "category", "params", "price", "title", "adImpCount", "adClickCount")
      .cache()

    (ctxAdImpressions_ads_users_eval, ctxAdImpressions_ads_users_train, ctxAdImpressions_ads_users_validate, ctxAdImpressions_ads_users_small)
  }

    // TrainingData.calcErrors(ctxAdImpressions, trainSet, validateSet, testSet)
  def calcErrors(ctxAdImpressions: DataFrame, trainSet: DataFrame, validateSet: DataFrame, testSet: DataFrame) = {

    println("calculating errors")

    import com.sparkydots.kaggle.avito.functions.DFFunctions.calcError

    println(calcError(ctxAdImpressions.
      join(trainSet, trainSet("mid") === ctxAdImpressions("mid")).
      select(ctxAdImpressions("histctr"), ctxAdImpressions("isClick"))))

      println(calcError(ctxAdImpressions.
      join(validateSet, validateSet("mid") === ctxAdImpressions("mid")).
      select(ctxAdImpressions("histctr"), ctxAdImpressions("isClick"))))

        println(calcError(ctxAdImpressions.
      join(testSet, testSet("mid") === ctxAdImpressions("mid")).
      select(ctxAdImpressions("histctr"), ctxAdImpressions("isClick"))))

  }

}
