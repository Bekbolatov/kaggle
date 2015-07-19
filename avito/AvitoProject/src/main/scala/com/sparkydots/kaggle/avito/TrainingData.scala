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
            splitFracs: Array[Double] = Array(0.65, 0.25, 0.10), seed: Long = 101L) = {

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

    val Array(_trainSet, _validateSet, _testSet) = evaluateSet.randomSplit(splitFracs, seed)

    val trainSet = _trainSet.flatMap(_._3).repartition(1).toDF("mid").cache()
    val validateSet = _validateSet.flatMap(_._3).repartition(1).toDF("mid").cache()
    val testSet = _testSet.flatMap(_._3).repartition(1).toDF("mid").cache()

    // trim historical data
    val cutoffTimes = evaluateSet.map(r => (r._1, r._2)).toDF("userId", "cutoffTime")

    val histCtxAdImpressions =
      ctxAdImpressions.
        join(searches, ctxAdImpressions("searchId") === searches("id")).
        select(searches("userId"), searches("searchTime"), ctxAdImpressions("mid")).
        join(cutoffTimes, cutoffTimes("userId") === searches("userId")).
        filter(searches("searchTime") < cutoffTimes("cutoffTime")).
        select(ctxAdImpressions("mid")).
        repartition(24).
        cache()


    //  Prepate data sets
    val users_hist = users.repartition(24)

    val searches_users = searches.join(users_hist, users_hist("id") === searches("userId"))
    .select(searches("id"), searches("searchTime"), searches("searchQuery"), searches("searchLoc"), searches("searchCat"), searches("searchParams"),
        searches("loggedIn"), users_hist("os"), users_hist("uafam"))
    .repartition(24)

    //  Train Set
    val ctxAdImpressions_ads_train = ctxAdImpressions.join(trainSet, trainSet("mid") === ctxAdImpressions("mid"))
      .join(ads, ads("id") === ctxAdImpressions("adId"))
      .select("searchId", "adId","position", "histctr", "isClick", "category", "params", "price", "title")
      .repartition(24)

    val ctxAdImpressions_ads_users_train = ctxAdImpressions_ads_train.join(searches_users, searches_users("id") === ctxAdImpressions_ads_train("searchId"))
      .select("isClick",
        "os", "uafam",
        "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
        "position", "histctr",
        "category", "params", "price", "title")
    .cache()

    // Validate Set
    val ctxAdImpressions_ads_validate = ctxAdImpressions.join(validateSet, validateSet("mid") === ctxAdImpressions("mid"))
      .join(ads, ads("id") === ctxAdImpressions("adId"))
      .select("searchId", "adId","position", "histctr", "isClick", "category", "params", "price", "title")
      .repartition(24).cache()

    val ctxAdImpressions_ads_users_validate = ctxAdImpressions_ads_validate.join(searches_users, searches_users("id") === ctxAdImpressions_ads_validate("searchId"))
      .select("isClick",
        "os", "uafam",
        "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
        "position", "histctr",
        "category", "params", "price", "title")
      .cache()

    // Test Set
    val ctxAdImpressions_ads_test = ctxAdImpressions.join(testSet, testSet("mid") === ctxAdImpressions("mid"))
      .join(ads, ads("id") === ctxAdImpressions("adId"))
      .select("searchId", "adId","position", "histctr", "isClick", "category", "params", "price", "title")
      .repartition(24).cache()

    val ctxAdImpressions_ads_users_test = ctxAdImpressions_ads_test.join(searches_users, searches_users("id") === ctxAdImpressions_ads_test("searchId"))
      .select("isClick",
        "os", "uafam",
        "searchTime", "searchQuery", "searchLoc", "searchCat", "searchParams", "loggedIn",
        "position", "histctr",
        "category", "params", "price", "title")
      .cache()

//    saveDF(sqlContext, ctxAdImpressions_ads_users_train, "DATA_TRAIN_1")
//    saveDF(sqlContext, ctxAdImpressions_ads_users_validate, "DATA_VALIDATE_1")
//    saveDF(sqlContext, ctxAdImpressions_ads_users_test, "DATA_TEST_1")


    (ctxAdImpressions_ads_users_train, ctxAdImpressions_ads_users_validate, ctxAdImpressions_ads_users_test)
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
