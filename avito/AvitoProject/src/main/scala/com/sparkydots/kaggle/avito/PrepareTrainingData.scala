package com.sparkydots.kaggle.avito

import org.apache.spark.sql.{DataFrame, SQLContext}

object PrepareTrainingData {

  def split(sqlContext: SQLContext,
            users: DataFrame,
            ads: DataFrame, ctxAds: DataFrame, nonCtxAds: DataFrame,
            searches: DataFrame,
            ctxAdImpressions: DataFrame, ctxAdImpressionsToFind: DataFrame,
            vists: DataFrame, phoneRequests: DataFrame,
            splitFracs: Array[Double] = Array(0.65, 0.25, 0.10)) = {

    import sqlContext.implicits._

    val evaluateSet = //: RDD[(Int, Int, Iterable[Long])] =  // [  (userId, cutoffTime, [mid, mid, ...]), (userId, cutoffTime, [mid, mid, ...]), ... ]
      ctxAdImpressions.
        join(searches.filter("searchTime > 1900800"), ctxAdImpressions("searchId") === searches("id")).
        select(searches("userId"), searches("searchTime"), ctxAdImpressions("histctr"), ctxAdImpressions("isClick"), ctxAdImpressions("mid")).
        map(r => (r.getInt(0), (r.getInt(1), r.getDouble(2), r.getInt(3), r.getLong(4)))).
        groupByKey().
        map({ case (userId, impressions) =>
        val els = impressions.groupBy(imp => imp._1 / 60).toSeq.map(_._2)
        val el = els(41 * userId % els.size)
        (userId, el.map(_._1).min, el.map(_._4).toSeq)
      })

    val Array(_trainSet, _validateSet, _testSet) = evaluateSet.randomSplit(splitFracs)

    val trainSet = _trainSet.flatMap(_._3).toDF("mid").cache()
    val validateSet = _validateSet.flatMap(_._3).toDF("mid").cache()
    val testSet = _testSet.flatMap(_._3).toDF("mid").cache()

    // trim historical data
    val cutoffTimes = evaluateSet.map(r => (r._1, r._2)).toDF("userId", "cutoffTime")

    val histCtxAdImpressions =
      ctxAdImpressions.
        join(searches, ctxAdImpressions("searchId") === searches("id")).
        select(searches("userId"), searches("searchTime"), ctxAdImpressions("mid")).
        join(cutoffTimes, cutoffTimes("userId") === searches("userId")).
        filter(searches("searchTime") < cutoffTimes("cutoffTime")).
        select(ctxAdImpressions("mid")).
        cache()

    (trainSet, validateSet, testSet, histCtxAdImpressions)

  }

  def calcErrors(ctxAdImpressions: DataFrame, trainSet: DataFrame, validateSet: DataFrame, testSet: DataFrame) = {

    println("calculating errors")

    import com.sparkydots.kaggle.avito.functions.DFFunctions.calcError

    val trainError = calcError(ctxAdImpressions.
      join(trainSet, trainSet("mid") === ctxAdImpressions("mid")).
      select(ctxAdImpressions("histctr").as("pred"), ctxAdImpressions("isClick")))

    val validateError = calcError(ctxAdImpressions.
      join(validateSet, validateSet("mid") === ctxAdImpressions("mid")).
      select(ctxAdImpressions("histctr").as("pred"), ctxAdImpressions("isClick")))

    val testError = calcError(ctxAdImpressions.
      join(testSet, testSet("mid") === ctxAdImpressions("mid")).
      select(ctxAdImpressions("histctr").as("pred"), ctxAdImpressions("isClick")))

    println(s"Errors:\nTrain\tValidate\tTest\n${trainError}\t${validateError}\t${testError}")

  }

}
