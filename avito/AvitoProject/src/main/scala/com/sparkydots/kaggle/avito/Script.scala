package com.sparkydots.kaggle.avito

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext


object Script {

  def info = {
    println("spark-shell --jars AvitoProject-assembly-1.0.jar")
    println("val (users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests, trainSet, validateSet, testSet, histCtxAdImpressions) = com.sparkydots.kaggle.avito.Script.run(sc, orig = ?)")

    //sqlContext.udf.register("strLen", (s: String) => s.length())
    //sqlContext.udf.register("errf", _error)

  }

  def run(sc: SparkContext, orig: Boolean = false) = {

    Logger.getLogger("amazon.emr.metrics").setLevel(Level.OFF)
    Logger.getLogger("com.amazon.ws.emr").setLevel(Level.ERROR)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val (users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests) =
      if (orig)
        LoadSave.origLoad(sqlContext)
      else
        LoadSave.load(sqlContext)

    val (trainSet, validateSet, testSet, histCtxAdImpressions) =
      PrepareTrainingData.split(sqlContext, users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests)

    PrepareTrainingData.calcErrors(ctxAdImpressions, trainSet, validateSet, testSet)

    (users, ads, ctxAds, nonCtxAds, searches, ctxAdImpressions, ctxAdImpressionsToFind, visits, phoneRequests, trainSet, validateSet, testSet, histCtxAdImpressions)
  }
}
