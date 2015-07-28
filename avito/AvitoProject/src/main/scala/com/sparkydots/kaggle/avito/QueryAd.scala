package com.sparkydots.kaggle.avito

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{SQLContext, DataFrame}
import com.sparkydots.kaggle.avito.functions.Functions._
import com.sparkydots.kaggle.avito.load.LoadSave
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._


object QueryAd {

  // val (newRawTrain, newRawValidate, newRawEval, newRawSmall) = QueryAd.addQueryTitleAffinity(sqlContext, rawTrain, rawValidate, rawEval, rawSmall)
  def addQueryTitleAffinity(sqlContext: SQLContext,
                            rawTrain: DataFrame, rawValidate: DataFrame,
                            rawEval: DataFrame, rawSmall: DataFrame,
                            rank: Int = 10, numIterations: Int = 20): (DataFrame, DataFrame, DataFrame, DataFrame) = {
    val wordsDict = sqlContext.sparkContext.broadcast(LoadSave.loadDF(sqlContext, "words50").map({
      case Row(word: String, wordId: Int) => word -> wordId
    }).collect.toMap)

    val (medRawTrain, medRawValidate) = addQueryTitleAffinity(sqlContext, wordsDict, 1, "queryTitlePos", rawTrain, rawValidate, rank, numIterations)
    val (medRawEval, medRawSmall) = addQueryTitleAffinity(sqlContext, wordsDict, 1, "queryTitlePos", rawEval, rawSmall, rank, numIterations)

    val (newRawTrain, newRawValidate) = addQueryTitleAffinity(sqlContext, wordsDict, 0, "queryTitleNeg", medRawTrain, medRawValidate, rank, numIterations)
    val (newRawEval, newRawSmall) = addQueryTitleAffinity(sqlContext, wordsDict, 0, "queryTitleNeg", medRawEval, medRawSmall, rank, numIterations)

    (newRawTrain, newRawValidate, newRawEval, newRawSmall)
  }

  def addQueryTitleAffinity(sqlContext: SQLContext, wordsDict: Broadcast[Map[String, Int]],
                            isClick: Int, colName: String,
                            rawTrain: DataFrame, rawValidate: DataFrame,
                            rank: Int, numIterations: Int): (DataFrame, DataFrame) = {
    import sqlContext.implicits._

    val queryTitle = rawTrain.
      filter(s"isClick = $isClick").
      select("searchQuery", "title").
      flatMap { r =>
      splitStringWithCutoff(r.getString(0), 2).flatMap { wq =>
        splitStringWithCutoff(r.getString(1), 2).flatMap { wt =>
          val wqi = wordsDict.value.get(stemString(wq))
          val wti = wordsDict.value.get(stemString(wt))
          if (wqi.isEmpty || wti.isEmpty) {
            None
          } else {
          Some(((wqi.get, wti.get), 1))
          }
        }
      }
    }.reduceByKey(_ + _).
      map { case ((wqi, wti), cnt) => Rating(wqi, wti, cnt) }.cache()

    val model = ALS.train(queryTitle, rank, numIterations, 0.01)

//    val qa = queryTitle.toDF()

//    val qterms = qa.select("user").distinct
//    val aterms = qa.select("product").distinct

//    val bc_qterms = sqlContext.sparkContext.broadcast(qterms.map(_.getInt(0)).collect().toSet)
//    val bc_saterms = sqlContext.sparkContext.broadcast(aterms.map(_.getInt(0)).collect().toSet)

    val users = model.userFeatures.map(_._1).distinct().toDF()
    val products = model.productFeatures.map(_._1).distinct().toDF()

    val bc_qterms = sqlContext.sparkContext.broadcast(users.map(r => r.getInt(0)).collect().toSet)
    val bc_aterms = sqlContext.sparkContext.broadcast(products.map(r => r.getInt(0)).collect().toSet)

    val pairs = users.join(products).map(r => (r.getInt(0), r.getInt(1)))
    val bc_preds = sqlContext.sparkContext.broadcast(model.predict(pairs).map { case Rating(user, product, rating) => (user*10000 + product, rating) }.collectAsMap())

    val calculatePos = udf[Double, String, String](
      (qs: String, ts: String) => {
        val qis =  splitStringWithCutoff(qs, 2).flatMap(w => wordsDict.value.get(stemString(w)))
        val tis =  splitStringWithCutoff(ts, 2).flatMap(w => wordsDict.value.get(stemString(w)))
        (for {
          qi <- qis if bc_qterms.value.contains(qi)
          ti <- tis if bc_aterms.value.contains(ti)
          r <- bc_preds.value.get(qi * 10000 + ti)
        } yield r).sum
      }
    )

    val newRawTrain = rawTrain.withColumn(colName, calculatePos(col("searchQuery"), col("title")))
    val newRawValidate = rawValidate.withColumn(colName, calculatePos(col("searchQuery"), col("title")))
    (newRawTrain, newRawValidate)
  }

}
