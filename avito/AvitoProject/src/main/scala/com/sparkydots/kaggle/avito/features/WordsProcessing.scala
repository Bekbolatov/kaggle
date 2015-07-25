package com.sparkydots.kaggle.avito.features

import com.sparkydots.kaggle.avito.functions.Functions.{splitString, stemString}
import com.sparkydots.kaggle.avito.load.LoadSave
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

object WordsProcessing extends Serializable {

  def generateAndSaveWordDictionaries(sc: SparkContext, sqlContext: SQLContext,
                                      rawEval: DataFrame, rawSmall: DataFrame,
                                      filename: String = "words",
                                      thresholds: Seq[Int] = Seq(100, 500, 1000, 5000, 10000, 20000)) = {
    import sqlContext.implicits._

    val counts11 = rawEval.select("title").flatMap({
      case Row(title: String) => splitString(title).map(x => (stemString(x), 1))
      case _ => Seq()
    }).reduceByKey((x, y) => x + y)

    val counts12 = rawEval.select("searchQuery").flatMap({
      case Row(title: String) => splitString(title).map(x => (stemString(x), 1))
      case _ => Seq()
    }).reduceByKey((x, y) => x + y)

    val counts21 = rawSmall.select("title").flatMap({
      case Row(title: String) => splitString(title).map(x => (stemString(x), 1))
      case _ => Seq()
    }).reduceByKey((x, y) => x + y)

    val counts22 = rawSmall.select("searchQuery").flatMap({
      case Row(title: String) => splitString(title).map(x => (stemString(x), 1))
      case _ => Seq()
    }).reduceByKey((x, y) => x + y)

    val counts = counts11.union(counts12).union(counts21).union(counts22).reduceByKey((x, y) => x + y).toDF("word", "cnt").orderBy("cnt").cache

    thresholds.foreach { threshold =>
      val words = counts.filter(s"word != 'и' and word != 'в' and word != 'с' and word != 'для' and cnt > $threshold").select("word").collect().map(x => x.getString(0))
      val wordsDict = sc.parallelize(words.zipWithIndex).toDF("word", "wordId").repartition(1)
      LoadSave.saveDF(sqlContext, wordsDict, s"$filename$threshold")
    }
  }

  def generateAndSaveParamDictionaries(sc: SparkContext, sqlContext: SQLContext, rawEval: DataFrame, rawSmall: DataFrame, filename: String = "words") = {
    import sqlContext.implicits._

    val counts11 = rawEval.select("params").flatMap({
      case Row(params: Seq[Int]) => params.map(x => (x, 1))
      case _ => Seq()
  }).reduceByKey((x, y) => x + y)

    val counts12 = rawEval.select("searchParams").flatMap({
      case Row(params: Seq[Int]) => params.map(x => (x, 1))
      case _ => Seq()
    }).reduceByKey((x, y) => x + y)

    val counts21 = rawSmall.select("params").flatMap({
      case Row(params: Seq[Int]) => params.map(x => (x, 1))
      case _ => Seq()
    }).reduceByKey((x, y) => x + y)

    val counts22 = rawSmall.select("searchParams").flatMap({
      case Row(params: Seq[Int]) => params.map(x => (x, 1))
      case _ => Seq()
    }).reduceByKey((x, y) => x + y)

    val counts = counts11.union(counts12).union(counts21).union(counts22).reduceByKey((x, y) => x + y).toDF("param", "cnt").orderBy("cnt").cache
    val countsAll = counts.select("param").collect().map(x => x.getInt(0))

    val countsAllDict = sc.parallelize(countsAll.zipWithIndex).toDF("param", "paramId").repartition(1)
    LoadSave.saveDF(sqlContext, countsAllDict, s"${filename}1000")

  }

}
