package com.sparkydots.kaggle.avito.features

import com.sparkydots.kaggle.avito.functions.Functions.{splitString, stemString}
import com.sparkydots.kaggle.avito.load.LoadSave
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

object WordsProcessing extends Serializable {

  def generateAndSaveWordDictionaries(sc: SparkContext, sqlContext: SQLContext, rawEval: DataFrame, rawSmall: DataFrame, filename: String = "words") = {
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

    val words20000 = counts.filter("word != 'и' and word != 'в' and word != 'с' and word != 'для' and cnt > 20000").select("word").collect().map(x => x.getString(0))
    val words10000 = counts.filter("word != 'и' and word != 'в' and word != 'с' and word != 'для' and cnt > 10000").select("word").collect().map(x => x.getString(0))
    val words5000 = counts.filter("word != 'и' and word != 'в' and word != 'с' and word != 'для' and cnt > 5000").select("word").collect().map(x => x.getString(0))

    val wordsDict20000 = sc.parallelize(words20000.zipWithIndex).toDF("word", "wordId").repartition(1)
    val wordsDict10000 = sc.parallelize(words10000.zipWithIndex).toDF("word", "wordId").repartition(1)
    val wordsDict5000 = sc.parallelize(words5000.zipWithIndex).toDF("word", "wordId").repartition(1)

    LoadSave.saveDF(sqlContext, wordsDict20000, s"${filename}20000")
    LoadSave.saveDF(sqlContext, wordsDict10000, s"${filename}10000")
    LoadSave.saveDF(sqlContext, wordsDict5000, s"${filename}5000")

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
