package com.sparkydots.kaggle.liberty.features

import org.apache.spark.sql.DataFrame

trait FeatureEncoder[T] extends Serializable {

  def size: Int

  def apply(value: T): (Int, Double)

  def vocabSize: Int

}

class IntFeatureEncoder extends FeatureEncoder[Int] {

  def size: Int = 1

  def apply(value: Int): (Int, Double) = (0, value.toDouble)

  def vocabSize: Int = -1

}

class CategoricalFeatureEncoder(universeDF: DataFrame, colname: String) extends FeatureEncoder[String] {

  def size: Int = 1

  def apply(value: String): (Int, Double) = (0, dictionary(value).toDouble)

  def vocabSize: Int = dictionary.size

  private val dictionary = universeDF.select(colname).distinct.orderBy(colname).map(_.getString(0)).collect().zipWithIndex.toMap

}

class CategoricalFeatureOHEEncoder(universeDF: DataFrame, colname: String) extends FeatureEncoder[String] {

  def size: Int = dictionary.size

  def apply(value: String): (Int, Double) = (dictionary(value), 1.0)

  def vocabSize: Int = dictionary.size

  private val dictionary = universeDF.select(colname).distinct.orderBy(colname).map(_.getString(0)).collect().zipWithIndex.toMap

}

