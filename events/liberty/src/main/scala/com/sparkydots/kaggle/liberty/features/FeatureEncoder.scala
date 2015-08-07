package com.sparkydots.kaggle.liberty.features

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StringType}

trait FeatureEncoder[T] extends Serializable {

  def size: Int

  def apply(value: T): (Int, Double)

}

class IntFeatureEncoder extends FeatureEncoder[Int] {

  def size: Int = 1

  def apply(value: Int): (Int, Double) = (0, value.toDouble)

}

class CategoricalFeatureEncoder(universeDF: DataFrame, colname: String) extends FeatureEncoder[String] {

  def size: Int = dictionary.size

  def apply(value: String): (Int, Double) = (dictionary(value), 1.0)

  private lazy val dictionary = universeDF.select(colname).distinct.orderBy(colname).map(_.getString(0)).collect().zipWithIndex.toMap

}
