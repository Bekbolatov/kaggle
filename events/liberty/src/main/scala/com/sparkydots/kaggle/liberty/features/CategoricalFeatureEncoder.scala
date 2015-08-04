package com.sparkydots.kaggle.liberty.features

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, StringType}

class CategoricalFeatureEncoder(universeDF: DataFrame, colname: String) extends Serializable {

  lazy val size = dictionary.size
  def apply(value: Any) = dictionary(value)

  private val dictionary = generateDictionary(colname)

  private def generateDictionary(colname: String): Map[Any, Int] = {
    universeDF.schema(colname).dataType match {
      case StringType =>
        universeDF.select(colname).distinct.orderBy(colname).map(_.getString(0)).collect().zipWithIndex.toMap
      case IntegerType =>
        val (smallest, largest) = universeDF.select(min(colname), max(colname)).map(r => (r.getInt(0), r.getInt(1))).collect()(0)
        (smallest to largest).map { n => (n, n - smallest) }.toMap
      case _ => Map.empty[Any, Int]
    }
  }

}
