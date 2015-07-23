package com.sparkydots.kaggle.avito.functions

import com.sparkydots.kaggle.avito.functions.Functions._
import org.apache.spark.sql.functions._

object UdfFunctions extends Serializable {

  val udf_toInt = udf[Int, String](toInt)
  val udf_toIntOrMinus = udf[Int, String](toIntOrMinus)
  val udf_toDoubleOrMinus = udf[Double, String](toDoubleOrMinus)

  val udf_length = udf[Int, String](length)
  val udf_toLower = udf[String, String](toLower)

  val udf_toMid = udf[Long, String, String](toMid)

  val udf_parseTime = udf[Int, String](parseTime)

  val udf_parseParams = udf[Seq[Int], String](parseParams)

  val udf_error = udf[Double, Double, Double](error)

}

