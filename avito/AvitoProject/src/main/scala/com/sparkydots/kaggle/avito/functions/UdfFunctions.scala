package com.sparkydots.kaggle.avito.functions

import com.sparkydots.kaggle.avito.functions.Functions._
import org.apache.spark.sql.functions._

object UdfFunctions extends Serializable {

  val toInt = udf[Int, String](_toInt)
  val toIntOrMinus = udf[Int, String](_toIntOrMinus)
  val toDoubleOrMinus = udf[Double, String](_toDoubleOrMinus)

  val length = udf[Int, String](_length)
  val toLower = udf[String, String](_toLower)

  val toMid = udf[Long, String, String](_toMid)

  val parseTime = udf[Int, String](_parseTime)

  val parseParams = udf[Seq[Int], String](_parseParams)

  val error = udf[Double, Double, Double](_error)

}

