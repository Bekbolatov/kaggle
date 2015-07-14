package com.sparkydots.kaggle.avito

import scala.util.Try

object Functions extends Serializable {


  val _toInt = (text: String) => text.toInt

  val _toIntOrMinus = (s: String) => Try(s.toInt).getOrElse(-1)

  val _toDoubleOrMinus = (s: String) => Try(s.toDouble).getOrElse(-1.0)

  val _length = (text: String) => text.length

  val _toLower = (text: String) => text.toLowerCase

  val _parseTime = (tt: String) => {
    val format = new java.text.SimpleDateFormat("yyyy-MM-dd hh:mm:ss.0")
    format.parse(tt).getTime / 1000
  }

  val _parseParams = (tt: String) => {
    if (tt.isEmpty) {
      ""
    } else {
      val terms = tt.substring(1, tt.length - 1).replaceAll("'[^']+'","XX").split(", ")
      terms.map(x => x.split(":")(0)).mkString(",")
    }
  }
}

object UdfFunctions extends Serializable {
  import org.apache.spark.sql.functions._
  import Functions._

  val toInt = udf[Int, String](_toInt)
  val toIntOrMinus = udf[Int, String](_toIntOrMinus)
  val toDoubleOrMinus = udf[Double, String](_toDoubleOrMinus)
  val length = udf[Int, String](_length)
  val toLower = udf[String, String](_toLower)
  val parseTime = udf(_parseTime)
  val parseParams = udf(_parseParams)

}

