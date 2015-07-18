package com.sparkydots.kaggle.avito

import org.joda.time.DateTime

import scala.util.Try

object Functions extends Serializable {

  //  Time and Date functions //
  val _APR20 = new java.text.SimpleDateFormat("yyyy-MM-dd hh:mm:ss.0").parse("2015-04-20 00:00:00.0").getTime
  val _SECONDS_IN_HOUR = 60*60
  val _SECONDS_IN_DAY = _SECONDS_IN_HOUR * 24
  val _SECONDS_IN_WEEK = _SECONDS_IN_DAY * 7

  val _parseTime = (tt: String) => {
    val format = new java.text.SimpleDateFormat("yyyy-MM-dd hh:mm:ss.0")
    ((format.parse(tt).getTime - _APR20)/1000).toInt
  }

  /**
   * 0: Monday
   * 1: Tuesday
   * ...
   * 6: Sunday
   */
  val _dayOfWeek = (t: Int) => t/_SECONDS_IN_DAY % 7

  /**
   * 0, 1, ..., 23
   */
  val _hourOfDay = (t: Int) => t/_SECONDS_IN_HOUR % 24


  // Other //
  val _toInt = (text: String) => text.toInt

  val _toIntOrMinus = (s: String) => Try(s.toInt).getOrElse(-1)

  val _toDoubleOrMinus = (s: String) => Try(s.toDouble).getOrElse(-1.0)

  val _length = (text: String) => text.length

  val _toLower = (text: String) => text.toLowerCase

  val _toMid = (searchId: String, adId: String) => 10000000L * searchId.toInt + adId.toInt

  val _parseParams = (tt: String) => {
    if (tt.isEmpty) {
      ""
    } else {
      val terms = tt.substring(1, tt.length - 1)
        .replaceAll("\\:\\{[^}]+'}",":'XX'")
        .replaceAll("'[^']+'","'XX'")
        .split(", ")
      terms.map(x => x.split(":")(0)).mkString(",")
    }
  }

  val _error = (guess: Double, actual: Int) => {
    - math.log(math.max(if (actual == 0) 1 - guess else guess, 1e-15))
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
  val toMid = udf[Long, String, String](_toMid)
  val parseTime = udf(_parseTime)
  val parseParams = udf(_parseParams)
  val error = udf[Double, Double, Int](_error)
}

