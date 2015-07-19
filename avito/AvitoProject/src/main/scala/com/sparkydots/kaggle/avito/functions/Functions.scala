package com.sparkydots.kaggle.avito.functions

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
  val _weekend = (t: Int) => {
    val d = _dayOfWeek(t)
    if(d == 5 || d == 6) 1 else 0
  }

  /**
   * 0, 1, ..., 23
   */
  val _hourOfDay = (t: Int) => t/_SECONDS_IN_HOUR % 24
  val _time_morning = (hour: Int) => if (hour >= 6 && hour < 10) 1 else 0
  val _time_noon = (hour: Int) => if (hour >= 10 && hour < 14) 1 else 0
  val _time_afternoon = (hour: Int) => if (hour >= 14 && hour < 17) 1 else 0
  val _time_evening = (hour: Int) => if (hour >= 17 && hour < 7) 1 else 0
  val _time_late_evening = (hour: Int) => if (hour >= 19 && hour < 23) 1 else 0
  val _time_night = (hour: Int) => if (hour >= 23 || hour < 6) 1 else 0


  // Other //
  val _toInt = (text: String) => text.toInt

  val _toIntOrMinus = (s: String) => Try(s.toInt).getOrElse(-1)

  val _toDoubleOrMinus = (s: String) => Try(s.toDouble).getOrElse(-1.0)

  val _length = (text: String) => text.length

  val _toLower = (text: String) => text.toLowerCase

  val _toMid = (searchId: String, adId: String) => 10000000L * searchId.toInt + adId.toInt

  val _parseParams = (tt: String) => {
    if (tt.isEmpty) {
      Seq.empty
    } else {
      val terms = tt.substring(1, tt.length - 1)
        .replaceAll("\\:\\{[^\\}]+\\}",":'XX'")
        .replaceAll("\\:\\[[^\\]]+\\]",":'XX'")
        .replaceAll("\\:'[^']+'",":'XX'")
        .split(", ")
      terms.map(x => x.split(":")(0)).filter(_.nonEmpty).map(_.toInt).toSeq.sorted
    }
  }

  val _error = (guess: Double, actual: Double) => {
    - math.log(math.max(if (actual == 0.0) 1 - guess else guess, 1e-15))
  }

}
