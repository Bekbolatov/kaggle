package com.sparkydots.kaggle.avito.functions

import scala.util.Try

object Functions extends Serializable {

  val APR20 = new java.text.SimpleDateFormat("yyyy-MM-dd hh:mm:ss.0").parse("2015-04-20 00:00:00.0").getTime
  val SECONDS_IN_HOUR = 60*60
  val SECONDS_IN_DAY = SECONDS_IN_HOUR * 24
  val SECONDS_IN_WEEK = SECONDS_IN_DAY * 7

  val parseTime = (tt: String) => {
    val format = new java.text.SimpleDateFormat("yyyy-MM-dd hh:mm:ss.0")
    ((format.parse(tt).getTime - APR20)/1000).toInt
  }

  /**
   * 0: Monday
   * 1: Tuesday
   * ...
   * 6: Sunday
   */
  val dayOfWeek = (t: Int) => t/SECONDS_IN_DAY % 7
  val weekend = (t: Int) => {
    val d = dayOfWeek(t)
    if(d == 5 || d == 6) 1 else 0
  }

  /**
   * 0, 1, ..., 23
   */
  val hourOfDay = (t: Int) => t/SECONDS_IN_HOUR % 24
  val time_morning = (hour: Int) => if (hour >= 7 && hour < 11) 1 else 0
  val time_afternoon = (hour: Int) => if (hour >= 11 && hour < 14) 1 else 0
  val time_noon = (hour: Int) => if (hour >= 14 && hour < 17) 1 else 0
  val time_evening = (hour: Int) => if (hour >= 17 && hour < 20) 1 else 0
  val time_late_evening = (hour: Int) => if (hour >= 20 || hour < 23) 1 else 0
  val time_night = (hour: Int) => if (hour >= 23 || hour < 7) 1 else 0


  val toInt = (text: String) => text.toInt

  val toIntOrMinus = (s: String) => Try(s.toInt).getOrElse(-1)

  val toDoubleOrMinus = (s: String) => Try(s.toDouble).getOrElse(-1.0)

  val length = (text: String) => text.length

  val toLower = (text: String) => text.toLowerCase

  val toMid = (searchId: String, adId: String) => 10000000L * searchId.toInt + adId.toInt

  val splitString = (tt: String) => {
    if (tt == null || tt.isEmpty) {
      Seq.empty[String]
    } else {
      tt.split("-")
        .mkString(" ").split(" ")
        .mkString(",").split(",")
        .mkString("\"").split("\"")
        .mkString("'").split("'")
        .mkString(":").split(":")
        .mkString(";").split(";")
        .mkString("-").split("-")
        .filter(_.nonEmpty)
        .toSeq
    }
  }

  val stemString = (x: String) => if (x.length > 4) x.dropRight(2) else if (x.length == 4) x.dropRight(1) else x

  val parseParams = (tt: String) => {
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

  val error = (guess: Double, actual: Double) => {
    - math.log(math.max(if (actual == 0.0) 1 - guess else guess, 1e-15))
  }

}
