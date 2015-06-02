package com.sparkydots.kaggle.taxi

import com.github.nscala_time.time.Imports._
import com.sparkydots.util.geo.{Earth, Point}
import com.sparkydots.util.time.{Holidays, TimeProximity}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.json4s.jackson.JsonMethods._

import scala.util.Try

class Extract(@transient sc: SparkContext, @transient sqlContext: SQLContext, earth: Earth = Earth(Point(41.14, -8.62))) extends Serializable {

  def data(s3: Boolean = false, hdfs: Boolean = false, hd: Boolean = true, cv: Double = 0.0) = {
    val (rawTrainData, rawTestData) = if (s3) {
      (_readRawTripData("train", header = true, pathPrefix = "s3n://sparkydotsdata"),
        _readRawTripData("test", header = true, pathPrefix = "s3n://sparkydotsdata"))
    } else if(hdfs) {
      (_readRawTripData("train_1_10th", header = false, pathPrefix = "/user/ds/data"),
        _readRawTripData("test", header = true, pathPrefix = "/user/ds/data"))
    } else {
      (_readRawTripData("train", header = true, pathPrefix = "/data"),
        _readRawTripData("test", header = true, pathPrefix = "/data"))
    }

    val (trainData, _cvData) = if (cv > 0.0) {
      val Array(trData, rawCvData) = rawTrainData.randomSplit(Array(1.0 - cv, cv))
          val cvData = rawCvData.filter(_.polyline.length > 4).map { case rawTripData =>

            val actualPoints = rawTripData.polyline.length
            val cutoff = (
              math.abs(
                rawTripData.tripId.hashCode +
                rawTripData.polyline.headOption.map(p => (p.lon * 1e6).toInt).getOrElse(0)
                ) % (rawTripData.polyline.length - 1)) + 1

            val truncatedTrip = rawTripData.copy(polyline = rawTripData.polyline.take(cutoff))
            (actualPoints, truncatedTrip)
          }
      (trData, cvData)
    } else {
      (rawTrainData, sc.emptyRDD[(Int, TripData)])
    }

    val cvData = _cvData.collect.toList
    val testData = rawTestData.collect.toList.sortBy(_.tripId.drop(1).toInt)

    (trainData, cvData, testData)
  }

  def _readRawTripData(fileName: String = "test", header: Boolean = false, pathPrefix: String = "/user/ds/data"): RDD[TripData] = {

    val csvData = sqlContext.load("com.databricks.spark.csv", Map("path" -> s"${pathPrefix}/kaggle/taxi/$fileName.csv", "header" -> header.toString))

    csvData.map { r => //"TRIP_ID" 0 ,"CALL_TYPE" 1 ,"ORIGIN_CALL" 2 ,"ORIGIN_STAND" 3 ,"TAXI_ID" 4 ,"TIMESTAMP" 5 ,"DAY_TYPE" 6 ,"MISSING_DATA" 7 ,"POLYLINE"
      val timeInSecs = r.getString(5).toInt
      val dow = Extract.dayOfWeek(timeInSecs)
      val hod = Extract.hourOfDay(timeInSecs)

      TripData(
        r.getString(0), r.getString(4).toInt,

        if (dow < 5) 0 else 1,  // 0 weekday, 1 weekend
        if (hod < 11) 0 else if (hod < 4) 1 else if (hod < 7) 2 else 3,  //0 morning, 1 midday, 2 evening, 3 night

        Try(r.getString(2).toInt).toOption, Try(r.getString(3).toInt).toOption,
        Extract.parsePoints(r.getString(8), false)
      )
    }
  }
}


object Extract extends Serializable {

  // Mon, 11 Aug 2014 00:00:00
  val timeOrigin = 1407715200
  val secsInWeek = 604800
  val secsInDay = 86400
  val secsInHour = 3600

  // 0: Mon, 1: Tue, ... 6: Sun
  def dayOfWeek(timeInSecs: Int): Int = ((timeInSecs - timeOrigin) % secsInWeek) / secsInDay

  // 0:  [00:00..00:59], 1: [01:00..01:59], ... 23: [23:00..23:59]
  def hourOfDay(timeInSecs: Int): Int = ((timeInSecs - timeOrigin) % secsInDay) / secsInHour


  /**
   * Parse path
   * @param path String in format:  [ [23.00,22.43],[21.2,44.7] ]
   * @param latLon if true then [lat, lon]   else [lon, lat]
   * @return
   */
  def parsePoints(path: String, latLon: Boolean = true): Seq[Point] = {
    implicit val formats = org.json4s.DefaultFormats
    val parts = parse(path).extract[List[List[Double]]]
    if (latLon)
      parts.map(p => Point(p(0), p(1)))
    else
      parts.map(p => Point(p(1), p(0)))
  }

}