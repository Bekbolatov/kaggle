package com.sparkydots.kaggle.taxi

import com.github.nscala_time.time.Imports._
import com.sparkydots.util.geo.{Earth, Point}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

object Extract extends Serializable {

  def info =
    """
      |
      |def read(
      |          sqlContext: SQLContext,
      |          fileName: String = "test",
      |          header: Boolean = false,
      |          pathPrefix: String = "/user/ds/data",  //s3n://sparkydotsdata   or
      |          filterOp: (TripData) => Boolean = (t: TripData) => t.travelTime < 3600)
      |          (implicit earth: Earth = Earth(Point(41.14, -8.62))): (RDD[TripData], RDD[TripData]) = {
      |
    """.stripMargin

  def read(
          sqlContext: SQLContext,
          fileName: String = "test",
          header: Boolean = false,
          pathPrefix: String = "/user/ds/data",  //s3n://sparkydotsdata   or
          filterOp: (TripData) => Boolean = (t: TripData) => t.travelTime < 3600)
          (implicit earth: Earth = Earth(Point(41.14, -8.62))): (RDD[TripData], RDD[TripData]) = {

    val fullFileOption =  s"${pathPrefix}/kaggle/taxi/$fileName.csv"
    val headerOption = if (header) "true" else "false"
    val options = Map("path" -> fullFileOption, "header" -> headerOption)

    val csvData = sqlContext.load("com.databricks.spark.csv", options)
    val rawTripData = csvData.map { r => RawTripData(r.getString(0), r.getString(1), r.getString(2), r.getString(3), r.getString(4), r.getString(5), r.getString(6), r.getString(7), r.getString(8)) }
    val tripData = rawTripData.map(createTripData(_, earth))
    val cleanTripData = tripData.filter(filterOp)
    (cleanTripData, tripData)
  }

  private case class RawTripData(
                          TRIP_ID: String,
                          CALL_TYPE: String,
                          ORIGIN_CALL: String,
                          ORIGIN_STAND: String,
                          TAXI_ID: String,
                          TIMESTAMP: String,
                          DAY_TYPE: String,
                          MISSING: String,
                          POLYLINE: String)

  private def createTripData(rawTripData: RawTripData, e: Earth): TripData = {
    val rawPathPoints = Earth.parsePoints(rawTripData.POLYLINE, false)
      val pathPoints = e.cleanTaxiPath(rawPathPoints, 15)
      val pathSegments = e.pathPointsToPathSegments(pathPoints)

      TripData(
        rawTripData.TRIP_ID,
        rawTripData.CALL_TYPE,
        rawTripData.ORIGIN_CALL match {
          case s if s.nonEmpty && s != "NA" => Some(s.trim)
          case _ => None
        },
        rawTripData.ORIGIN_STAND match {
          case s if s.nonEmpty && s != "NA" => Some(s.trim)
          case _ => None
        },
        rawTripData.TAXI_ID.toInt,
        new DateTime(rawTripData.TIMESTAMP.toLong),
        rawTripData.DAY_TYPE,
        rawTripData.MISSING.toBoolean,
        rawPathPoints,
        pathPoints,
        pathSegments,
        math.max(rawPathPoints.length - 1, 0) * 15)
  }

}