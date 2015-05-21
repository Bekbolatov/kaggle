package com.sparkydots.kaggle.taxi

import com.github.nscala_time.time.Imports._
import com.sparkydots.kaggle.taxi.Transform._
import com.sparkydots.util.geo.{Earth, Point}
import org.apache.spark.SparkContext
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
      |
      | def readS3(sc: SparkContext) = { ...
      | def readHDFS(sc: SparkContext) = { ...
      |         (tripData, tripDataAll, tripDataTestAll, segments, lastSegs)
      |
    """.stripMargin

  def readS3(sc: SparkContext) = {
    val sqlContext = new SQLContext(sc)
    implicit val earth = Earth(Point(41.14, -8.62))

    val (tripData, tripDataAll) = com.sparkydots.kaggle.taxi.Extract.read(sqlContext, "train", true, "s3n://sparkydotsdata")
    val (tripDataTest, tripDataTestAll) = com.sparkydots.kaggle.taxi.Extract.read(sqlContext, "test", true, "s3n://sparkydotsdata")

    val segments = pathSegments(tripData).cache()
    val lastSegs = tripDataTestAll.map(t => (t.tripId, t.pathSegments.lastOption)).collect().toMap
    (tripData, tripDataAll, tripDataTestAll, segments, lastSegs)
  }

  def readHDFS(sc: SparkContext) = {
    val sqlContext = new SQLContext(sc)
    implicit val earth = Earth(Point(41.14, -8.62))

    val (tripData, tripDataAll) = com.sparkydots.kaggle.taxi.Extract.read(sqlContext, "train_1_10th")
    val (tripDataTest, tripDataTestAll) = com.sparkydots.kaggle.taxi.Extract.read(sqlContext, "test", true)

    val segments = pathSegments(tripData).cache()
    val lastSegs = tripDataTestAll.map(t => (t.tripId, t.pathSegments.lastOption)).collect().toMap
    (tripData, tripDataAll, tripDataTestAll, segments, lastSegs)
  }


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
    val timestamp = new DateTime(rawTripData.TIMESTAMP.toLong)
    val originCall = rawTripData.ORIGIN_CALL match {
      case s if s.nonEmpty && s != "NA" => Some(s.trim)
      case _ => None
    }
    val originStand = rawTripData.ORIGIN_STAND match {
      case s if s.nonEmpty && s != "NA" => Some(s.trim)
      case _ => None
    }
    val rawPathPoints = Earth.parsePoints(rawTripData.POLYLINE, false)
      val pathPoints = e.cleanTaxiPath(rawPathPoints, 15)
      val pathComponents = e.pathComponents(rawPathPoints, 15)
      val pathSegments = e.pathPointsToPathSegments(pathPoints, timestamp, rawTripData.CALL_TYPE, originCall, originStand)

      TripData(
        rawTripData.TRIP_ID,
        rawTripData.CALL_TYPE,
        originCall,
        originStand,
        rawTripData.TAXI_ID.toInt,
        timestamp,
        rawTripData.DAY_TYPE,
        rawTripData.MISSING.toBoolean,
        rawPathPoints,
        pathPoints,
        pathComponents,
        pathSegments,
        math.max(rawPathPoints.length - 1, 0) * 15)
  }

}