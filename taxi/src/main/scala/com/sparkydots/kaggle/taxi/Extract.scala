package com.sparkydots.kaggle.taxi

import com.github.nscala_time.time.Imports._
import com.sparkydots.util.geo.{Earth, Point}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.json4s.jackson.JsonMethods._

class Extract(@transient sc: SparkContext, @transient sqlContext: SQLContext, earth: Earth = Earth(Point(41.14, -8.62))) extends Serializable {

  def data(s3: Boolean = false, hdfs: Boolean = false) = {

    val (rawTripDataAll, rawTestDataAll) = if (s3) {
      (_readRawTripData("train", header = true, pathPrefix = "s3n://sparkydotsdata"),
        _readRawTripData("test", header = true, pathPrefix = "s3n://sparkydotsdata"))
    } else {
      (_readRawTripData("train_1_10th", header = false, pathPrefix = "/user/ds/data"),
        _readRawTripData("test", header = true, pathPrefix = "/user/ds/data"))
    }

    rawTripDataAll.cache()

    val tripDataFiltered = rawTripDataAll.
      filter(t => t.POLYLINE.length > 1 && t.POLYLINE.length < 240)

    val tripDataCut = tripDataFiltered.
      map { case rawTripData =>
      val cutoff = (
        math.abs((rawTripData.TRIP_ID.hashCode +
          rawTripData.TIMESTAMP.hashCode +
          rawTripData.POLYLINE.headOption.map(p => (p.lon * 1e6).toInt).getOrElse(0)
          )) % rawTripData.POLYLINE.length) + 1
      (math.log(15.0 * (rawTripData.POLYLINE.length - 1) + 1), rawTripData.copy(POLYLINE = rawTripData.POLYLINE.take(cutoff)))
    }

    val origTripData = tripDataFiltered.map(createTripData)

    val tripData = tripDataCut map { case (actualTime, trip) => (actualTime, createTripData(trip)) }
    val testData = rawTestDataAll map { trip => (-1.0, createTripData(trip)) }

    val tripIds = testData.map(_._2.tripId).collect.toList.sortBy(_.drop(1).toInt)
    val knownLowerBound = testData.map { case (_, trip) => (trip.tripId, trip.elapsedTime) }.collect().toMap

    (testData, origTripData, tripData, tripDataFiltered, rawTestDataAll, rawTripDataAll, tripIds, knownLowerBound)
  }

  def _readRawTripData(fileName: String = "test", header: Boolean = false, pathPrefix: String = "/user/ds/data"): RDD[RawTripData] = {
    val csvData = sqlContext.load("com.databricks.spark.csv", Map("path" -> s"${pathPrefix}/kaggle/taxi/$fileName.csv", "header" -> header.toString))
    csvData.map { r =>
      RawTripData(r.getString(0), r.getString(1), r.getString(2), r.getString(3), r.getString(4), r.getString(5), r.getString(6), r.getString(7),
        Extract._parsePoints(r.getString(8), false)
      )
    }
  }


  def createTripData(rawTripData: RawTripData): TripData = {

    val timestamp =  new DateTime(rawTripData.TIMESTAMP.toLong * 1000L)
    val hourOfDay = timestamp.hourOfDay().get()
    val originCall = rawTripData.ORIGIN_CALL match {
      case s if s.nonEmpty && s != "NA" => Some(s.trim)
      case _ => None
    }
    val originStand = rawTripData.ORIGIN_STAND match {
      case s if s.nonEmpty && s != "NA" => Some(s.trim)
      case _ => None
    }
    val (pathPoints, avgSpeed) = earth.cleanTaxiPath(rawTripData.POLYLINE, 15)

    val avgOver = 2
    val approximateOrigin = Point(pathPoints.take(avgOver).map(_.lat).sum / avgOver, pathPoints.take(avgOver).map(_.lon).sum / avgOver)
    val approximateDestination = Point(pathPoints.takeRight(avgOver).map(_.lat).sum / avgOver, pathPoints.takeRight(avgOver).map(_.lon).sum / avgOver)
    val (north, est, mag, dir) = (approximateDestination - approximateOrigin).dirs(earth)

    TripData(
      rawTripData.TRIP_ID,
      rawTripData.CALL_TYPE,
      originCall,
      originStand,
      rawTripData.TAXI_ID.toInt,
      timestamp,
      rawTripData.POLYLINE,
      hourOfDay,

      approximateOrigin, //origin
      approximateDestination, //dest
      avgSpeed, // avg speed
      north, //north
      est, //east
      mag,
      dir,

      math.max(rawTripData.POLYLINE.length - 1, 0) * 15)
  }


  /**
   * make sure to cache this tripData RDD before running this method
   * @param tripData
   * @return
   */
  def featurize(tripData: RDD[(Double, TripData)]): (RDD[LabeledPoint], Map[Int, Int], Map[String, Int], Map[Option[String], Int]) = {

    val callTypes = tripData.map(_._2.callType).distinct().collect().zipWithIndex.toMap

    val originStands = tripData.map(_._2.originStand).distinct().collect().zipWithIndex.toMap

    val bcCallTypes = sc.broadcast(callTypes)
    val bcOriginStands = sc.broadcast(originStands)

    val dataPoints = tripData.map { case (travelTime, trip) =>
      Extract.featureVector(travelTime, trip, bcCallTypes.value, bcOriginStands.value)
    }

    (dataPoints, Map(0 -> callTypes.size, 1 -> originStands.size), callTypes, originStands)
  }

}


object Extract extends Serializable {
  /**
   * Parse path
   * @param path String in format:  [ [23.00,22.43],[21.2,44.7] ]
   * @param latLon if true then [lat, lon]   else [lon, lat]
   * @return
   */
  def _parsePoints(path: String, latLon: Boolean = true): Seq[Point] = {
    implicit val formats = org.json4s.DefaultFormats
    val parts = parse(path).extract[List[List[Double]]]
    if (latLon)
      parts.map(p => Point(p(0), p(1)))
    else
      parts.map(p => Point(p(1), p(0)))
  }

  def featureVector(label: Double, trip: TripData, callTypes: Map[String, Int], originStands: Map[Option[String], Int]) = {
    val features = Vectors.dense(
      callTypes(trip.callType),
      originStands(trip.originStand),
      trip.hourOfDay,
      trip.approximateOrigin.lat,
      trip.approximateOrigin.lon,
      trip.approximateDestination.lat,
      trip.approximateDestination.lon,
      trip.avgSpeed,
      trip.avgNorthDirection,
      trip.avgEastDirection
    )
    LabeledPoint(label, features)
  }

}