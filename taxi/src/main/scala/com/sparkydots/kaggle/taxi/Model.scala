package com.sparkydots.kaggle.taxi

import com.github.nscala_time.time.Imports._
import com.sparkydots.util.geo.Point

case class RawTripData(
    TRIP_ID: String,
    CALL_TYPE: String,
    ORIGIN_CALL: String,
    ORIGIN_STAND: String,
    TAXI_ID: String,
    TIMESTAMP: String,
    DAY_TYPE: String,
    MISSING: String,
    POLYLINE: Seq[Point])

case class TripData(
   tripId: String,
   callType: String,
   originCall: Option[String],
   originStand: Option[String],
   taxiID: Int,
   timestamp: DateTime,
   rawDataPoints: Seq[Point],
   hourOfDay: Int, //

   approximateOrigin: Point,
   approximateDestination: Point,
   avgSpeed: Double,
   avgNorthDirection: Double, //
   avgEastDirection: Double, //
   magnitude: Double,
   direction: Double,

   elapsedTime: Double)

// in training data, 99.4% is below 3600s (1 HR)


case class PathSegment(
    begin: Point,
    end: Point,
    distance: Double,
    direction: Double,
    numSegmentsBefore: Int,
    numSegmentsAfter: Int,
    origin: Point,
    destination: Point,
    originTimestamp: DateTime,
    callType: String,
    originCall: Option[String],
    originStand: Option[String]) extends Serializable

