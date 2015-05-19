package com.sparkydots.kaggle.taxi

import com.github.nscala_time.time.Imports._
import com.sparkydots.util.geo.{PathSegment, Point}

case class TripData(
                     tripId: String,
                     callType: String,
                     originCall: Option[String],
                     originStand: Option[String],
                     taxiID: Int,
                     timestamp: DateTime,
                     dayType: String,    // always: "A"
                     missing: Boolean,   // most always: False
                     rawPathPoints: Seq[Point],
                     pathPoints: Seq[Point],
                     pathSegments: Seq[PathSegment],
                     travelTime: Double)  // in training data, 99.4% is below 3600s (1 HR)

