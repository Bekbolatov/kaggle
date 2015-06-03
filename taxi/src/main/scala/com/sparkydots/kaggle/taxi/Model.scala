package com.sparkydots.kaggle.taxi

import com.sparkydots.util.geo.Point

case class TripData(
    tripId: String, taxiId: Int,
    weekday: Int, timeOfDay: Int,
    originCall: Option[Int], originStand: Option[Int],
    polyline: Seq[Point])

// in training data, 99.4% is below 3600s (1 HR)
