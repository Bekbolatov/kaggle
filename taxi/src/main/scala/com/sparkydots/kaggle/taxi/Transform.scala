package com.sparkydots.kaggle.taxi

import com.sparkydots.util.geo.{Earth, PathSegment, Point}
import org.apache.spark.rdd.RDD

object Transform extends Serializable {

  def pathSegments(trips: RDD[TripData]): RDD[(PathSegment, String)] =
    for {
      trip <- trips
      segment <- trip.pathSegments
    } yield (segment, trip.tripId)


  def closeSegments(thisSegment: PathSegment,
                    thisTripId: String,
                    pathSegments: RDD[(PathSegment, String)])
                   (implicit earth: Earth): RDD[(PathSegment, String)] = {

    val allCloseSegments = for {
      (segment, tripId) <- pathSegments if earth.isSegmentNear(segment, thisSegment) && tripId != thisTripId
    } yield (segment, tripId)

    allCloseSegments
      .groupBy { case (segment, tripId) => tripId }
      .mapValues(_.head)
      .map { case (_, (segment, tripId)) => (segment, tripId) }
  }

  def closeSegments(trip: TripData,
                    pathSegments: RDD[(PathSegment, String)])
                   (implicit earth: Earth): Option[RDD[(PathSegment, String)]] =
    trip.pathSegments.lastOption.map(segment => closeSegments(segment, trip.tripId, pathSegments)(earth))

  def closeSegments(begin: Point, pathSegments: RDD[(PathSegment, String)])
                   (implicit earth: Earth): RDD[(PathSegment, String)] = {
    for {
      (segment, tripId) <- pathSegments if earth.isPointNear(segment.begin, begin)
    } yield (segment, tripId)
  }

  def closeTrips(thisSegment: PathSegment,
                 thisTripId: String,
                 trips: RDD[TripData])
                (implicit earth: Earth): RDD[String] = {

    val pathSegment = pathSegments(trips)
    closeSegments(thisSegment, thisTripId, pathSegment)
      .map { case (segment, tripId) => tripId }
  }

  def closeTrips(trip: TripData,
                 trips: RDD[TripData])
                (implicit earth: Earth): Option[RDD[String]] = {
    trip.pathSegments.lastOption.map(segment => closeTrips(segment, trip.tripId, trips)(earth))
  }



}
