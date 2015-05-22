package com.sparkydots.util.geo

import com.github.nscala_time.time.Imports._

/**
 *
 * @param lat in degrees (360 for 2Pi)
 * @param lon in degrees (360 for 2Pi)
 */
case class Point(lat: Double, lon: Double) extends Serializable {
  def -(other: Point): Point = Point(this.lat - other.lat, this.lon - other.lon)
  def p = s"$lat,$lon"
}


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
