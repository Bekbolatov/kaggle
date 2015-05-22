package com.sparkydots.util.geo

import scala.collection.mutable
import com.github.nscala_time.time.Imports._


class Earth(p: Point, defaultIntervalSeconds: Int =  Earth.DEFAULT_SECONDS_PER_SEGMENT) extends Serializable {

  val KM_PER_LAT_DEG = math.sin(1 / 180.0 * math.Pi)
  val KM_PER_LON_DEG = math.sin(1 / 180.0 * math.Pi) * math.cos(p.lat / 180.0 * math.Pi)

  /**
   *
   * @param p point (vector) representing small change (ignored Earth curvature)
   * @return  (distance in km, direction in degrees (0 - 360)
   */
  def toPolar(p: Point): (Double, Double) = {
    val (dxx, dyy) = (p.lon * KM_PER_LON_DEG, p.lat * KM_PER_LAT_DEG)
    val _theta = math.atan2(dyy, dxx)
    val theta = if (_theta >= 0) _theta else _theta + 2 * math.Pi
    (Earth.RADIUS_KM * Math.sqrt(math.pow(dxx, 2) + math.pow(dyy, 2)), theta)
  }

  /**
   * Attempts to clean data points.
   * Removes temporary jumps in paths and picks largest candidate path.
   * (Jump is a "small" subset of consecutive points that are further than connecting points to the rest of the sequence)
   * (Jump limit is max taxi speed.)
   *
   *
   *
   * @param pts   Seq[(lat,lon)]
   * @param intervalSeconds  number of seconds between data points
   * @return  cleaned path
   */
  def cleanTaxiPath(pts: Seq[Point], intervalSeconds: Int = defaultIntervalSeconds): Seq[Point] = {
    if (pts.isEmpty)
      Seq()
    else {
      var pointSets = Seq[mutable.MutableList[Point]]()
      var ps = pts

      while (ps.nonEmpty) {
        val cur = ps.head
        ps = ps.tail
        var psets = pointSets
        var connected = false
        while (psets.nonEmpty && !connected) {
          val pset = psets.head
          val last = pset.last
          val (dist, angle) = toPolar(cur - last)
          if (dist < Earth.MAX_TAXI_SPEED_KM_PER_SEC * intervalSeconds) {
            pset += cur
            connected = true
          }
          psets = psets.tail
        }
        if (!connected) {
          pointSets +:= mutable.MutableList(cur)
        }
      }
      pointSets.maxBy(_.length).toList
    }
  }

  def pathComponents(pts: Seq[Point], intervalSeconds: Int = defaultIntervalSeconds): (Int, Seq[Double]) = {
    var jumps = mutable.MutableList[Double]()

    if (pts.isEmpty)
      (0, jumps.toList)
    else {
      var pointSets = Seq[mutable.MutableList[Point]]()
      var ps = pts
      var lastPt: Option[Point] = None

      while (ps.nonEmpty) {
        val cur = ps.head
        // BEGIN: Compute jumps ///
        if (lastPt.nonEmpty) {
          val (dist, angle) = toPolar(cur - lastPt.get)
          if (dist > Earth.MAX_TAXI_SPEED_KM_PER_SEC * intervalSeconds) {
            jumps += dist
          }
        }
        lastPt = Some(cur)
        // END: Compute jumps ///
        ps = ps.tail
        var psets = pointSets
        var connected = false
        while (psets.nonEmpty && !connected) {
          val pset = psets.head
          val last = pset.last
          val (dist, angle) = toPolar(cur - last)
          if (dist < Earth.MAX_TAXI_SPEED_KM_PER_SEC * intervalSeconds) {
            pset += cur
            connected = true
          }
          psets = psets.tail
        }
        if (!connected) {
          pointSets +:= mutable.MutableList(cur)
        }
      }
      (pointSets.length, jumps.toList)
    }
  }

  def pathPointsToPathSegments(points: Seq[Point], timestamp: DateTime, callType: String, originCall: Option[String], originStand: Option[String]): Seq[PathSegment] = {
    val numSegmentsAfter = points.length - 2
    val (origin, destination) = (points.headOption, points.lastOption)
    points.sliding(2).filter(_.length == 2).zipWithIndex.map { case (begin +: end +: Nil, index) =>
      val (distance, direction) = this.toPolar(end - begin)
      PathSegment(begin, end, distance, direction, index, numSegmentsAfter - index, origin.get, destination.get, timestamp, callType, originCall, originStand)
    }.toList
  }

  def isPointNear(p0: Point,
                  p1: Point,
                  maxDistSquared: Double = Earth.MAX_TAXI_NEAR_DIST_RAW_SQUARED): Boolean = {
    val dp = p1 - p0
    val (dxx, dyy) = (dp.lon * KM_PER_LON_DEG, dp.lat * KM_PER_LAT_DEG)
    Math.pow(dxx, 2) + Math.pow(dyy, 2) < Earth.MAX_TAXI_NEAR_DIST_RAW_SQUARED
  }

  def isValidSegment(s: PathSegment): Boolean = {
    s.distance > (Earth.MIN_TAXI_SPEED_KM_PER_SEC * Earth.DEFAULT_SECONDS_PER_SEGMENT)
  }

  def isSegmentNear(s0: PathSegment,
                    s1: PathSegment,
                    maxDistSquared: Double = Earth.MAX_TAXI_NEAR_DIST_RAW_SQUARED,
                    maxDir: Double = Earth.MAX_TAXI_NEAR_DIRECTION): Boolean =
    isPointNear(s0.begin, s1.begin) && math.abs(s1.direction - s0.direction) < maxDir && s0.distance > Earth.MIN_TAXI_SPEED_KM_PER_SEC * defaultIntervalSeconds && s1.distance > Earth.MIN_TAXI_SPEED_KM_PER_SEC * defaultIntervalSeconds

  def isSegmentNearStricter(s0: PathSegment,
                    s1: PathSegment,
                    maxDistSquared: Double = Earth.MAX_TAXI_NEAR_DIST_RAW_SQUARED,
                    maxDir: Double = Earth.MAX_TAXI_NEAR_DIRECTION): Boolean =
    isPointNear(s0.begin, s1.begin) &&
      math.abs(s1.direction - s0.direction) < maxDir &&
      s0.distance > Earth.MIN_TAXI_SPEED_KM_PER_SEC * defaultIntervalSeconds &&
      s1.distance > Earth.MIN_TAXI_SPEED_KM_PER_SEC * defaultIntervalSeconds &&
      s0.callType == s1.callType &&
      {
        val originStand0 = s0.originStand match {
          case None => None
          case Some("57") => Some("57")
          case Some("15") => Some("15")
          case other => Some("O")
        }
        val originStand1 = s1.originStand match {
          case None => None
          case Some("57") => Some("57")
          case Some("15") => Some("15")
          case other => Some("O")
        }
        originStand0 == originStand1
      } &&
      {
        val originCall0 = s0.originCall match {
          case None => None
          case other => Some("1")
        }
        val originCall1 = s1.originCall match {
          case None => None
          case other => Some("1")
        }
        originCall0 == originCall1
      }

}

object Earth extends Serializable {

  val RADIUS_KM = 6371
  val MAX_TAXI_SPEED_KM_PER_SEC = 200.0 / 60 / 60
  val MAX_HUMAN_SPEED_KM_PER_SEC = 450.0 / 60 / 60

  val DEFAULT_SECONDS_PER_SEGMENT = 15

  val MAX_TAXI_NEAR_DIRECTION = 90.0 * math.Pi / 180 // in radians (0 - 2Pi)
  val MIN_TAXI_SPEED_KM_PER_SEC = 1.0 / 60 / 60
  val MAX_TAXI_NEAR_DIST_RAW = 50.0 / 1000 / Earth.RADIUS_KM
  val MAX_TAXI_NEAR_DIST_RAW_SQUARED = math.pow(MAX_TAXI_NEAR_DIST_RAW, 2)

  def apply(p: Point): Earth = new Earth(p)

  /**
   *
   * Haversine distance (for small distances) between two points
   *
   * @return distance in km
   */
  def haversineDistance(p1: Point, p2: Point): Double = {
    val la1 = p1.lat * math.Pi/180
    val lo1 = p1.lon * math.Pi/180
    val la2 = p2.lat * math.Pi/180
    val lo2 = p2.lon * math.Pi/180

    val la = math.abs(la1 - la2)
    val lo = math.abs(lo1-lo2)

    val a = math.sin(la/2)*math.sin(la/2)+math.cos(la1)*math.cos(la2)*math.sin(lo/2)*math.sin(lo/2)
    val d = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d * RADIUS_KM
  }

  /**
   * Parse path
   * @param path String in format:  [ [23.00,22.43],[21.2,44.7] ]
   * @param latLon if true then [lat, lon]   else [lon, lat]
   * @return
   */
  def parsePoints(path: String, latLon: Boolean = true): Seq[Point] = {
    val parts = path.split(Array(' ', '[', ']', ',', '\n', '\t', '\r')).filter(_.trim.nonEmpty)
    val (xs, ys) = parts.zipWithIndex.partition(_._2 % 2 == 0)
    val pairs = xs.zip(ys).map(xiyi => (xiyi._1._1.toDouble, xiyi._2._1.toDouble)).toSeq
    if (latLon)
      pairs.map(p => Point(p._1, p._2))
    else
      pairs.map(p => Point(p._2, p._1))
  }



}
