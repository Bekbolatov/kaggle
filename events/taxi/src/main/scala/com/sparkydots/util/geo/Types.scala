package com.sparkydots.util.geo

/**
 *
 * @param lat in degrees (360 for 2Pi)
 * @param lon in degrees (360 for 2Pi)
 */
case class Point(lat: Double, lon: Double) extends Serializable {
  def +(other: Point): Point = Point(this.lat + other.lat, this.lon + other.lon)
  def -(other: Point): Point = Point(this.lat - other.lat, this.lon - other.lon)
  def /(k: Double): Point = Point(this.lat/k, this.lon/k)
  def p = s"$lat,$lon"
  def dirs(e: Earth) = {
    val (mag, dir) = e.toPolar(this)
    val dirRad = math.Pi * dir / 180
    (math.cos(dirRad), math.sin(dirRad), mag, dir)
  }
}


