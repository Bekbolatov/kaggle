package com.sparkydots.kaggle.taxi

import com.sparkydots.util.geo.{Point, Earth}
import com.sparkydots.util.io.FileIO
import org.apache.spark.SparkContext
import com.databricks.spark.csv._
import com.sparkydots.kaggle.taxi.Transform._
import com.sparkydots.kaggle.taxi._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.util.StatCounter


object Script {


  def writePaths(sc: SparkContext, tripA: TripData, tripData: RDD[TripData])(implicit earth: Earth) = {
    val closeTripsA = closeTrips(tripA.pathSegments.last, tripA.tripId, tripData)(earth)
    val closeTripsSetA = closeTripsA.collect.toSet
    val bctr = sc.broadcast(closeTripsSetA)
    val paths = tripData.collect { case t if bctr.value(t.tripId) => (t.tripId,  t.rawPathPoints) }.flatMap(t => t._2.map(s => s"${t._1},${s.lon},${s.lat}")).collect
    FileIO.write(paths, "/Users/renatb/data/kaggle/taxi_trip/pathsForTripA.csv")
  }

  def writePathsWithHour(sc: SparkContext, tripA: TripData, tripData: RDD[TripData])(implicit earth: Earth) = {
    val closeTripsA = closeTrips(tripA.pathSegments.last, tripA.tripId, tripData)(earth)
    val closeTripsSetA = closeTripsA.collect.toSet
    val bctr = sc.broadcast(closeTripsSetA)
    val paths = tripData.collect {
      case t if bctr.value(t.tripId) => (t.timestamp.hourOfDay().getAsText,  t.rawPathPoints) }
      .flatMap(t => t._2.map(s => s"${t._1},${s.lon},${s.lat}")).collect
    FileIO.write(paths, "/Users/renatb/data/kaggle/taxi_trip/pathsForTripAWithHours.csv")
  }

  def writePathsWithHourOtherFactors(sc: SparkContext, tripA: TripData, tripData: RDD[TripData])(implicit earth: Earth) = {
    val closeTripsA = closeTrips(tripA.pathSegments.last, tripA.tripId, tripData)(earth)
    val closeTripsSetA = closeTripsA.collect.toSet
    val bctr = sc.broadcast(closeTripsSetA)
    val paths = tripData.collect {
      case t if bctr.value(t.tripId) => (t.rawPathPoints, t.timestamp.hourOfDay().getAsText, t.callType, t.originCall.getOrElse("O"), t.originStand.getOrElse("O")) }
      .flatMap(t => t._1.map(s => s"${s.lon},${s.lat},${t._2},${t._3},${t._4},${t._5}")).collect
    FileIO.write(paths, "/Users/renatb/data/kaggle/taxi_trip/pathsForTripAWithHours.csv")
  }

  def run(sc: SparkContext): Unit = {
    implicit val earth = Earth(Point(41.14, -8.62))


        val (tripData, tripDataAll, tripDataTestAll, segments, lastSegs) = com.sparkydots.kaggle.taxi.Extract.readHDFS(sc)
//    val (tripData, tripDataAll, tripDataTestAll, segments, lastSegs) = com.sparkydots.kaggle.taxi.Extract.readS3(sc)

    val tripA = tripDataTestAll.take(1)(0)
    Script.writePathsWithHourOtherFactors(sc, tripA, tripData)




    //    val lastSegsList = lastSegs.toSeq.take(5)
//    val emptyStats = lastSegsList.map(a => StatCounter())
//    val bcLastSegs = sc.broadcast(lastSegsList)
//    segments.aggregate(emptyStats)(
//    seqOp = (stats, nextSeg) => {
//      if (nextSeg._1)
//    },
//    combOp = ???)
//
//      segment =>
//      bcLastSegs.value.foreach {
//        case (thisTripId, Some(thisSegment)) => ???
//        case (thisTripId, None) => StatCounter()
//
//      }
//    }
//
//

    val stats = lastSegs.take(5).map { // .take(10).map {
      case (tripId, Some(lastSeg)) =>
        (tripId,
          closeSegments(lastSeg, tripId, segments)
            .map { case (segment, _) => math.log(15*(lastSeg.numSegmentsBefore + segment.numSegmentsAfter + 2)) }
            .stats())
      case (tripId, None) => (tripId, StatCounter())
    }


    val estimates2 = stats.toSeq.map(p => (p._1, p._2.mean)).sortBy(_._1.drop(1).toInt)
    val known = tripDataTestAll.map( t=> (t.tripId, t.travelTime + 30)).collect().toSeq
    val ests = estimates2.zip(known).map { case ((id1, e1), (id2, e2)) => (id1, math.max( math.max(e1, e2), 600.0).toInt ) }

    ests.foreach(p => println(s"${p._1},${p._2}"))




    val closest = closeSegments(tripA, segments).get.cache()
    val closestp = closeSegments(tripA.pathSegments.last.begin, segments).cache()
    val closestB = closeSegments(tripA.pathSegments.last.copy(direction = tripA.pathSegments.last.direction - 3.14), tripA.tripId, segments).cache()


    closest.map { case (segment, _) => segment.numSegmentsAfter }.stats
    closest.map { case (segment, _) => (segment.numSegmentsAfter) }.stats

    //val sampleStrips = tripDataTest.take(3)
    //val closest = sampleStrips.map { trip => closeSegments(trip, segments).map{_.cache()} }


    // http://www.darrinward.com/lat-long/?id=586088
    closest.take(30).foreach(p =>println(s"${p._1.begin.lat}, ${p._1.begin.lon}"))

    closest.take(15).foreach(println)

    val trip_1 = tripData.filter(_.tripId == "1385972694620000329").take(1)(0)




    // times by originCall
    tripDataTestAll.
      map(t => (t.originCall, t.travelTime)).
      mapValues(d => StatCounter(d)).
      reduceByKey(_.merge(_)).
      collect().
      sortBy(-_._2.count).
      take(10).foreach(println)

    // times by originStand
    tripDataTestAll.
      map(t => (t.originStand, t.travelTime)).
      mapValues(d => StatCounter(d)).
      reduceByKey(_.merge(_)).
      collect().
      sortBy(-_._2.count).
      take(10).foreach(println)


    // MISSING, also totalTravel time < 3600 is 99.4% of data

    val newguess = tripDataTestAll.map { t =>
      val segs = t.pathSegments
       val newtime =
         math.max(
           segs.lastOption.map { s =>
             t.travelTime + (
               if (s.distance > 0.6)
                 10
               else if (s.distance > 0.4)
                 4
               else if (s.distance > 0.2)
                 3
               else 1
               )*15
           }.getOrElse(t.travelTime),
           660)
      (t.tripId, newtime.toInt)
       }

//    import sqlContext.implicits._
//    newguess.toDF().saveAsCsvFile("hello2.csv")

    /*
  val dirs = data.flatMap(d => {
    val total = d.segments.length
    d.points.dropRight(1).zip(d.segments).zipWithIndex.map { case ((p,s), i) =>  (p, s, i, total - i) }
  }).cache()


  dirs.filter(d => e.nearTaxi((41.1414, -8.6186), d._1)).take(40).foreach(println)

  dirs.sample(false, 0.01, 101L).take(15).foreach(println)

  val t = dirs.filter(d => e.nearTaxi((41.2019843,-8.5729285), d._1))
  t.map(_._2._2).filter(_ < 3).stats
  t.map(_._2._2).filter(_ > 3).stats



  val t = dirs.filter(d => e.nearTaxi((41.1699931,-8.5919172), d._1))
  t.map(_._2._2).filter(_ < 3).stats
  t.map(_._2._2).filter(_ > 3).stats




  data.filter(_.points.nonEmpty).map(d => s"${d.points.head._2},${d.points.head._1},${d.travelTime}").saveAsTextFile("/tmp/first_times_train.txt")




  data.take(3).foreach(println)

  val dirs = data.filter(!_._4).flatMap(p => p._6.zipWithIndex.map(x => (p._1, x._2, x._1)))
  val dirspol = dirs.map(p => (p._1, p._2, toPolar(p._3._1, p._3._2)))



//path
val a = data.filter(_._1 == "1372717190620000388").collect
a(0)._5.foreach(p =>  println(s"${p(1)},${p(0)}"))


val tdata = readData(sc, "test").cache()
val dirs = tdata.map(d => {
    val pred = if (d.segments.nonEmpty) {
      val curlength = d.segments.length
      val speed = d.segments.last._1
      math.max(d.travelTime + 15, if (speed < 0.4) {
        curlength match {
          case a if a < 20 => math.exp(6.422448) - 1
          case a if a < 100 => math.exp(6.874584) - 1
          case _ => math.exp(8.373896) - 1
        }
      } else {
        curlength match {
          case a if a < 20 => math.exp(6.486373) - 1
          case a if a < 100 =>  math.exp(6.979766) - 1
          case _ => math.exp(8.551063) - 1
        }
      })
    } else {
      660.0
    }
    Row(d.id, math.max(660, pred).toInt)
  }).cache()


 */

  }

}
