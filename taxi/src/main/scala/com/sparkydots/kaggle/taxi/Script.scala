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
import com.github.nscala_time.time.Imports._


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

        val (tripData, tripDataAll, tripDataTestAll, segments, lastSegs, tripIds, knownLowerBound) = com.sparkydots.kaggle.taxi.Extract.readHDFS(sc)
//    val (tripData, tripDataAll, tripDataTestAll, segments, lastSegs, tripIds, knownLowerBound) = com.sparkydots.kaggle.taxi.Extract.readS3(sc)


    val bcLastSegs = sc.broadcast(lastSegs)
    val bcEarth = sc.broadcast(earth)

    val segmentMatchesStats = tripIds.zip(segments.map { case (segment, tripId) =>
      bcLastSegs.value.take(10).map {
        case (testTripId, Some(testSegment)) =>
          val stat1 = if (bcEarth.value.isSegmentNearStricter(segment, testSegment)) {
            StatCounter(math.log(15.0 * (testSegment.numSegmentsBefore + segment.numSegmentsAfter + 2)))
          } else StatCounter()
          val stat2 = if (bcEarth.value.isSegmentNear(segment, testSegment)) {
            StatCounter(math.log(15.0 * (testSegment.numSegmentsBefore + segment.numSegmentsAfter + 2)))
          } else StatCounter()
          (stat1, stat2)
        case (testTripId, None) =>
          (StatCounter(), StatCounter())
      }
    }.reduce((aa, a) => aa.zip(a).map { case ((xs, ys) , (x, y)) => (xs.merge(x), ys.merge(y)) })).toMap


    val plains = Seq(660.0, 600.0).map(p => math.log(p + 1))

    val guessedTimes = plains.map{ plain =>
      tripIds.map { case tripId =>
      (tripId, {

          math.max(
          {
            val sv = segmentMatchesStats(tripId)._1.mean
            val nsv = segmentMatchesStats(tripId)._2.mean
            val ss =
              if (sv > 0.0) sv
              else if (nsv > 0.0) nsv
              else plain
            math.exp(ss) - 1
          },
          knownLowerBound(tripId))
        })
    }.map(p => (p._1, p._2.toInt)) }

    guessedTimes.zipWithIndex.foreach { case (gt,i) => com.sparkydots.kaggle.taxi.Extract.writeResults(gt, s"/home/hadoop/results${i}.csv")
    //guessedTimes.zipWithIndex.foreach { case (gt,i) => com.sparkydots.kaggle.taxi.Extract.writeResults(gt, "/Users/renatb/data/kaggle/taxi_trip/tmpres.csv")
    //com.sparkydots.kaggle.taxi.Extract.writeResults(guessedTimes)


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

    val stats = lastSegs.toSeq.map { // .take(10).map {
      case (tripId, Some(lastSeg)) =>
        (tripId,
          closeSegments(lastSeg, tripId, segments)
            .map { case (segment, _) => math.log(15*(lastSeg.numSegmentsBefore + segment.numSegmentsAfter + 1)) }
            .stats())
      case (tripId, None) => (tripId, StatCounter())
    }

    val statsStricter = lastSegs.toSeq.map { // .take(10).map {
      case (tripId, Some(lastSeg)) =>
        (tripId,
          closeSegmentsStricter(lastSeg, tripId, segments)
            .map { case (segment, _) => math.log(15*(lastSeg.numSegmentsBefore + segment.numSegmentsAfter + 1)) }
            .stats())
      case (tripId, None) => (tripId, StatCounter())
    }

    val estimates2 = stats.toSeq.map(p => (p._1, p._2.mean)).sortBy(_._1.drop(1).toInt)
    val estimates2Stricter = statsStricter.toSeq.map(p => (p._1, p._2.mean)).sortBy(_._1.drop(1).toInt)

//    val known = tripDataTestAll.map { t =>
//      if (t.pathSegments.lastOption.map(segment => segment.direction > 0.4).getOrElse(false)) {
//        (t.tripId, t.travelTime + 60)
//      } else if (t.pathSegments.lastOption.map(segment => segment.direction > 0.1).getOrElse(false)) {
//        (t.tripId, t.travelTime + 30)
//      } else {
//        (t.tripId, t.travelTime)
//      }
//    }.collect().toSeq

    val known = tripDataTestAll.map { t => (t.tripId, t.travelTime) }.collect().toSeq

    val ests = estimates2.zip(estimates2Stricter).zip(known).map { case (((id1, e1),(id1stricter, e1stricter)), (id2, e2)) =>
      (id1, math.max(
        math.max(math.exp(if (e1stricter > 0) e1stricter else e1), e2),
        600.0
      ).toInt )
    }

    val ests500 = estimates2.zip(estimates2Stricter).zip(known).map { case (((id1, e1),(id1stricter, e1stricter)), (id2, e2)) =>
      (id1, math.max(
        math.max(math.exp(if (e1stricter > 0) e1stricter else e1), e2),
        500.0
      ).toInt )
    }

    //(count: 1700490, mean: 683.085440, stdev: 436.574915, max: 3585.000000, min: 0.000000)  //tripData.map { t => t.travelTime } .stats
    //(count: 1700490, mean: 6.231940, stdev: 1.126682, max: 8.184793, min: 0.000000) => 507.7414  //tripData.map { t => math.log(t.travelTime+1) } .stats
    //
    FileIO.write(ests.map(p => s"${p._1},${p._2}"), "/home/hadoop/eststr.csv")
    FileIO.write(ests500.map(p => s"${p._1},${p._2}"), "/home/hadoop/ests500.csv")
/////////////


        val tripA = tripDataTestAll.take(1)(0)
        Script.writePathsWithHourOtherFactors(sc, tripA, tripData)



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
