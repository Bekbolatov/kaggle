package com.sparkydots.kaggle.taxi

import org.apache.spark.SparkContext
import com.databricks.spark.csv._
import Extract._
import org.apache.spark.sql.SQLContext
import org.apache.spark.util.StatCounter


object Script {

  def run(sqlContext: SQLContext): Unit = {

    val tripData = com.sparkydots.kaggle.taxi.Extract.readTripData(sqlContext, "train_1_10th").cache()
    val tripDataTest = com.sparkydots.kaggle.taxi.Extract.readTripData(sqlContext, "test", true).cache()


    // times by originCall
    tripDataTest.
      map(t => (t.originCall, t.travelTime)).
      mapValues(d => StatCounter(d)).
      reduceByKey(_.merge(_)).
      collect().
      sortBy(-_._2.count).
      take(10).foreach(println)

    // times by originStand
    tripDataTest.
      map(t => (t.originStand, t.travelTime)).
      mapValues(d => StatCounter(d)).
      reduceByKey(_.merge(_)).
      collect().
      sortBy(-_._2.count).
      take(10).foreach(println)


    // MISSING, also totalTravel time < 3600 is 99.4% of data

    val newguess = tripDataTest.map { t =>
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

    import sqlContext.implicits._
    newguess.toDF().saveAsCsvFile("hello2.csv")

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
