package com.sparkydots.kaggle.taxi

import com.sparkydots.util.geo.{Point, Earth}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.util.StatCounter
import org.apache.log4j.Logger
import org.apache.log4j.Level

object Script {

  def run(sc: SparkContext, sqlContext: SQLContext): Unit = {


    Logger.getLogger("amazon.emr.metrics").setLevel(Level.OFF)


    val bcparams = sc.broadcast(Map("dist" -> 2.0, "offset" -> 0.0))

    val earth: Earth = Earth(Point(41.14, -8.62))
    val bce = sc.broadcast(earth)

    val extract = new com.sparkydots.kaggle.taxi.Extract(sc, sqlContext)

//    val (trainData, cvData, testData) = extract.data(hdfs = true, cv = 0.002)  /// s3=true
//    trainData.setName("trainData").cache()
//    val bcTripsCV = sc.broadcast(cvData.map(_._2))
//    val knownLowerBoundCV = cvData.map(_._2).map { case trip => (trip.tripId, 15*math.max(trip.polyline.length - 1, 0)) }.toMap
//    val (bcTrySet, trySet, lowerBound) =  (bcTripsCV, cvData.map(_._2), knownLowerBoundCV)

        val (trainData, cvData, testData) = extract.data(s3 = true, cv = 0.3)  /// s3=true
        trainData.setName("trainData").cache()
        val bcTrips = sc.broadcast(testData)
        val knownLowerBound = testData.map { case trip => (trip.tripId, 15*math.max(trip.polyline.length - 1, 0)) }.toMap
        val (bcTrySet, trySet, lowerBound) =  (bcTrips, testData, knownLowerBound)


    val tss = trData.map { t1 =>
      bcTrySet.value.map { t0 =>
        t0.polyline.takeRight(2) match {
          case lastSegmentBegin +: lastSegmentEnd +: Nil if t1.weekday == t0.weekday && t1.timeOfDay == t0.timeOfDay && t1.polyline.length < 240 =>

            val (dist, dir) = bce.value.toPolar(lastSegmentEnd - lastSegmentBegin)
            val nearThresh = dist * dist * bcparams.value("dist")

            var i = 0
            var found = false
            var remaining = t1.polyline

            while (i < t1.polyline.length - 1 && !found) {

              val segBegin = remaining.head
              remaining = remaining.tail
              val segEnd = remaining.head

              val (sDist, sDir) = bce.value.toPolar(segEnd - segBegin)

              if (bce.value.lastSegmentNear(lastSegmentBegin, segBegin, nearThresh, dist, sDist, dir, sDir)) {
                found = true
              } else {
                i = i + 1
              }
            }

            if (found) {
              val remainingInT1 = t1.polyline.length - i - 1
              val st1 = StatCounter(math.log(  15.0 * (t0.polyline.length + remainingInT1 - 1 +  bcparams.value("offset"))   + 1))
              var st = st1
              if (t0.originCall == t1.originCall) {
                st = st1.copy().merge(st1.copy())
              }
              if (t0.originStand == t1.originStand) {
                st = st1.copy().merge(st1.copy())
              }
              if (t0.taxiId == t1.taxiId) {
                st = st.merge(st1.copy())
              }
              st
            } else {
              StatCounter()
            }

          case _ => StatCounter()
        }
      }
    }.aggregate(bcTrySet.value.map(zp => StatCounter()))(
                seqOp = (a, b) => a.zip(b).map(p => p._1.merge(p._2)),
                combOp = (a, b) => a.zip(b).map(p => p._1.merge(p._2))
                )

    //val means = tss.map(_.mean)

        val estimates = trySet.zip(tss).map {
          case (trip, stat) =>
          (trip.tripId, (stat.count, math.max( math.exp(stat.mean) - 1, lowerBound(trip.tripId)).toInt))
        }.toMap


    /////////////////

    val results = _cutCvData.map { case (actual, tripData ) =>
      val est = estimates.get(tripData.tripId)
        if (est.exists(s => s._1 > 10 &&  s._2 > 15)) {
          (tripData.tripId, est.get._2, 15.0*math.max(actual - 1, 0))
        } else {
          (tripData.tripId, 560, 15.0*math.max(actual - 1, 0))
        }
    }

    com.sparkydots.kaggle.taxi.Evaluate.logError(sc.parallelize(results.map(p => (p._2.toDouble, p._3))))


////////////////////

    val td: Seq[String] = testData.map(_.tripId)
    val ests2 = td.map { t =>
      val est = estimates.get(t)
      if (est.exists(s => s._1 > 10 &&  s._2 > 15)) {
        (t, est.get._2)
      } else {
        (t, 560)
      }
    }
        com.sparkydots.kaggle.taxi.Save.writeResults(ests, "/Users/renatb/data/kaggle/taxi_trip/htro2.csv")






    //    origTripData.cache()
//    val tss = origTripData.map { trip =>
//      val data = bctd.value.map { thisTrip =>
//        if (bce.value.isPointNear(thisTrip.approximateOrigin, trip.approximateOrigin) &&
//          bce.value.isDirNear(thisTrip.direction, trip.direction)) {
//          StatCounter(math.log(trip.elapsedTime + 1.0))
//        } else {
//          StatCounter()
//        }
//      }
//      data
//    }.aggregate(td.map(p => StatCounter()))(
//        seqOp = (a, b) => a.zip(b).map(p => p._1.merge(p._2)),
//        combOp = (a, b) => a.zip(b).map(p => p._1.merge(p._2))
//        )
//
//      val means = tss.map(_.mean)

//    tripData.setName("tripData").cache()
//    val (data, categoricalFeaturesInfo, callTypes, taxiIDs, originCalls, originStands) = extract.featurize(tripData)
//
//    data.cache()
//    val Array(trainingSet, testingSet, otherTestingSet) = data.randomSplit(Array(0.7, 0.2, 0.1))   //

//    trainingSet.setName("trainingSet").cache()  //
//    testingSet.setName("testingSet").cache()   //
//    otherTestingSet.cache()   //

//    val impurity = "variance"
//    val maxDepth = 5
//    val maxBins = 15118 //64
//    val numTrees = 20
//    val featureSubsetStrategy = "auto"

//    import org.apache.spark.mllib.tree.RandomForest
//    val model = RandomForest.trainRegressor(data, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
//    val model = RandomForest.trainRegressor(trainingSet, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

//    tripData.unpersist()

//    val labelsAndPredictions = testingSet.map { point =>
//      val prediction = model.predict(point.features)
//      (point.label, prediction)
//    }
//
//    com.sparkydots.kaggle.taxi.Evaluate.error(testingSet.map(p => (p.label, model.predict(p.features))))
//    com.sparkydots.kaggle.taxi.Evaluate.error(otherTestingSet.sample(false, 0.1, 101L).map(p => (p.label, model.predict(p.features))))


//    val evals = for {
//      maxDepth <- Array(9, 10, 11)
//      numTrees <- Array(25, 30, 35)
//      maxBins <- Array(64)
//    } yield {
//        val model = RandomForest.trainRegressor(trainingSet, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
//        val accuracy = com.sparkydots.kaggle.taxi.Evaluate.error(testingSet.map(p => (p.label, model.predict(p.features))))
//        val accuracyTrain = com.sparkydots.kaggle.taxi.Evaluate.error(trainingSet.map(p => (p.label, model.predict(p.features))))
//        (maxDepth, numTrees, maxBins, accuracy, accuracyTrain)
//      }

    /////// START Model X     12, 512, 80
    val maxDepthX = 11
    val maxBinsX = 256
    val numTreesX = 60



//    val modelX = RandomForest.trainRegressor(trainingSet, categoricalFeaturesInfo, numTreesX, featureSubsetStrategy, impurity, maxDepthX, maxBinsX)
//
//    val tXte1 = com.sparkydots.kaggle.taxi.Evaluate.error(testingSet.map(p => (p.label, modelX.predict(p.features))))
//    val tXte2 = com.sparkydots.kaggle.taxi.Evaluate.error(otherTestingSet.map(p => (p.label, modelX.predict(p.features))))
//    val tXtr = com.sparkydots.kaggle.taxi.Evaluate.error(trainingSet.map(p => (p.label, modelX.predict(p.features))))
    /////// STOP Model X

//    val labelsAndPredictions2 = otherTestingSet.sample(false, 0.02).map { point =>
//      val prediction = model.predict(point.features)
//      (point.label, prediction)
//    }

//    com.sparkydots.kaggle.taxi.Evaluate.error(labelsAndPredictions2)

    //model dep 9, bins 64, trees 50

    //    trainingSet.unpersist()

//    val bcCallTypes = sc.broadcast(callTypes)
//    val bcOriginStands = sc.broadcast(originStands)
//    val bcKnownLowerBound = sc.broadcast(knownLowerBound)
//
//    val estimates2 = td.zip(means).map { case (trip, mean) =>
//      if (trip.elapsedTime < 16.0) {
//        (trip.tripId, math.max(560, knownLowerBound(trip.tripId)).toInt)
//      } else {
//        (trip.tripId, math.max(math.exp(mean) - 1, knownLowerBound(trip.tripId)).toInt)
//      }
//    }.toMap

//    val ests2 = estimates2.toSeq.sortBy(_._1.drop(1).toInt)
//
//    val ests2 = tripIds.map{ id => (id, estimates(id)) }

//    com.sparkydots.kaggle.taxi.Save.writeResults(ests, "/Users/renatb/data/kaggle/taxi_trip/htro2.csv")


  }
}
