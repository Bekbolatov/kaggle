package com.sparkydots.kaggle.taxi

import com.sparkydots.util.geo.{Point, Earth}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.util.StatCounter


object Script {

  def run(sc: SparkContext, sqlContext: SQLContext): Unit = {

    import org.apache.spark.mllib.tree.RandomForest

    val extract = new com.sparkydots.kaggle.taxi.Extract(sc, sqlContext)
    val (testData, origTripData, tripData, tripDataFiltered, rawTestDataAll, rawTripDataAll, tripIds, knownLowerBound) = extract.data(hdfs = true)  /// s3=true
//    val (testData, origTripData, tripData, tripDataFiltered, rawTestDataAll, rawTripDataAll, tripIds, knownLowerBound) = extract.data(s3 = true)  /// s3=true

    val td = testData.map(_._2).collect().toSeq.sortBy(_.tripId.drop(1).toInt)

    val earth: Earth = Earth(Point(41.14, -8.62))
    val bctd = sc.broadcast(td)
    val bce = sc.broadcast(earth)

    origTripData.cache()
    val tss = origTripData.map { trip =>
      val data = bctd.value.map { thisTrip =>
        if (bce.value.isPointNear(thisTrip.approximateOrigin, trip.approximateOrigin) &&
          bce.value.isDirNear(thisTrip.direction, trip.direction)) {
          StatCounter(math.log(trip.elapsedTime + 1.0))
        } else {
          StatCounter()
        }
      }
      data
    }.aggregate(td.map(p => StatCounter()))(
        seqOp = (a, b) => a.zip(b).map(p => p._1.merge(p._2)),
        combOp = (a, b) => a.zip(b).map(p => p._1.merge(p._2))
        )

      val means = tss.map(_.mean)

    tripData.cache()
    val (data, categoricalFeaturesInfo, callTypes, originStands) = extract.featurize(tripData)

    data.cache()
    val Array(trainingSet, testingSet, otherTestingSet) = data.randomSplit(Array(0.7, 0.2, 0.1))   //

    trainingSet.cache()  //
    testingSet.cache()   //
    otherTestingSet.cache()   //

    val impurity = "variance"
    val maxDepth = 5
    val maxBins = 64
    val numTrees = 20
    val featureSubsetStrategy = "auto"

//    val model = RandomForest.trainRegressor(data, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    val model = RandomForest.trainRegressor(trainingSet, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    tripData.unpersist()

    val labelsAndPredictions = testingSet.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    com.sparkydots.kaggle.taxi.Evaluate.error(testingSet.map(p => (p.label, model.predict(p.features))))
    com.sparkydots.kaggle.taxi.Evaluate.error(otherTestingSet.sample(false, 0.02, 101L).map(p => (p.label, model.predict(p.features))))


    val evals = for {
      maxDepth <- Array(9, 10, 11)
      numTrees <- Array(25, 30, 35)
      maxBins <- Array(64)
    } yield {
        val model = RandomForest.trainRegressor(trainingSet, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
        val accuracy = com.sparkydots.kaggle.taxi.Evaluate.error(testingSet.map(p => (p.label, model.predict(p.features))))
        val accuracyTrain = com.sparkydots.kaggle.taxi.Evaluate.error(trainingSet.map(p => (p.label, model.predict(p.features))))
        (maxDepth, numTrees, maxBins, accuracy, accuracyTrain)
      }

    /////// START Model X     12, 512, 80
    val maxDepthX = 11
    val maxBinsX = 256
    val numTreesX = 60



    val modelX = RandomForest.trainRegressor(trainingSet, categoricalFeaturesInfo, numTreesX, featureSubsetStrategy, impurity, maxDepthX, maxBinsX)

    val tXte1 = com.sparkydots.kaggle.taxi.Evaluate.error(testingSet.map(p => (p.label, modelX.predict(p.features))))
    val tXte2 = com.sparkydots.kaggle.taxi.Evaluate.error(otherTestingSet.map(p => (p.label, modelX.predict(p.features))))
    val tXtr = com.sparkydots.kaggle.taxi.Evaluate.error(trainingSet.map(p => (p.label, modelX.predict(p.features))))
    /////// STOP Model X

    val labelsAndPredictions2 = otherTestingSet.sample(false, 0.02).map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    com.sparkydots.kaggle.taxi.Evaluate.error(labelsAndPredictions2)

    //model dep 9, bins 64, trees 50

    //    trainingSet.unpersist()

    val bcCallTypes = sc.broadcast(callTypes)
    val bcOriginStands = sc.broadcast(originStands)
    val bcKnownLowerBound = sc.broadcast(knownLowerBound)

    val estimates2 = td.zip(means).map { case (trip, mean) =>
      if (trip.elapsedTime < 16.0) {
        (trip.tripId, math.max(660, knownLowerBound(trip.tripId)).toInt)
      } else {
        (trip.tripId, math.max(math.exp(mean) - 1, knownLowerBound(trip.tripId)).toInt)
      }
    }.toMap

//    val ests = estimates.toSeq.sortBy(_._1.drop(1).toInt)

    val ests2 = tripIds.map{ id => (id, estimates2(id)) }

    com.sparkydots.kaggle.taxi.Save.writeResults(ests2, "/Users/renatb/data/kaggle/taxi_trip/htro.csv")


  }
}
