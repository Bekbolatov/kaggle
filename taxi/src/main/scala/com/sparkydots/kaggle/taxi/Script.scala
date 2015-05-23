package com.sparkydots.kaggle.taxi

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext


object Script {

  def run(sc: SparkContext, sqlContext: SQLContext): Unit = {

    import org.apache.spark.mllib.tree.RandomForest

    val extract = new com.sparkydots.kaggle.taxi.Extract(sc, sqlContext)
//    val (testData, tripData, rawTripDataAll, tripIds, knownLowerBound) = extract.data(hdfs = true)  /// s3=true
    val (testData, tripData, rawTripDataAll, tripIds, knownLowerBound) = extract.data(s3 = true)  /// s3=true

    tripData.cache()
    val (data, categoricalFeaturesInfo, callTypes, originStands) = extract.featurize(tripData)

    val Array(trainingSet, testingSet) = data.randomSplit(Array(0.7, 0.3))   //

    trainingSet.cache()  //
    testingSet.cache()   //

    val impurity = "variance"
    val maxDepth = 9
    val maxBins = 64
    val numTrees = 50
    val featureSubsetStrategy = "auto"

//    val model = RandomForest.trainRegressor(data, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    val model = RandomForest.trainRegressor(trainingSet, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    tripData.unpersist()

    val labelsAndPredictions = testingSet.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }


    com.sparkydots.kaggle.taxi.Evaluate.error(labelsAndPredictions)

    //    trainingSet.unpersist()

    val bcCallTypes = sc.broadcast(callTypes)
    val bcOriginStands = sc.broadcast(originStands)
    val bcKnownLowerBound = sc.broadcast(knownLowerBound)

    val estimates = testData.map { case (_, trip) =>
      if (trip.elapsedTime < 16.0) {
        (trip.tripId, math.max(700, bcKnownLowerBound.value(trip.tripId)).toInt)
      } else {
        val point = com.sparkydots.kaggle.taxi.Extract.featureVector(0.0, trip, bcCallTypes.value, bcOriginStands.value)
        (trip.tripId, math.max(math.exp(model.predict(point.features) - 1), bcKnownLowerBound.value(trip.tripId)).toInt)
      }
    }.collect.toMap


    val ests = tripIds.map{ id => (id, estimates(id)) }

    com.sparkydots.kaggle.taxi.Save.writeResults(ests)


  }
}
