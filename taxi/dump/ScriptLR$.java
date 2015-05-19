package com.sparkydots.kaggle.taxi;

object ScriptLR {

   def run(sc: SparkContext) = {

     val sqlContext = new SQLContext(sc)
 //    val data = readData(sqlContext, "train2")
 //
 //    val splits = data.randomSplit(Array(0.7, 0.3))
 //    val (trainingData, testData) = (splits(0), splits(1))
 //
 //    // Building the model
 //    val numIterations = 100
 //    val model = LinearRegressionWithSGD.train(trainingData, numIterations)
 //
 //    // Evaluate model on test instances and compute test error
 //    val labelsAndPredictions = testData.map { point =>
 //      val prediction = model.predict(point.features)
 //      (point.label, prediction)
 //    }
 //    val testMSE = math.sqrt(labelsAndPredictions.map { case (v, p) => math.pow(math.log(v + 1) - math.log(p + 1), 2) }.mean())
 //    println("Test Mean Squared Error = " + testMSE)
   }
 }
