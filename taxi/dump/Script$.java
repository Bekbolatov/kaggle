package com.sparkydots.kaggle.taxi;

object Script {


   def run(sc: SparkContext) = {

     val sqlContext = new SQLContext(sc)

 //    val data = readData(sqlContext, "train1")
 //    //cars.select("TRIP_ID")
 //
 //    val splits = data.randomSplit(Array(0.7, 0.3))
 //    val (trainingData, testData) = (splits(0), splits(1))
 //
 //    val categoricalFeaturesInfo = Map[Int, Int]()
 //    val impurity = "variance"
 //    val maxDepth = 5
 //    val maxBins = 32
 //
 //    val model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
 //
 //    // Evaluate model on test instances and compute test error
 //    val labelsAndPredictions = testData.map { point =>
 //      val prediction = model.predict(point.features)
 //      (point.label, prediction)
 //    }
 //    val testMSE = math.sqrt(labelsAndPredictions.map { case (v, p) => math.pow(math.log(v+1) - math.log(p+1), 2) }.mean())
 //    println("Test Mean Squared Error = " + testMSE)
 //    println("Learned regression tree model:\n" + model.toDebugString)

     //
     //    // Load and parse the data file.
     //    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
     //    // Split the data into training and test sets (30% held out for testing)
     //    val splits = data.randomSplit(Array(0.7, 0.3))
     //    val (trainingData, testData) = (splits(0), splits(1))
     //
     //    // Train a DecisionTree model.
     //    //  Empty categoricalFeaturesInfo indicates all features are continuous.
     //    val categoricalFeaturesInfo = Map[Int, Int]()
     //    val impurity = "variance"
     //    val maxDepth = 5
     //    val maxBins = 32
     //
     //    val model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity,
     //      maxDepth, maxBins)
     //
     //    // Evaluate model on test instances and compute test error
     //    val labelsAndPredictions = testData.map { point =>
     //      val prediction = model.predict(point.features)
     //      (point.label, prediction)
     //    }
     //    val testMSE = labelsAndPredictions.map { case (v, p) => math.pow((v - p), 2) }.mean()
     //    println("Test Mean Squared Error = " + testMSE)
     //    println("Learned regression tree model:\n" + model.toDebugString)
     //
     //    // Save and load model
     //    model.save(sc, "myModelPath")
     //    val sameModel = DecisionTreeModel.load(sc, "myModelPath")

   }
 }
