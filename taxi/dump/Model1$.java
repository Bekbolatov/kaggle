package com.sparkydots.kaggle.taxi;

/**
  * @author Renat Bekbolatov (renatb@sparkydots.com) 5/16/15 12:09 PM
  */
        object Model1 {

           def run(sc: SparkContext): Unit = {

             val sqlContext = new SQLContext(sc)
             val df = readFile(sqlContext, "train")
         //    val data = df.flatMap(tripData)
         //    val splits = data.randomSplit(Array(0.7, 0.3))
         //    val (trainingData, testData) = (splits(0), splits(1))
         //
         //    val model2 = means(trainingData)
         //
         //    val labelsAndPredictions = testData.map { point =>
         //      val prediction = model2(point._1)
         //      (point._2, prediction)
         //    }
         //    //val testMSE = math.sqrt(labelsAndPredictions.map { case (v, p) => math.pow(math.log(v+1) - math.log(p+1), 2) }.mean())
         //    // Already as log(x+1)
         //    val testMSE = math.sqrt(labelsAndPredictions.map { case (v, p) => math.pow(v - p, 2) }.mean())
         //    println("Test Root Mean Squared Logarithmic Error (RMSLE) = " + testMSE)
         //
         //
         //    // Submission data
         //    val subm_df = readFile(sqlContext, "test")
         //    val subm_data = subm_df.flatMap(tripData)
         //
         //    subm_data.map(p => (p, model2(p._1)))




             val Array(tradata, tesdata) = data.randomSplit(Array(0.7, 0.3))
         //    val bytwo = tradata.map(p => ((p._1,p._2), p._4))
         //    val m = means(trainingData)
             val m = means( tradata.map(p => ((p._1,p._2), p._4)) )

             val p = tesdata.map(p => (p, m( (p._1, p._2) )))
             error(p)

           }
         }
