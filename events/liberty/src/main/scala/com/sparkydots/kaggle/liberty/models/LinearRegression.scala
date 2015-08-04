package com.sparkydots.kaggle.liberty.models

import com.sparkydots.spark.dataframe.ReadWrite
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.sql.{DataFrame, SQLContext}

object LinearRegression extends Serializable {

  def doit(sqlContext: SQLContext, rw: ReadWrite, knownFeats: DataFrame, lbFeats: DataFrame): Unit = {

    val parsedData = knownFeats.map { row => LabeledPoint(row.getDouble(0), row.getAs[Vector](1)) }

    // Building the model
    val numIterations = 10

    parsedData.cache()

    val model = LinearRegressionWithSGD.train(parsedData, numIterations)

    // Evaluate model on training examples and compute training error
    val valuesAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println("training Mean Squared Error = " + MSE)

    parsedData.unpersist()

    // Save and load model
   // model.save(sc, "myModelPath")
    //val sameModel = LinearRegressionModel.load(sc, "myModelPath")
  }
}


