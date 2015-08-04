package com.sparkydots.kaggle.liberty

import com.sparkydots.kaggle.liberty.dataset.DataSet
import com.sparkydots.spark.dataframe.ReadWrite
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object Script extends Serializable {

  val localPath:String = "/Users/rbekbolatov/data/kaggle/liberty/scratch"
  val location: String = "data/kaggle/liberty"
  val processedDir: String = "apple"

  def run(sc: SparkContext, featurePrefix: String = "ADAM") = {

    initLoggers
    val sqlContext = new SQLContext(sc)
    val rw = new ReadWrite(sqlContext, location, processedDir, localPath)

    val (typedKnown, typedLb) = DataSet.getDatasets(rw)

    (sqlContext, rw, typedKnown, typedLb)

  }

  def initLoggers() = {
    Logger.getLogger("amazon.emr.metrics").setLevel(Level.OFF)
    Logger.getLogger("com.amazon.ws.emr").setLevel(Level.WARN)
    Logger.getLogger("com.amazonaws").setLevel(Level.WARN)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("com").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
  }

}


/*

import org.apache.spark.sql.SQLContext
import com.sparkydots.spark.dataframe.{RichDataFrame, ReadWrite}
import com.sparkydots.kaggle.liberty._
import com.sparkydots.kaggle.liberty.models._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import com.sparkydots.spark.dataframe.ReadWrite
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{DataFrame, SQLContext}
import com.sparkydots.kaggle.liberty.transformers.BasicNumericColumnsFeaturizer
import com.sparkydots.spark.dataframe.{ReadWrite, RichDataFrame}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions.col




import com.sparkydots.kaggle.liberty._
val localPath:String = "/Users/rbekbolatov/data/kaggle/liberty/scratch"
val location: String = "data/kaggle/liberty"
val processedDir: String = "apple"


val rw = new ReadWrite(sqlContext, location, processedDir, localPath)

val (knownFeats, lbFeats) = DataSet.getDatasets(sc, sqlContext, rw, "ADAM", true, false)

LinearRegression.doit(sqlContext, rw, knownFeats, lbFeats)


Script.run(sc, sqlContext, new transformers.BasicNumericColumnsFeaturizer(
 */