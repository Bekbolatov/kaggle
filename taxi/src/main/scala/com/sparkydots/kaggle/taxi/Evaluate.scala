package com.sparkydots.kaggle.taxi

import com.sparkydots.util.Errors._
import org.apache.spark.rdd.RDD

object Evaluate extends Serializable {

  def prepareTestSet(someTripData: RDD[TripData]): RDD[TripData] = {
      ???
  }

  /*


case class TripData(
                     tripId: String,
                     callType: String,
                     originCall: Option[String],
                     originStand: Option[String],
                     taxiID: Int,
                     timestamp: DateTime,
                     dayType: String,    // always: "A"
                     missing: Boolean,   // most always: False
                     rawPathPoints: Seq[Point],
                     pathPoints: Seq[Point],
                     pathComponents: (Int, Seq[Double]),   //number of proposed disjoint paths (out of which the longest will be chosen) and also jump values
                     pathSegments: Seq[PathSegment],
                     travelTime: Double)  // in training data, 99.4% is below 3600s (1 HR)


   */


  def error(preds: RDD[(Double, Double)]) = math.sqrt(preds.map(diffsSq(_)).sum / preds.count)

  def error(preds: Array[(Double, Double)]) = math.sqrt(preds.map(diffsSq(_)).sum / preds.length)

  def logError(preds: RDD[(Double, Double)]) = math.sqrt(preds.map(diffLogsSq(_)).sum / preds.count)

}
