package com.sparkydots.kaggle.taxi

import com.sparkydots.util.io.FileIO

/**
 * @author Renat Bekbolatov (renatb@sparkydots.com) 5/22/15 5:02 PM
 */
object Save {

  def writeResults(results: Seq[(String, Int)], path: String = "/home/hadoop/results.csv") ={
    val withHeader = "\"TRIP_ID\",\"TRAVEL_TIME\"" +: results.map(p => s"${p._1},${p._2}")
    FileIO.write(withHeader, path)
  }

}
