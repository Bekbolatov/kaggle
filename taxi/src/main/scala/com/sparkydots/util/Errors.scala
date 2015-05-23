package com.sparkydots.util


object Errors {

  def diffsSq(xy: (Double, Double)): Double = math.pow(xy._1 - xy._2, 2)
  def diffLogsSq(xy: (Double, Double), a: Double = 1.0): Double = math.pow(math.log(xy._1 + a) - math.log(xy._2 + a), 2)

}
