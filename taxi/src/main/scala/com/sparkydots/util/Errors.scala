package com.sparkydots.util


object Errors {

  def pointErrorSquared(x: Double, y: Double): Double = math.pow(x - y, 2)
  def pointLogErrorSquared(x: Double, y: Double): Double = math.pow(math.log(x) - math.log(y), 2)
  def pointLogPlusOneErrorSquared(x: Double, y: Double): Double = math.pow(math.log(x + 1) - math.log(y + 1), 2)

}
