package com.sparkydots.kaggle.avito.optimization

import com.sparkydots.kaggle.avito.optimization.BLAS._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.optimization.Gradient


class LogLossGradient extends Gradient {
  /**
   * When `x` is positive and large, computing `math.log(1 + math.exp(x))` will lead to arithmetic
   * overflow. This will happen when `x > 709.78` which is not a very large number.
   * It can be addressed by rewriting the formula into `x + math.log1p(math.exp(-x))` when `x > 0`.
   *
   * @param x a floating-point value as input.
   * @return the result of `math.log(1 + math.exp(x))`.
   */

  val bound =  1e-15

  def log1pExp(x: Double): Double = {
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }

  def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val gradient = Vectors.zeros(weights.size)
    val loss = compute(data, label, weights, gradient)
    (gradient, loss)
  }

  override def compute(
                        data: Vector,
                        label: Double,
                        weights: Vector,
                        cumGradient: Vector): Double = {

    val margin = -1.0 * dot(data, weights)

    val multiplier = (1.0 / (1.0 + math.exp(margin))) - label

    axpy(multiplier, data, cumGradient)


    if (label > 0) {
      // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
      log1pExp(margin)
    } else {
      log1pExp(margin) - margin
    }





    //    val diff = dot(data, weights) - label
    //    axpy(diff, data, cumGradient)
    //    diff * diff / 2.0
  }
}



//class LogLossGradient extends Gradient {
//
//  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
//    val diff = dot(data, weights) - label
//    val loss = diff * diff / 2.0
//    val gradient = data.copy
//    scal(diff, gradient)
//    (gradient, loss)
//  }
//
//  override def compute(
//                        data: Vector,
//                        label: Double,
//                        weights: Vector,
//                        cumGradient: Vector): Double = {
//
//    val diff = dot(data, weights) - label
//
//    axpy(diff, data, cumGradient)
//
//    diff * diff / 2.0 // return loss value
//  }
//}


