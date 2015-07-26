package com.sparkydots.kaggle.avito.features

import java.io.FileWriter

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{SQLContext, DataFrame}

import scala.util.Random.{shuffle, nextInt}

class SelectFeatures(before: Int, startWith: Int, remove:Int, numAddInteractions: Int, filename: Option[String] = None) extends  Serializable {
  val add1 = math.sqrt(numAddInteractions).floor.toInt
  val add2 = math.sqrt(numAddInteractions).ceil.toInt
  assert(add1 + add2 < before - remove, "`add`s should be at most half of length of reduced feature set (`before` - `remove`)")
  val keptIndices = (0 until startWith).toList
  val softIndices = (startWith until before).toList

  // Seq(    (   (trId, remFeats..., addFeatInters...) , ( trainError, validateError) )  )
  var allTransforms: Seq[ ( (Int, Seq[Int], Seq[(Int, Int)]), (Double, Double))  ] = Seq.empty
  // trId
  var transformId = 0
  // (trId, remFeats..., addFeatInters...)
  var currentTransform: (Int, Seq[Int], Seq[(Int, Int)]) = (0, Seq.empty, Seq.empty)
  var currentKepts: Seq[(Int, Int)] = Seq.empty
  var currentInteractions: Seq[(Int, (Int, Int))] = Seq.empty
  var currentOffset = 0
  var currentNumFeatures = 0

  var bestTransformId = 0
  var bestTransform: (Int, Seq[Int], Seq[(Int, Int)]) = (0, Seq.empty, Seq.empty)
  var bestKepts: Seq[(Int, Int)] = Seq.empty
  var bestInteractions: Seq[(Int, (Int, Int))] = Seq.empty
  var bestOffset = 0
  var bestNumFeatures = 0
  var bestValidateError = 0.04571

  rejiggle()

  def rejiggle(): (Int, Seq[Int], Seq[(Int, Int)]) = {
    transformId = transformId + 1

    val newIndices = (keptIndices ++ shuffle(softIndices).take(before - startWith - remove)).sorted
    val left = shuffle(newIndices).take(add1)
    val right = shuffle(newIndices.toSet.diff(left.toSet).toSeq).take(add2)

    val addedInteractions = (for (a <- left; b <- right) yield (a, b)).toSeq

    currentTransform = (transformId, newIndices, addedInteractions)
    currentKepts = newIndices.zipWithIndex.map(_.swap)
    currentInteractions = addedInteractions.zipWithIndex.map(_.swap)
    currentOffset = newIndices.size
    currentNumFeatures = newIndices.size + currentInteractions.size
    currentTransform
  }

  def transform(features: Vector): Vector = {
    val keptFeatures = currentKepts.flatMap { case (i, a) =>
      val fa = features(a)
      if (fa == 0.0) {
        None
      } else {
        Some((i, features(a)))
      }
    }
    val interactions = currentInteractions.flatMap { case (i, (a, b)) =>
      val fa = features(a)
      val fb = features(b)
      if (fa == 0.0 || fb == 0.0) {
        None
      } else {
        Some((i + currentOffset, features(a) * features(b)))
      }
    }

    Vectors.sparse(currentNumFeatures, keptFeatures ++ interactions)
  }

  def transform(lp: LabeledPoint): LabeledPoint = LabeledPoint(lp.label, transform(lp.features))

  def transform(sqlContext: SQLContext,data : DataFrame): DataFrame = {
      import sqlContext.implicits._
      data.map { r => LabeledPoint(r.getDouble(0), transform(r.getAs[Vector](1))) }.toDF()
  }

  def report(trainError: Double, validateError: Double) = {
    allTransforms = allTransforms :+ (currentTransform, (trainError, validateError))


    val betterFound = if (validateError < bestValidateError) {
      bestTransformId = currentTransform._1
      bestTransform = currentTransform
      bestKepts = currentKepts
      bestInteractions = currentInteractions
      bestOffset = currentOffset
      bestNumFeatures = currentNumFeatures
      bestValidateError = validateError
      true
    } else {
      false
    }

    checkpoint(trainError, validateError)
    betterFound
  }

  def setBestTransform() = {
    currentTransform = bestTransform
    currentInteractions = bestInteractions
    currentOffset = bestOffset
    currentNumFeatures = bestNumFeatures
  }

  def checkpoint(trainError: Double, validateError: Double) = {
    filename.map { f =>
      val reportFile = new FileWriter(s"/home/hadoop/${f}.csv", true)
      reportFile.write(f"${currentTransform._1}}:$trainError%1.8f:$trainError%1.8f:${currentTransform._2.mkString(",")}:${currentTransform._3.mkString(",")}\n\n")
      reportFile.close
    }
  }


}
