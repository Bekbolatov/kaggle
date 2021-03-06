package com.sparkydots.kaggle.avito.features

import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{Matchers, FlatSpec}

class TestFeatures extends FlatSpec with Matchers {

  val hasher = new FeatureHashing(15)

  "Features" should "work with sentences" in {
//    hasher.sentenceFeatures("query", "горные лыжи rossi") shouldBe Seq((2467,1.0), (19351,1.0), (24779,1.0))
//    hasher.sentenceFeatures("query", "лыжи") shouldBe Seq((2467,1.0))
//    hasher.sentenceFeatures("query", "термометр детский") shouldBe Seq((16209,1.0), (31819,1.0))
//    hasher.sentenceFeatures("query", "") shouldBe Seq()
  }

  it should "generate featurure subset" in {
    val sf = new SelectFeatures(100, 5, 80, 5)
    println(sf.rejiggle())

    val sf2 = new SelectFeatures(100, 10, 5, 5)
    println(sf2.rejiggle())

    val v = Vectors.sparse(100, Seq( (0,1.0), (30, 2.0)))
    println(sf2.transform(v))

  }

//  it should "work with lists of Ints" in {
//    hasher.intsFeatures("searchParams", Seq(1, 2, 3)) shouldBe Seq((10344,1.0), (23039,1.0), (23802,1.0))
//    hasher.hashValues("sd", Seq(1,2,3): _*) shouldBe Seq(2207, 30128, 24448)
//    hasher.hashValues("sd", Seq(): _*) shouldBe Seq()
//    hasher.hashValues("sd") shouldBe Seq()
//  }
//
//  it should "count param overlap" in {
//    hasher.paramOverlap(Seq(1, 1, 2, 3), Seq(2, 3, 4)) shouldBe 0
//    hasher.paramOverlap(Seq(), Seq(2, 3, 4)) shouldBe 0
//    hasher.paramOverlap(Seq(), Seq()) shouldBe 0
//
//    hasher.paramOverlap(Seq(1, 2, 3), Seq(2, 3, 4)) shouldBe 2
//    hasher.paramOverlap(Seq(1, 2, 3), Seq(1, 2, 3, 4)) shouldBe 3
//    hasher.paramOverlap(Seq(1, 2, 3, 6, 7), Seq(1, 2, 3, 4)) shouldBe 3
//  }
}
