package com.sparkydots.kaggle.avito.features

import org.scalatest.{Matchers, FlatSpec}

class TestFeatures extends FlatSpec with Matchers {

  "Features" should "work with sentences" in {
    FeatureHashing.sentenceFeatures("query", "горные лыжи rossi") shouldBe Seq((2467,1.0), (19351,1.0), (24779,1.0))
    FeatureHashing.sentenceFeatures("query", "лыжи") shouldBe Seq((2467,1.0))
    FeatureHashing.sentenceFeatures("query", "термометр детский") shouldBe Seq((16209,1.0), (31819,1.0))
    FeatureHashing.sentenceFeatures("query", "") shouldBe Seq()
  }

  it should "work with lists of Ints" in {
    FeatureHashing.intsFeatures("searchParams", Seq(1, 2, 3)) shouldBe Seq((10344,1.0), (23039,1.0), (23802,1.0))
    FeatureHashing.hashValues("sd", Seq(1,2,3): _*) shouldBe Seq(2207, 30128, 24448)
    FeatureHashing.hashValues("sd", Seq(): _*) shouldBe Seq()
    FeatureHashing.hashValues("sd") shouldBe Seq()
  }

  it should "count param overlap" in {
    FeatureHashing.paramOverlap(Seq(1, 1, 2, 3), Seq(2, 3, 4)) shouldBe 0
    FeatureHashing.paramOverlap(Seq(), Seq(2, 3, 4)) shouldBe 0
    FeatureHashing.paramOverlap(Seq(), Seq()) shouldBe 0

    FeatureHashing.paramOverlap(Seq(1, 2, 3), Seq(2, 3, 4)) shouldBe 2
    FeatureHashing.paramOverlap(Seq(1, 2, 3), Seq(1, 2, 3, 4)) shouldBe 3
    FeatureHashing.paramOverlap(Seq(1, 2, 3, 6, 7), Seq(1, 2, 3, 4)) shouldBe 3
  }
}
