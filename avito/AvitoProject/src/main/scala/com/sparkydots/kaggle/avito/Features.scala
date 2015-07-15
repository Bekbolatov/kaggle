package com.sparkydots.kaggle.avito

import scala.util.hashing.MurmurHash3
import scala.math._

object Features {


  def hash(feature: String, numBuckets: Int) : Int = {
    return (abs(MurmurHash3.stringHash(feature)) % numBuckets)
  }

}
