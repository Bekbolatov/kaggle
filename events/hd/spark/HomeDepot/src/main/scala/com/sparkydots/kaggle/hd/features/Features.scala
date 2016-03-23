package com.sparkydots.kaggle.hd.features

import com.sparkydots.kaggle.hd.load.Product

object Features extends Serializable {

  case class SpellCheck(product: Product) {

    def correct(word: String): String = {
      word
    }
  }

  // Need to treat numbers differently
  // Find matches for incoming query words
  def testFeatures(product: Product): Seq[(Int, Seq[Boolean])] = {
    val spell_check_model = SpellCheck(product)
    val words = product.queries.map { case (id: Int, q: String) =>
      (id, q.split(" ").map(word => spell_check_model.correct(word)))
    }
    words.map { case (id, ws) => (id, Seq(true))}
  }

  def loadLocalMatches() = {
    val pattern = raw"\[([0-9]+),WrappedArray\((.*)\),WrappedArray\((.*)\),WrappedArray\((.*)\),WrappedArray\((.*)\),WrappedArray\((.*)\),WrappedArray\((.*)\),WrappedArray\((.*)\)\]".r
    //[102853,WrappedArray(popcorn, paint),WrappedArray(),WrappedArray(paint),WrappedArray(),WrappedArray(),WrappedArray(),WrappedArray()]
    val testit = "[102853,WrappedArray(popcorn, paint),WrappedArray(),WrappedArray(paint),WrappedArray(),WrappedArray(),WrappedArray(),WrappedArray()]"

    testit match {
      case pattern(id, a1, a2, a3, a4, a5, a6, a7) =>
        println(id)
    }
  }


}
