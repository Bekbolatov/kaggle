package com.sparkydots.kaggle.hd.features

import com.sparkydots.kaggle.hd.load.Queries

object Features extends Serializable {

  case class SpellCheck(product: Queries) {

    def correct(word: String): String = {
      word
    }
  }

  // Need to treat numbers differently
  // Find matches for incoming query words
  def testFeatures(product: Queries): Seq[(Int, Seq[Boolean])] = {
    val spell_check_model = SpellCheck(product)
    val words = product.queries.map { case (id: Int, q: String) =>
      (id, q.split(" ").map(word => spell_check_model.correct(word)))
    }
    words.map { case (id, ws) => (id, Seq(true))}
  }
}
