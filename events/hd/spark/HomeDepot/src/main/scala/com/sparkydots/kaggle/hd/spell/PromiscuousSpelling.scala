package com.sparkydots.kaggle.hd.spell

import com.sparkydots.kaggle.hd.load.Product

case class PromiscuousSpelling(product: Product) extends Serializable {
  val ALPHABET = "abcdefghijklmnopqrstuvwxyz".toSeq
  val title_words = product.title.split(" ").toSet
  val descr_words = product.desc.split(" ").toSet
  val attrs_keys_values_words = product.attrs.flatMap {
    case (k, v) => k.split(" ").toSeq ++ v.split(" ").toSeq
  }.toSet

  val brand_words = product.brand.map(_.split(" ").toSeq).getOrElse(Seq[String]()).toSet


  val query_words = product.queries.map {
    case (id, q) => (id, q.split(" "))
  }

  def matches() = {
    product.queries.map { case (id, qs) =>
      val qss = qs.split(" ").toSeq
      val matches_title = qss.map(q => findMatchTitle(q)).reduce((a,b) => a union b)
      val matches_descr = qss.map(q => findMatchDesc(q)).reduce((a,b) => a union b)
      val matches_attr = qss.map(q => findMatchAttrs(q)).reduce((a,b) => a union b)
      val matches_brand = qss.map(q => findMatch(q, brand_words)).reduce((a,b) => a union b)

      (id, matches_title, matches_descr, matches_attr, matches_brand)
    }
  }

  def findMatchTitle(word: String) = findMatch(word, title_words)
  def findMatchDesc(word: String) = findMatch(word, descr_words)
  def findMatchAttrs(word: String) = findMatch(word, attrs_keys_values_words)

  def findMatch(word: String, base_set: Set[String]): Set[String] = {
    if (base_set.contains(word)) {
      Set(word)
    } else {
      if (word.length > 3) {
        val word_edits_1 = word_edits(word)
        val inter1 = word_edits_1.intersect(base_set)
        if (inter1.nonEmpty) {
          inter1
        } else {
          if (word.length > 4) {
            val word_edits_2 = word_edits_1.flatMap(w1 => word_edits(w1))
            val inter2 = word_edits_2.intersect(base_set)
            if (inter2.nonEmpty) {
              inter2
            } else {
              Set[String]()
            }
          } else {
            Set[String]()
          }
        }
      } else {
        Set[String]()
      }

    }
  }

  def word_edits(word: String): Set[String] = {
    (0 to word.length).flatMap { i =>
      val a = word.take(i)
      val b = word.drop(i)

      var changes = Seq[String]()
      changes = changes :+ (a ++ b.drop(1))  // deletes
      if (b.length > 1) {
        changes = changes :+ a ++ b.slice(1, 2) ++ b.take(1) + b.drop(2)  // transposes
      }
      changes = changes ++ ALPHABET.map(c => a ++ c.toString ++ b.drop(1))  // replaces
      changes = changes ++ ALPHABET.map(c => a ++ c.toString ++ b)  // inserts
      changes
    }.toSet
  }

}
