package com.sparkydots.kaggle.hd.spell

import com.sparkydots.kaggle.hd.load.Product

case class PromiscuousSpelling(product: Product) extends Serializable {
  val ALPHABET = "abcdefghijklmnopqrstuvwxyz".toSeq

  val title_tokens = product.title.split(" ").toSeq
  val title_words = title_tokens.toSet
  val title_word_pairs = if (title_tokens.size > 1) title_tokens.sliding(2).map(x => (x(0), x(1))).toSet else Set[(String, String)]()

  val descr_tokens = product.desc.split(" ").toSeq
  val descr_words = descr_tokens.toSet
  val descr_word_pairs = if (descr_tokens.size > 1) descr_tokens.sliding(2).map(x => (x(0), x(1))).toSet else Set[(String, String)]()

  val attrs_keys_values_words = product.attrs.flatMap {
    case (k, v) => k.split(" ").toSeq ++ v.split(" ").toSeq
  }.toSet

  val brand_tokens = product.brand.map(_.split(" ").toSeq).getOrElse(Seq[String]())
  val brand_words = brand_tokens.toSet
  val brand_word_pairs = if (brand_tokens.size > 1) brand_tokens.sliding(2).map(x => (x(0), x(1))).toSet else Set[(String, String)]()

  val query_words = product.queries.map {
    case (id, q) =>
      val tokens = q.split(" ").toSeq
      (id, tokens.toSet, if (tokens.size > 1) tokens.sliding(2).map(x => (x(0), x(1))).toSet else Set[(String, String)]())
  }

  def matches() = {
    query_words.map { case (id, qs, qqs) =>

      val matches_title = qs.map(q => findMatch(q, title_words)).fold(Set[String]())((a,b) => a union b).toSeq
      val matches_title_pairs = qqs.map(q => findMatchPairs(q, title_word_pairs)).fold(Set[String]())((a,b) => a union b).toSeq

      val matches_descr = qs.map(q => findMatch(q, descr_words)).fold(Set[String]())((a,b) => a union b).toSeq
      val matches_descr_pairs = qqs.map(q => findMatchPairs(q, descr_word_pairs)).fold(Set[String]())((a,b) => a union b).toSeq

      val matches_attr = qs.map(q => findMatch(q, attrs_keys_values_words)).fold(Set[String]())((a,b) => a union b).toSeq

      val matches_brand = qs.map(q => findMatch(q, brand_words)).fold(Set[String]())((a,b) => a union b).toSeq
      val matches_brand_pairs = qqs.map(q => findMatchPairs(q, brand_word_pairs)).fold(Set[String]())((a,b) => a union b).toSeq

      (id, (matches_title, matches_title_pairs), (matches_descr, matches_descr_pairs), matches_attr, (matches_brand, matches_brand_pairs))

    }
  }

  def findMatchPairs(words: (String, String), base_set: Set[(String, String)]): Set[String] = {
    findMatch(words._1 + "_" + words._2, base_set.map(x => x._1 + "_" + x._2))
  }


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
