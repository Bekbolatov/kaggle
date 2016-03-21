package com.sparkydots.kaggle.hd.spell

import com.sparkydots.kaggle.hd.load.Product

case class PromiscuousSpelling(product: Product) extends Serializable {
  val ALPHABET = "abcdefghijklmnopqrstuvwxyz".toSeq
  val title_words = product.title.split(" ").toSet
  val descr_words = product.desc.split(" ").toSet
  val attrs_keys_values_words = product.attrs.flatMap {
    case (k, v) => k.split(" ").toSet ++ v.split(" ").toSet
  }



  val query_words = product.queries.map {
    case (id, q) => (id, q.split(" "))
  }

  def matches_title() = {
    product.queries.map { case (id, qs) =>
      val qss = qs.split(" ")
      (id, qss.map(q => findMatchTitle(q)).toSeq.exists(p => p))
    }
  }

  def findMatchTitle(word: String) = findMatch(word, title_words)

  def findMatch(word: String, base_set: Set[String]) = {
    if (base_set.contains(word)) {
      true
    } else {
      val word_edits_1 = word_edits(word)
      if (word_edits_1.intersect(base_set).nonEmpty) {
        true
      } else {
        val word_edits_2 = word_edits_1.flatMap(w1 => word_edits(w1))
        if (word_edits_2.intersect(base_set).nonEmpty) {
          true
        } else {
          false
        }
      }
    }
  }

  def word_edits(word: String): Set[String] = {
    (0 to word.length).flatMap { i =>
      val a = word.take(i)
      val b = word.drop(i)

      var changes = Seq[String]()
      changes = changes :+ (a ++ b.drop(1))  // deletes
//      println(a ++ b.drop(1))
      if (b.length > 1) {
        changes = changes :+ a ++ b.slice(1, 2) ++ b.take(1) + b.drop(2)  // transposes
      }
      changes = changes ++ ALPHABET.map(c => a ++ c.toString ++ b.drop(1))  // replaces
      changes = changes ++ ALPHABET.map(c => a ++ c.toString ++ b)  // inserts
      changes
    }.toSet
  }



}


//
//
//def edits1(word):
//splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]

//deletes    = [a + b[1:] for a, b in splits if b]
//transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
//replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
//inserts    = [a + c + b     for a, b in splits for c in alphabet]
//return set(deletes + transposes + replaces + inserts)
