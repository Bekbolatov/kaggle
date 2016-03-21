package com.sparkydots.kaggle.hd.spell

import com.sparkydots.kaggle.hd.load.{CleanText, Product}
import org.scalatest.{Matchers, FlatSpec}

class SpellTest extends FlatSpec with Matchers {
  "Spell" should "check" in {
    val p = Product(1, "shawe bolts", "these bolts are awesome", Seq(), Seq((2, "shawer bolt")))
    val s = PromiscuousSpelling(p)

    println(s.findMatchTitle("shawer"))
    println(s.findMatchTitle("shwae"))
    println(s.findMatchTitle("show"))
    println(s.findMatchTitle("shuw"))
    println(s.findMatchTitle("shu"))
  }


  it should "check bakers rack" in {
    val p = Product(1, "baker rack", "these bolts are awesome", Seq(), Seq((2, "shawer bolt")))
    val s = PromiscuousSpelling(p)

    println("---")
    println(s.findMatchTitle("baker"))
    println(s.findMatchTitle("bakers"))
  }

  it should "work with tokenizer" in {
    val title = CleanText.clean("Baker's rack")
    val q = CleanText.clean("bakers rack")
    val p = Product(1, title, "these bolts are awesome", Seq(), Seq((2, q)))
    val s = PromiscuousSpelling(p)

    println("===")
    println(title)
    println(q)
    s.matches_title().foreach(println)

  }
}
