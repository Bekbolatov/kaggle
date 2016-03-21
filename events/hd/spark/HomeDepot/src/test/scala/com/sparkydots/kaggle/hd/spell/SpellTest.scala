package com.sparkydots.kaggle.hd.spell

import com.sparkydots.kaggle.hd.load.{CleanText, Product}
import org.scalatest.{Matchers, FlatSpec}

class SpellTest extends FlatSpec with Matchers {
  "Spell" should "check" in {
    val p = Product(1, "shawe bolts", "these bolts are awesome", Seq(), Seq((2, "shawer bolt")))
    val s = PromiscuousSpelling(p)

//    println(s.findMatchTitle("shawer"))
//    println(s.findMatchTitle("shwae"))
//    println(s.findMatchTitle("show"))
//    println(s.findMatchTitle("shuw"))
//    println(s.findMatchTitle("shu"))
  }


  it should "check bakers rack" in {
    val p = Product(1, "baker rack", "these bolts are awesome", Seq(), Seq((2, "shawer bolt")))
    val s = PromiscuousSpelling(p)

//    println("---")
//    println(s.findMatchTitle("baker"))
//    println(s.findMatchTitle("bakers"))
  }

  it should "work with tokenizer" in {
    val title = CleanText.clean("Baker's rack")
    val q = CleanText.clean("bakers rack")
    val p = Product(1, title, "these bolts are awesome", Seq(), Seq((2, q)))
    val s = PromiscuousSpelling(p)

    println("===")
    println(title)
    println(q)
    s.matches().foreach(println)

  }


  it should "work with real" in {
    val p1 = Product(
      152288,
      "cooper wiring devices 20 amp 125 volt tamper resistant afci duplex receptacle black",
      "cooper wiring devices 20 amp afci duplex receptacle perfect outlet commercial residential application arc fault circuit interrupters established stable technology used conjunction various loads home both afci breakers afci receptacles device protects against arc faults resulting damaged insulation causes lead electrical fires increased safety receptacle tamper resistant meets 2014 nec article 406 12 states 15 amp 20 amp 125 volt ac receptacles installed dwelling units must tamper resistant meets exceeds 10 ka short circuit testing underwriters laboratories ul 1699a ul 498 safety standards ground termination back wire clamp provides secure wiring reduces installation time line side terminals backed out stakedfast installation downstream receptacles wired load side 20 amp feed rating offers full protection terminal mounting screws tri combo heads versatile installation cooper wiring devices dedicated providing up date wiring device solutions solutions increase energy efficiency increase productivity promote safety work home produce reliable performance cooper wiring devices trusted name electrical products 175 years features arc fault technology saves lives industry standard construction arc fault technology proven arc fault breakers iterations reduce unwanted tripping built latest electrical safety technology",
      Seq(("huy", "na"), ("mfg brand name", "tamper")),
      Seq((143102, "20 amp tamper resitance duplex receptacle"))
    )

    val p2 = Product(111584, "steel city 1 gang square electrical box cover case 25", "steel city 1 gang square electrical box cover mounts flat offers handy box cutout cover meets ul csa standards safety resists corrosion zinc plated construction covers square electrical box steel construction zinc plated corrosion resistance ul listed",
      Seq(), Seq((46636, "steel electrical box")))

    val s = PromiscuousSpelling(p1)

    s.matches().foreach(println)


    println("===")


    val s2 = PromiscuousSpelling(p2)

    s2.matches().foreach(println)

  }




}
