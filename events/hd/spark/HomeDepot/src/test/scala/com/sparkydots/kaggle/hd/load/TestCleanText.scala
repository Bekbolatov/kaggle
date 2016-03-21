package com.sparkydots.kaggle.hd.load

import org.scalatest.{FlatSpec, Matchers}


class TestCleanText  extends FlatSpec with Matchers {

  "CleanText" should "work" in {
    val t = "The Cooper Wiring Devices 20-Amp AFCI Duplex Receptacle is the perfect outlet for any commercial or residential application. Arc fault circuit interrupters have been established as a stable technology which can be used in conjunction with various loads through the home in both AFCI breakers and AFCI receptacles. This device protects against arc faults resulting from damaged insulation and other causes that can lead to electrical fires. For increased safety, this receptacle is Tamper Resistant and meets 2014 NEC Article 406.12 which states that all 15-Amp and 20-Amp, 125-Volt/AC receptacles installed in dwelling units must be tamper resistant. Meets and exceeds 10ka Short Circuit Testing and Underwriters Laboratories UL 1699A and UL 498 Safety standards. Ground termination with back wire clamp provides secure wiring and reduces installation time. Line side terminals are backed out and staked for fast installation. When downstream receptacles are wired from load side, a 20-Amp feed-through rating offers full protection. Terminal and mounting screws have tri-combo heads for versatile installation. Cooper Wiring Devices is dedicated to providing the most up-to-date wiring device solutions; solutions that increase energy efficiency, increase productivity, promote safety at work and in the home and produce reliable performance. Cooper Wiring Devices has been a trusted name in electrical products for over 175-years.Features arc fault technology that saves lives and is an industry standard in new constructionArc fault technology has been proven through arc fault breakersHas gone through several iterations to reduce unwanted trippingBuilt with all of the latest electrical safety technology"
    val tt = CleanText.clean(t)

    println(t)
    println(tt)
  }
}
