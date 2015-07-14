package com.sparkydots.kaggle.avito

import org.scalatest.{Matchers, FlatSpec}
import Functions._

class TestFunctions extends FlatSpec with Matchers {

  "Functions" should "work" in {
    _toInt("1") shouldBe 1

    _toIntOrMinus("1") shouldBe 1
    _toIntOrMinus("sd") shouldBe -1

    _length("Katelo") shouldBe 6
    _length("Продам ходули складные") shouldBe 22

    _parseParams("{}") shouldBe ""
    _parseParams("") shouldBe ""
    _parseParams("{817:'Кузов', 5:'Запчасти', 598:'Для автомобилей'}") shouldBe "817,5,598"
    _parseParams("{45:'Кровати, диваны и кресла'}") shouldBe "45"

  }
}
