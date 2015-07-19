package com.sparkydots.kaggle.avito.functions

import com.sparkydots.kaggle.avito.functions.Functions._
import org.scalatest.{FlatSpec, Matchers}

class TestFunctions extends FlatSpec with Matchers {

  "Functions" should "work" in {
    _toInt("1") shouldBe 1

    _toIntOrMinus("1") shouldBe 1
    _toIntOrMinus("sd") shouldBe -1

    _length("Katelo") shouldBe 6
    _length("Продам ходули складные") shouldBe 22

    _parseParams("{}") shouldBe Seq()
    _parseParams("") shouldBe Seq()
    _parseParams("{817:'Кузов', 5:'Запчасти', 598:'Для автомобилей'}") shouldBe Seq(5,598,817)
    _parseParams("{45:'Кровати, диваны и кресла'}") shouldBe  Seq(45)
    _parseParams("{593:{to:400000, from:350000}, 187:['Фургон', Минивэн', Микроавтобус'], 210:'Citroen'}") shouldBe  Seq(187,210,593)
    _parseParams("{797:'15', 709:'Диски', 799:'5', 5:'Шины, диски и колёса', 801:{to:'45', from:'30'}, 800:'100'}") shouldBe  Seq(5,709,797,799,800,801)
    _parseParams("{797:'16', 796:'Литые', 709:'Диски', 799:'5', 798:'6.5', 5:'Шины, диски и колёса', 801:{to:'47', from:'47'}, 800:'114.3'}") shouldBe  Seq(5,709,796,797,798,799,800,801)
    _parseParams("{797:'15', 709:'Диски', 799:'5', 5:'Шины, диски и колёса', 801:{from:'-40'}, 800:'139.7'}") shouldBe  Seq(5,709,797,799,800,801)


    _parseParams("{45:'Кровати, диваны и кресла'}") shouldBe Seq(45)
    _parseParams("{817:'Кузов', 5:'Запчасти', 598:'Для автомобилей'}") shouldBe Seq(5,598,817)
    _parseParams("") shouldBe Seq()
    _parseParams("{12:}") shouldBe Seq(12)
    _parseParams("{12:{}}") shouldBe Seq(12)
    _parseParams("{}") shouldBe Seq()


    _parseTime("2015-04-20 00:00:00.0") shouldBe 0
    _parseTime("2015-04-20 00:00:01.0") shouldBe 1
    _parseTime("2015-04-20 00:01:01.0") shouldBe 61
    _parseTime("2015-04-20 01:01:01.0") shouldBe 3661

    _parseTime("2015-04-25 00:00:00.0") shouldBe 432000

    _parseTime("2015-05-12 00:00:00.0") shouldBe 1900800

    _parseTime("2015-05-20 00:00:00.0") shouldBe 2592000
    _parseTime("2015-05-20 00:00:01.0") shouldBe 2592001

    _dayOfWeek(_parseTime("2015-04-20 01:00:00.0")) shouldBe 0
    _dayOfWeek(_parseTime("2015-04-25 01:00:00.0")) shouldBe 5
    _dayOfWeek(_parseTime("2015-04-26 13:00:00.0")) shouldBe 6

    _hourOfDay(_parseTime("2015-04-20 00:30:00.0")) shouldBe 0
    _hourOfDay(_parseTime("2015-04-21 14:30:00.0")) shouldBe 14
    _hourOfDay(_parseTime("2015-05-10 08:06:00.0")) shouldBe 8
  }
}
