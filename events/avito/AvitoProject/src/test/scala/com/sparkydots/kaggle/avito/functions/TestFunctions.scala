package com.sparkydots.kaggle.avito.functions

import com.sparkydots.kaggle.avito.functions.Functions._
import org.scalatest.{FlatSpec, Matchers}

class TestFunctions extends FlatSpec with Matchers {

  "Functions" should "work" in {
    toInt("1") shouldBe 1

    toIntOrMinus("1") shouldBe 1
    toIntOrMinus("sd") shouldBe -1

    parseParams("{}") shouldBe Seq()
    parseParams("") shouldBe Seq()
    parseParams("{817:'Кузов', 5:'Запчасти', 598:'Для автомобилей'}") shouldBe Seq(5,598,817)
    parseParams("{45:'Кровати, диваны и кресла'}") shouldBe  Seq(45)
    parseParams("{593:{to:400000, from:350000}, 187:['Фургон', Минивэн', Микроавтобус'], 210:'Citroen'}") shouldBe  Seq(187,210,593)
    parseParams("{797:'15', 709:'Диски', 799:'5', 5:'Шины, диски и колёса', 801:{to:'45', from:'30'}, 800:'100'}") shouldBe  Seq(5,709,797,799,800,801)
    parseParams("{797:'16', 796:'Литые', 709:'Диски', 799:'5', 798:'6.5', 5:'Шины, диски и колёса', 801:{to:'47', from:'47'}, 800:'114.3'}") shouldBe  Seq(5,709,796,797,798,799,800,801)
    parseParams("{797:'15', 709:'Диски', 799:'5', 5:'Шины, диски и колёса', 801:{from:'-40'}, 800:'139.7'}") shouldBe  Seq(5,709,797,799,800,801)


    parseParams("{45:'Кровати, диваны и кресла'}") shouldBe Seq(45)
    parseParams("{817:'Кузов', 5:'Запчасти', 598:'Для автомобилей'}") shouldBe Seq(5,598,817)
    parseParams("") shouldBe Seq()
    parseParams("{12:}") shouldBe Seq(12)
    parseParams("{12:{}}") shouldBe Seq(12)
    parseParams("{}") shouldBe Seq()


    parseTime("2015-04-20 00:00:00.0") shouldBe 0
    parseTime("2015-04-20 00:00:01.0") shouldBe 1
    parseTime("2015-04-20 00:01:01.0") shouldBe 61
    parseTime("2015-04-20 01:01:01.0") shouldBe 3661

    parseTime("2015-04-25 00:00:00.0") shouldBe 432000

    parseTime("2015-05-12 00:00:00.0") shouldBe 1900800

    parseTime("2015-05-20 00:00:00.0") shouldBe 2592000
    parseTime("2015-05-20 00:00:01.0") shouldBe 2592001

    dayOfWeek(parseTime("2015-04-20 01:00:00.0")) shouldBe 0
    dayOfWeek(parseTime("2015-04-25 01:00:00.0")) shouldBe 5
    dayOfWeek(parseTime("2015-04-26 13:00:00.0")) shouldBe 6

    hourOfDay(parseTime("2015-04-20 00:30:00.0")) shouldBe 0
    hourOfDay(parseTime("2015-04-21 14:30:00.0")) shouldBe 14
    hourOfDay(parseTime("2015-05-10 08:06:00.0")) shouldBe 8
  }
}
