package com.sparkydots.kaggle.hd.load

object CleanText extends Serializable {

  // 4x2
  val XBY = "xby"
  val pattern_xby_d = raw"(x[0-9])".r
  val pattern_d_xby = raw"([0-9])x".r

  // units
  val pattern_inch = raw"([0-9])( *)(inches|inch|in|')\.?".r
  val pattern_foot = raw"""([0-9])( *)(foot|feet|ft|''|")\.?""".r
  val pattern_pound = raw"([0-9])( *)(pounds|pound|lbs|lb)\.?".r
  val pattern_sqft = raw"([0-9])( *)(square|sq) ?\.?(feet|foot|ft)\.?".r
  val pattern_gallons = raw"([0-9])( *)(gallons|gallon|gal)\.?".r
  val pattern_oz = raw"([0-9])( *)(ounces|ounce|oz)\.?".r
  val pattern_cm = raw"([0-9])( *)(centimeters|cm)\.?".r
  val pattern_mm = raw"([0-9])( *)(milimeters|mm)\.?".r
  val pattern_deg = raw"([0-9])( *)(degrees|degree)\.?".r
  val pattern_volt = raw"([0-9])( *)(volts|volt)\.?".r
  val pattern_watt = raw"([0-9])( *)(watts|watt)\.?".r
  val pattern_amp = raw"([0-9])( *)(amperes|ampere|amps|amp)\.?".r
  val pattern_kiloamp = raw"([0-9])( *)(kiloamperes|kiloampere|kamps|kamp|ka)\.?".r

  // split
  val pattern_split = raw"[^0-9a-z]"

  // remove known helper words
  val set_known_words = Set("the", "a", "an",
    "this", "that", "which", "whose",
    "other", "and", "or",
    "be", "is", "are", "been",
    "have", "has", "had",
    "can", "could", "will", "would",
    "go", "gone",
    "all", "some", "any", "most", "several", "none", "nothing",
    "as", "of", "in", "on", "at", "over", "from", "to",
    "with", "through", "for", "when", "then",
    "new", "old"
  )

  def clean(s: String) = {
    var ss = s.toLowerCase.trim

    // 4ft x 2ft
    ss = ss.replaceAll(" x "," " + XBY + " ")
    ss = ss.replaceAll(raw"\*"," " + XBY + " ")
    ss = ss.replaceAll(" by "," " + XBY)
    ss = pattern_xby_d.replaceAllIn(ss, " " + XBY + " \1")
    ss = pattern_d_xby.replaceAllIn(ss, "\1 " + XBY + " ")

    // units
    ss = pattern_inch.replaceAllIn(ss, raw"\1 inch ")
    ss = pattern_foot.replaceAllIn(ss, raw"\1 foot ")
    ss = pattern_pound.replaceAllIn(ss, raw"\1 pound ")
    ss = pattern_sqft.replaceAllIn(ss, raw"\1 sqft ")
    ss = pattern_gallons.replaceAllIn(ss, raw"\1 gal ")
    ss = pattern_oz.replaceAllIn(ss, raw"\1 oz ")
    ss = pattern_cm.replaceAllIn(ss, raw"\1 cm ")
    ss = pattern_mm.replaceAllIn(ss, raw"\1 mm ")
    ss = pattern_deg.replaceAllIn(ss, raw"\1 deg ")
    ss = pattern_volt.replaceAllIn(ss, raw"\1 volt ")
    ss = pattern_watt.replaceAllIn(ss, raw"\1 watt ")
    ss = pattern_amp.replaceAllIn(ss, raw"\1 amp ")
    ss = pattern_kiloamp.replaceAllIn(ss, raw"\1 ka ")


    // some by hand
    ss = ss.replaceAll("whirpool","whirlpool")
    ss = ss.replaceAll("whirlpoolga", "whirlpool")
    ss = ss.replaceAll("whirlpoolstainless","whirlpool stainless")

    // split into words and remove empty and known helper words
    ss = ss.split(pattern_split).filter(w => w.nonEmpty && !set_known_words.contains(w)).mkString(" ")

    ss
  }



}
