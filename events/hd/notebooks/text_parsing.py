import re

pattern_camel = re.compile(r"([a-z]+)([0-9A]|([A-Z][^ ]+))")
pattern_lcase_number = re.compile(r"([a-z])([0-9])")
pattern_digit_lcase = re.compile(r"([0-9])([a-z])")
pattern_s = re.compile(r"([a-z])'s")
pattern_number_commas = re.compile(r"([0-9]),([0-9])")

pattern_number_point = re.compile(r"([0-9]+)\.([0-9]+)")
pattern_number_ratio = re.compile(r"([0-9]+)/([0-9]+)")


# 4x2
XBY = "xby"
pattern_xby_d = re.compile(r"(x[0-9])")
pattern_d_xby = re.compile(r"([0-9])x")

# units
pattern_inch = re.compile(r"([0-9])( *-?)(inches|inch|in|')\.?")
pattern_foot = re.compile(r"([0-9])( *-?)(foot|feet|ft|''|\")\.?")
pattern_pound = re.compile(r"([0-9])( *-?)(pounds|pound|lbs|lb)\.?")
pattern_sqft = re.compile(r"([0-9])( *-?)(square|sq) ?\.? ?(feet|foot|ft)\.?")
pattern_cuft = re.compile(r"([0-9])( *-?)(cubic|cu) ?\.? ?(feet|foot|ft)\.?")
pattern_gallons = re.compile(r"([0-9])( *-?)(gallons|gallon|gal)\.?")
pattern_oz = re.compile(r"([0-9])( *-?)(ounces|ounce|oz)\.?")
pattern_cm = re.compile(r"([0-9])( *-?)(centimeters|cm)\.?")
pattern_mm = re.compile(r"([0-9])( *-?)(milimeters|mm)\.?")
pattern_deg = re.compile(r"([0-9])( *-?)(degrees|degree)\.?")
pattern_volt = re.compile(r"([0-9])( *-?)(volts|volt)\.?")
pattern_watt = re.compile(r"([0-9])( *-?)(watts|watt)\.?")
pattern_amp = re.compile(r"([0-9])( *-?)(amperes|ampere|amps|amp)\.?")
pattern_kamp = re.compile(r"([0-9])( *-?)(kiloamperes|kiloampere|kamps|kamp|ka)\.?")

# split
pattern_split = re.compile('[^0-9a-z-]')

known_words = set(["the", "a", "an",
    "this", "that", "which", "whose",
    "other", "and", "or",
    "be", "is", "isn't", "are", "aren't", "been",
    "have", "haven't",  "has", "hasn't", "had", "hadn't",
    "can", "can't", "could", "couldn't", "will", "won't", "would", "wouldn't",
    "do", "don't", "does", "doesn't", "did", "didn't", "done",
    "go", "gone", "see", "seen",
    "not",
    "all", "some", "any", "most", "several", "no", "none", "nothing",
    "as", "of", "in", "on", "at", "over", "from", "to",
    "with", "through", "for", "when", "then",
    "new", "old",
    "you", "your", "yours", "me", "i", "my", "mine", "it", "its"])

def str_stem(s):
    if isinstance(s, str) or isinstance(s, unicode):

        s = pattern_camel.sub(r"\1 \2", s)
        s = pattern_lcase_number.sub(r"\1 \2", s)
        s = pattern_digit_lcase.sub(r"\1 \2", s)
        s = pattern_number_commas.sub(r"\1\2", s)
        s = pattern_s.sub(r"\1", s)

        # 3.1 2.14  3p1  3p14
        s = pattern_number_point.sub(r" \1p\2 ", s)
        # 3/4      3d4
        s = pattern_number_ratio.sub(r" \1d\2 ", s)


        s = s.lower().strip()

        # 4ft x 2ft
        s = s.replace(" x "," " + XBY + " ")
        s = s.replace("*"," " + XBY + " ")
        s = s.replace(" by "," " + XBY)
        s = pattern_xby_d.sub(" " + XBY + " \1", s)
        s = pattern_d_xby.sub("\1 " + XBY + " ", s)

        # units
        s = pattern_inch.sub(r"\1 unitsinches ", s)
        s = pattern_foot.sub(r"\1 unitsfeet ", s)
        s = pattern_pound.sub(r"\1 unitspounds ", s)
        s = pattern_sqft.sub(r"\1 unitssquareft ", s)
        s = pattern_cuft.sub(r"\1 unitscubicft ", s)
        s = pattern_gallons.sub(r"\1 unitsgallons ", s)
        s = pattern_oz.sub(r"\1 unitsounces ", s)
        s = pattern_cm.sub(r"\1 unitscentims ", s)
        s = pattern_mm.sub(r"\1 unitsmillims ", s)
        s = pattern_deg.sub(r"\1 unitsdegrees ", s)
        s = pattern_volt.sub(r"\1 unitsvolts ", s)
        s = pattern_watt.sub(r"\1 unitswatts ", s)
        s = pattern_amp.sub(r"\1 unitsamperes ", s)
        s = pattern_kamp.sub(r"\1 unitskamperes ", s)

        # some by hand
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        s = s.replace("pressure-treated","pressure-treated pt")

        s = ' '.join([x for x in pattern_split.split(s) if x and x not in known_words])
        return s
    else:
        #raise ValueError("Type of " + str(s) + " is " + str(type(s)))
        #print "HUY"
        return 'null'