import pytest
from src.text.normalizer import TextNormalizer


@pytest.fixture
def cs():
    return TextNormalizer(language="cs")


@pytest.fixture
def sk():
    return TextNormalizer(language="sk")


def test_no_digits_or_symbols_survive(cs):
    out = cs.normalize("Dne 1.1.2024 v 14:30 zaplatil 100 Kč, tj. cca 4 € (20 %).")
    assert not any(c.isdigit() for c in out) and "%" not in out and "€" not in out


def test_numbers_cs(cs):
    assert cs.normalize("Mám 5 jablek.") == "Mám pět jablek."
    assert cs.normalize("Je mi 25 let.") == "Je mi dvacet pět let."          # Czech: separate words


def test_numbers_sk(sk):
    assert sk.normalize("Mám 25 rokov.") == "Mám dvadsaťpäť rokov."          # Slovak: one word


def test_case_from_context_cs(cs):
    assert cs.normalize("Šel s 5 přáteli.") == "Šel s pěti přáteli."


@pytest.mark.parametrize("text,expected", [
    ("1. ledna 2024", "Prvního ledna dva tisíce dvacet čtyři"),
    ("15.3.2024", "Patnáctého března dva tisíce dvacet čtyři"),
])
def test_dates_cs(cs, text, expected):
    assert cs.normalize(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("Karel IV.", "Karel čtvrtý."), ("XXI. století", "Dvacáté první století"), ("III. díl", "Třetí díl"),
])
def test_roman_numerals_cs(cs, text, expected):
    assert cs.normalize(text) == expected


def test_abbreviations(cs, sk):
    assert cs.normalize("např. toto") == "například toto"
    assert sk.normalize("atď.") == "a tak ďalej."


def test_dashes_and_ellipsis(cs):
    assert cs.normalize("A pak – nic...") == "A pak — nic…"


def test_punctuation_and_empty(cs):
    assert cs.normalize("Máš 5 jablek?").endswith("?")
    assert cs.normalize("") == ""
    assert cs.normalize("Toto je věta bez čísel.") == "Toto je věta bez čísel."


# ---- added with Prompt 2.2 ------------------------------------------------------------------
import logging
import time

from src.text.phonemizer import CzechSlovakPhonemizer

EXAMPLES = {  # id: (language, book_config, input, expected output)
    "english-phrases": ("cs", {"english": ["Harry", "Harry Potter"]},
                        "Harry Potter a Harry, ale ne Harrymu ani harry.",
                        "<en>Harry Potter</en> a <en>Harry</en>, ale ne Harrymu ani harry."),
    "hand-tagged-span": ("cs", None, "Četl <en>The Hobbit</en> 2 roky.", "Četl <en>The Hobbit</en> dva roky."),
    "sk-personal-masculine": ("sk", None, "Prišli 2 muži.", "Prišli dvaja muži."),
    "adverb-before-adjective": ("cs", None, "Koupil 2 velmi staré knihy.", "Koupil dvě velmi staré knihy."),
    "coordinated-adjectives": ("cs", None, "Koupil 2 červené a modré knihy.", "Koupil dvě červené a modré knihy."),
    "count-before-comma-clause": ("cs", None, "Zůstali 2, ženy odešly.", "Zůstali dva, ženy odešly."),
    "count-before-coordinated-clause": ("cs", None, "Zůstali 2, ale staré ženy odešly.",
                                        "Zůstali dva, ale staré ženy odešly."),
    "coordinated-counts": ("cs", None, "Koupil 2 nebo 3 knihy a 2, 3 nebo 4 židle.",
                           "Koupil dvě nebo tři knihy a dvě, tři nebo čtyři židle."),
    "sk-coordinated-counts": ("sk", None, "Prišli 2 alebo 3 muži.", "Prišli dvaja alebo traja muži."),
    "noun-after-many-modifiers": ("cs", None, "Koupil 2 mimořádně dobře zachovalé vzácné historické knihy.",
                                  "Koupil dvě mimořádně dobře zachovalé vzácné historické knihy."),
    "sk-adverb-before-adjective": ("sk", None, "Prišli 2 veľmi starí muži a 2 naozaj staré ženy.",
                                   "Prišli dvaja veľmi starí muži a dve naozaj staré ženy."),
    "unit": ("cs", None, "Ujel 5 km.", "Ujel pět kilometrů."),
    "time-cs": ("cs", None, "Vlak jede v 14:30.", "Vlak jede ve čtrnáct třicet."),
    "time-sk": ("sk", None, "Stretneme sa o 14:30.", "Stretneme sa o štrnástej tridsať."),
    "currency-cs": ("cs", None, "Zaplatil 4,50 €.", "Zaplatil čtyři eura padesát centů."),
    "currency-sk": ("sk", None, "Zaplatil 2 €.", "Zaplatil dve eurá."),
    "negative-amount": ("cs", None, "Dlužil −4,50 € a -0,50 €.", "Dlužil mínus čtyři eura padesát centů a mínus padesát centů."),
    "negative-prefixed-currency": ("cs", None, "Dlužil -$4.50 a -€4.",
                                   "Dlužil mínus čtyři dolary padesát centů a mínus čtyři eura."),
    "sk-animals-count-like-things": ("sk", None, "Boli tam 2 vlci a videl som 2 psov.",
                                     "Boli tam dva vlci a videl som dvoch psov."),
    "negative-range": ("cs", None, "Teplota byla −5–−1 °C.", "Teplota byla mínus pět až mínus jeden stupeň Celsia."),
    "leading-zero-decimal": ("cs", None, "Vážilo to 0.500 kg.", "Vážilo to nula celá pět desetin kilogramu."),
    "time-with-hod": ("cs", None, "Sraz je ve 14.30 hod. Pak odjedeme.", "Sraz je ve čtrnáct třicet. Pak odjedeme."),
    "dot-time-range": ("cs", None, "Otevřeno 15.30–16.00 hod.", "Otevřeno patnáct třicet až šestnáct hodin."),
    "negative-decimal-range": ("cs", None, "Teplota −1,5–−0,5 °C.",
                               "Teplota mínus jedna celá pět desetin až mínus nula celá pět desetin stupně Celsia."),
    "integer-end-of-a-decimal-range-takes-its-case": ("cs", None, "Při 30–35,5 °C. Po 1,5–2 hodinách odešel. Při 30,5–35 °C.",
                                                      "Při třiceti až třicet pět celých pět desetin stupně Celsia. Po jedna "
                                                      "celá pět desetin až dvou hodinách odešel. Při třicet celých pět "
                                                      "desetin až třiceti pěti stupních Celsia."),
    "sign-after-opening-quote": ("cs", None, "Řekl: „-5 °C“.", "Řekl: „Mínus pět stupňů Celsia“."),
    "sk-diminutive-animals": ("sk", None, "Prišli 2 ježkovia.", "Prišli dva ježkovia."),
    "spaced-per-unit": ("cs", None, "Stojí to 100 Kč / kg.", "Stojí to sto korun za kilogram."),
    "spaced-per-second": ("cs", None, "Jel 5 m / s.", "Jel pět metrů za sekundu."),
    "per-minute": ("sk", None, "Čerpadlo dá 20 l/min.", "Čerpadlo dá dvadsať litrov za minútu."),
    "ordinal-before-capitalised-noun": ("cs", None, "Hrála 5. Symfonie.", "Hrála pátá Symfonie."),
    "number-ends-sentence-before-noun": ("cs", None, "Měl jen 2. Děti odešly.", "Měl jen dva. Děti odešly."),
    "rank-at-sentence-end": ("cs", None, "Skončil 2. Skončila 2. Pak odešla. Koupil 5.",
                             "Skončil druhý. Skončila druhá. Pak odešla. Koupil pět."),
    "sk-rank-at-sentence-end": ("sk", None, "Dobehla 3. Umiestnil sa 2.", "Dobehla tretia. Umiestnil sa druhý."),
    "subject-after-adverbs": ("cs", None, "Měl jen 2. Malé děti potom odešly.", "Měl jen dva. Malé děti potom odešly."),
    "subject-after-many-modifiers": ("cs", None, "Měl jen 2. Malé děti se tam potom už nikdy nevrátily.",
                                     "Měl jen dva. Malé děti se tam potom už nikdy nevrátily."),
    "sk-number-ends-sentence-before-noun": ("sk", None, "Mal len 2. Deti odišli.", "Mal len dva. Deti odišli."),
    "ordinal-inside-noun-phrase": ("cs", None, "Beethovenova 5. Symfonie zazněla.", "Beethovenova pátá Symfonie zazněla."),
    "roman-ends-sentence-before-noun": ("cs", None, "Vládl Karel IV. Velký požár vypukl. Karel IV. Lucemburský zemřel.",
                                        "Vládl Karel čtvrtý. Velký požár vypukl. Karel čtvrtý Lucemburský zemřel."),
    "sk-zero-hour": ("sk", None, "Stretneme sa o 0:30.", "Stretneme sa o nultej tridsať."),
    "zero-hour-after-preposition": ("cs", None, "Otevřeno od 0:00 do 6:00, sejdeme se v 0:00.",
                                    "Otevřeno od nuly hodin do šesti hodin, sejdeme se v nula hodin."),
    "dot-time-with-hour-word": ("cs", None, "Sejdeme se ve 14.30 hodin, v 8.00 hodiny.",
                                "Sejdeme se ve čtrnáct třicet, v osm hodin."),
    "sk-dot-time-with-hour-word": ("sk", None, "Stretneme sa o 14.30 hodín.", "Stretneme sa o štrnástej tridsať."),
    "negative-zero-scaled": ("cs", None, "Dluh -0 tis. Kč a $-0 mil.", "Dluh mínus nula korun a mínus nula dolarů."),
    "preposition-before-time-range": ("cs", None, "Otevřeno od 2:00–3:00.", "Otevřeno od dvou hodin až tří hodin."),
    "preposition-before-hour-word-range": ("cs", None, "Od 2:00 hod.–3:00 hod.", "Od dvou hodin až tří hodin."),
    "time-range-with-az-after-od": ("cs", None, "Od 8.30 až 9.30 hod. Od 1:00 až 2:00.",
                                    "Od osmi třiceti až devíti třiceti. Od jedné hodiny až dvou hodin."),
    "sk-preposition-before-time-range": ("sk", None, "Otvorené od 2:00–3:00.", "Otvorené od druhej až tretej."),
    "sk-time-range-with-az-after-od": ("sk", None, "Otvorené od 2:00 až 3:00.", "Otvorené od druhej až tretej."),
    "negative-zero": ("cs", None, "Bylo −0 °C, dluh $-0 a hodnota -0,0.",
                      "Bylo mínus nula stupňů Celsia, dluh mínus nula dolarů a hodnota mínus nula."),
    "sk-duration-time": ("sk", None, "Trať zabehol za 2:15, štart bol o 2:15.",
                         "Trať zabehol za dve pätnásť, štart bol o druhej pätnásť."),
    "capital-litre-symbols": ("cs", None, "Nalil 5 L vody a 5 mL oleje, litr stojí 30 Kč/L.",
                              "Nalil pět litrů vody a pět mililitrů oleje, litr stojí třicet korun za litr."),
    "rate-abbreviation-period": ("cs", None, "Stojí 100 Kč/hod. práce a 100 Kč/ks. včetně daně.",
                                 "Stojí sto korun za hodinu práce a sto korun za kus včetně daně."),
    "per-centimetre": ("cs", None, "Stojí 5 Kč/cm² a 10 g/mm.",
                       "Stojí pět korun za centimetr čtvereční a deset gramů za milimetr."),
    "duration-before-clause-dash": ("cs", None, "Let trval 3 h – 5 lidí čekalo.", "Let trval tři hodiny — pět lidí čekalo."),
    "per-powered-unit": ("cs", None, "Stojí 100 Kč/m² a 5 kg/m³.",
                         "Stojí sto korun za metr čtvereční a pět kilogramů za metr krychlový."),
    "ordinal-list": ("cs", None, "Přečti 2., 3. a 4. kapitolu.", "Přečti druhou, třetí a čtvrtou kapitolu."),
    "grouped-ordinal": ("cs", None, "Byl to 1 000. návštěvník a 1.000. host.", "Byl to tisící návštěvník a tisící host."),
    "scaled-rate-range": ("cs", None, "Stojí 5 tis. Kč/kg–10 tis. Kč/kg.",
                          "Stojí pět tisíc korun za kilogram až deset tisíc korun za kilogram."),
    "sk-ordinal-list": ("sk", None, "Prečítaj 2., 3. a 4. kapitolu.", "Prečítaj druhú, tretiu a štvrtú kapitolu."),
    "negative-zero-upper-end": ("cs", None, "Teplota 5–−0 a −5–−0 °C.",
                                "Teplota pět až mínus nula a mínus pět až mínus nula stupňů Celsia."),
    "scaled-price-per-unit": ("cs", None, "Stojí 5 tis. Kč/kg a 5–10 tis. Kč / kg.",
                              "Stojí pět tisíc korun za kilogram a pět až deset tisíc korun za kilogram."),
    "pounds-and-pence": ("cs", None, "Stálo to £4.50.", "Stálo to čtyři libry padesát pencí."),
    "multiplication": ("cs", None, "Plocha 3×4 m a 3x4 m.", "Plocha tři krát čtyři metry a tři krát čtyři metry."),
    "number-in-adjective": ("cs", None, "Přišel 25letý muž na 3denní výlet.",
                            "Přišel dvacetipětiletý muž na třídenní výlet."),
    "sk-number-in-adjective": ("sk", None, "Prišiel 5-ročný chlapec.", "Prišiel päťročný chlapec."),
    "sign-after-operator": ("cs", None, "Výsledek=-5.", "Výsledek rovná se mínus pět."),
    "price-per-unit-prefixed": ("cs", None, "Stojí to $4/kg.", "Stojí to čtyři dolary za kilogram."),
    "unit-after-scale": ("cs", None, "Ujel 5 tis. km.", "Ujel pět tisíc kilometrů."),
    "hyphenated-ordinal-range": ("cs", None, "Od 1.-5. ledna.", "Od prvního až pátého ledna."),
    "en-dash-minus": ("cs", None, "Teplota –5 °C.", "Teplota mínus pět stupňů Celsia."),
    "en-dash-sign-after-symbol": ("cs", None, "Dluh $–5 a €–5.", "Dluh mínus pět dolarů a mínus pět eur."),
    "plus-after-symbol": ("cs", None, "Změna $+5.", "Změna plus pět dolarů."),
    "plus-at-upper-end": ("cs", None, "Změna $5–$+10.", "Změna pět až plus deset dolarů."),
    "en-dash-sign-at-upper-end": ("cs", None, "Teplota −5––1 °C.", "Teplota mínus pět až mínus jeden stupeň Celsia."),
    "plus-at-range-top": ("cs", None, "Teplota −5–+5 °C.", "Teplota mínus pět až plus pět stupňů Celsia."),
    "plus-grouped-upper-end": ("cs", None, "Stálo to 1,000.00–+1,234.56 USD.",
                               "Stálo to tisíc dolarů až plus tisíc dvě stě třicet čtyři dolarů padesát šest centů."),
    "dotted-numbers": ("cs", None, "Verze 1.2.3 vyšla, adresa 192.168.1.1.",
                       "Verze jedna tečka dva tečka tři vyšla, adresa sto devadesát dva tečka sto šedesát osm tečka "
                       "jedna tečka jedna."),
    "sk-dotted-numbers": ("sk", None, "Verzia 1.2.3 vyšla.", "Verzia jeden bodka dva bodka tri vyšla."),
    "mixed-dot-groups": ("cs", None, "Číslo 1.234.56.", "Číslo jedna tečka dvě stě třicet čtyři tečka padesát šest."),
    "count-before-exclamation": ("cs", None, "Zůstali 2! Ženy odešly.", "Zůstali dva! Ženy odešly."),
    "iso-date": ("cs", None, "Dne 2024-01-15 odjel.", "Dne patnáctého ledna dva tisíce dvacet čtyři odjel."),
    "time-with-seconds": ("cs", None, "Doběhl za 2:15:30.", "Doběhl za dvě patnáct třicet."),
    "range-with-scale": ("cs", None, "Stálo to 5–10 tis. Kč.", "Stálo to pět až deset tisíc korun."),
    "range-per-unit": ("cs", None, "Stojí to 5–10 Kč/kg.", "Stojí to pět až deset korun za kilogram."),
    "prefixed-price-range-and-scale": ("cs", None, "Stálo to $5–10 mil.", "Stálo to pět až deset milionů dolarů."),
    "compact-time-range": ("cs", None, "Otevřeno 10:00-12:00.", "Otevřeno deset hodin až dvanáct hodin."),
    "dot-times-sharing-hour-word": ("cs", None, "Schůzky jsou v 8.30 a 9.30 hod.", "Schůzky jsou v osm třicet a devět třicet."),
    "hour-abbreviation-h": ("cs", None, "Otevřeno 14:30 h a 14.30 h.", "Otevřeno čtrnáct třicet a čtrnáct třicet."),
    "hour-unit-after-v-in-ranges-and-lists": ("cs", None, "Otevřeno v 9–17 h, přijdu v 1 h nebo v 5 nebo 6 h, po 2 h odejdu.",
                                              "Otevřeno v devět až sedmnáct hodin, přijdu v jednu hodinu nebo v pět nebo "
                                              "šest hodin, po dvou hodinách odejdu."),
    "time-unit-after-na-is-a-time-span": ("cs", None, "Na 2 h odešel.", "Na dvě hodiny odešel."),
    "time-unit-after-a-verb-with-na-and-the-locative": ("cs", None, "Záleží na 2 h.", "Záleží na dvou hodinách."),
    "mixed-time-range-with-hour-word": ("cs", None, "Otevřeno 8.30–9:30 hod.", "Otevřeno osm třicet až devět třicet."),
    "dot-time-before-az-do": ("cs", None, "Od 8.30 až do 9.30 hod.", "Od osmi třiceti až do devíti třiceti."),
    "sk-times-sharing-preposition": ("sk", None, "Stretnutia sú o 8.30 a 9.30 hod.",
                                     "Stretnutia sú o ôsmej tridsať a deviatej tridsať."),
    "seconds-hours-pieces": ("cs", None, "Trvalo to 5 s. Balení má 5 ks, let trval 3 h, mám 5 s sebou.",
                             "Trvalo to pět sekund. Balení má pět kusů, let trval tři hodiny, mám pět s sebou."),
    "sk-seconds-pieces": ("sk", None, "Trvalo to 5 s, balenie má 5 ks.", "Trvalo to päť sekúnd, balenie má päť kusov."),
    "dotted-units-range": ("cs", None, "Trvalo to 5 s.–10 s. a balení má 5 ks.–10 ks.",
                           "Trvalo to pět sekund až deset sekund a balení má pět kusů až deset kusů."),
    "time-range-with-hour-words": ("cs", None, "Otevřeno 10:00 hod.–12:00 hod. a 10.00 hod.–12.00 hod.",
                                   "Otevřeno deset hodin až dvanáct hodin a deset hodin až dvanáct hodin."),
    "repeated-spaces": ("cs", None, "Ujel 5  km.", "Ujel pět kilometrů."),
    "spaced-word-slash": ("cs", None, "Přijde on / ona.", "Přijde on nebo ona."),
    "one-letter-word-slash": ("cs", None, "Káva s/bez mléka.", "Káva s nebo bez mléka."),
    "one-letter-word-after-slash": ("cs", None, "Jel do/z Prahy, bez/s doprovodem.",
                                    "Jel do nebo z Prahy, bez nebo s doprovodem."),
    "one-letter-words-slash": ("cs", None, "Volba a/i záleží. Pohyb v/z budovy.",
                               "Volba a nebo i záleží. Pohyb v nebo z budovy."),
    "inclusive-forms": ("cs", None, "Vážený/á zákazník/ce, každý/á student/ka by měl/a přijít sám/a, "
                                    "i když přišel/a pozdě.",
                        "Vážený nebo vážená zákazník nebo zákaznice, každý nebo každá student nebo studentka "
                        "by měl nebo měla přijít sám nebo sama, i když přišel nebo přišla pozdě."),
    "sk-inclusive-forms": ("sk", None, "Prišiel/a si včas, mohol/a by si ostať, zákazník/čka.",
                           "Prišiel alebo prišla si včas, mohol alebo mohla by si ostať, zákazník alebo zákazníčka."),
    "plural-inclusive-forms": ("cs", None, "Milí/é studenti/ky, přišli/y jste.",
                               "Milí nebo milé studenti nebo studentky, přišli nebo přišly jste."),
    "sk-plural-inclusive-forms": ("sk", None, "Milí/é študenti/ky a zákazníci/čky.",
                                  "Milí alebo milé študenti alebo študentky a zákazníci alebo zákazníčky."),
    "currency-range-with-cents": ("cs", None, "Stojí to 1,50–2,50 €.",
                                  "Stojí to jedno euro padesát centů až dvě eura padesát centů."),
    "price-per-unit": ("cs", None, "Stojí to 100 Kč/kg.", "Stojí to sto korun za kilogram."),
    "sign-after-currency-symbol": ("cs", None, "Dluh $-4.50.", "Dluh mínus čtyři dolary padesát centů."),
    "comma-grouped-dollars": ("cs", None, "Stálo to $1,234.56.",
                              "Stálo to tisíc dvě stě třicet čtyři dolarů padesát šest centů."),
    "comma-grouped-suffix-currency": ("cs", None, "Stálo to 1,234.56 USD.",
                                      "Stálo to tisíc dvě stě třicet čtyři dolarů padesát šest centů."),
    "comma-grouped-number-and-unit": ("cs", None, "Žilo tam 1,234,567 lidí, cesta měřila 1,234.5 km.",
                                      "Žilo tam milion dvě stě třicet čtyři tisíc pět set šedesát sedm lidí, "
                                      "cesta měřila tisíc dvě stě třicet čtyři celé pět desetin kilometru."),
    "comma-grouped-range": ("sk", None, "Stálo to 1,234.50–2,000.75 €.",
                            "Stálo to tisícdvestotridsaťštyri eur päťdesiat centov až dvetisíc eur "
                            "sedemdesiatpäť centov."),
    "comma-decimal-stays": ("cs", None, "Vážilo to 1,234 kg.",
                            "Vážilo to jedna celá dvě stě třicet čtyři tisícin kilogramu."),
    "grouped-range": ("sk", None, "Stálo to 1 002–1 004 €.", "Stálo to tisícdve až tisícštyri eurá."),
    "vocalise-after-no-break-space": ("cs", None, "Šel s\u00a02 přáteli.", "Šel se\u00a0dvěma přáteli."),
    "num2words-variant": ("cs", {"num2words": {"construction": "inverted"}}, "Je mi 25 let.",
                          "Je mi pětadvacet let."),
    "period-ends-sentence": ("sk", None, "Kúpil chlieb, mlieko atď. Potom odišiel.",
                             "Kúpil chlieb, mlieko a tak ďalej. Potom odišiel."),
    "period-inside-sentence": ("cs", None, "Navštívil např. Prahu a Brno.", "Navštívil například Prahu a Brno."),
    "capital-at-sentence-start": ("cs", None, "5 lidí přišlo. 3 lidé odešli.", "Pět lidí přišlo. Tři lidé odešli."),
    "locative-abbreviation": ("sk", None, "Na str. 45 sa píše o tom.", "Na strane štyridsaťpäť sa píše o tom."),
    "abbreviations-in-capitals": ("cs", None, "NAPŘ. PRAHA A TAK DÁLE ATD.", "Například PRAHA A TAK DÁLE a tak dále."),
    "acronym-stays": ("cs", None, "Hrál na turnajích ATP.", "Hrál na turnajích ATP."),
    "slash-between-words": ("cs", None, "Přijde on a/nebo ona, on/ona.", "Přijde on a nebo ona, on nebo ona."),
    "legal-reference": ("cs", None, "Podle § 5 odst. 2 platí.", "Podle paragrafu pět odstavce dva platí."),
    "coordinated-references": ("cs", None, "Podle § 5 a § 6 platí.", "Podle paragrafu pět a paragrafu šest platí."),
    "reference-after-sentence-end": ("cs", None, "Podle § 5. § 6 platí.", "Podle paragrafu pět. Paragraf šest platí."),
    "shared-preposition-into-range": ("cs", None, "Pracoval s 1 kg a 2–3 kg.",
                                      "Pracoval s jedním kilogramem a dvěma až třemi kilogramy."),
    "shared-preposition-stops-at-sentence-end": ("cs", None, "Pracoval s 2 kg. A 3 kg stačily.",
                                                 "Pracoval se dvěma kilogramy. A tři kilogramy stačily."),
    "coordinated-measures-share-preposition": ("cs", None, "Pracoval s 2 kg a 3 kg, bez 5 € a 3 €.",
                                               "Pracoval se dvěma kilogramy a třemi kilogramy, bez pěti eur a tří eur."),
    "compact-labels": ("cs", None, "Podle §5 odst.2 písm.a zákona, viz str.45.",
                       "Podle paragrafu pět odstavce dva písmena a zákona, viz strana čtyřicet pět."),
    "sk-compact-labels": ("sk", None, "Podľa §5 ods.2 zákona.", "Podľa paragrafu päť odseku dva zákona."),
    "glued-to-next-word": ("cs", None, "Vyšel 5.díl, např.Praha.", "Vyšel pátý díl, například Praha."),
    "glued-multiplication": ("cs", None, "Stálo to 3x4.", "Stálo to tři krát čtyři."),
    "glued-before-value": ("cs", None, "Sejdeme se v14:30, teplota5 °C, dne1.1.2024.",
                           "Sejdeme se v čtrnáct třicet, teplota pět stupňů Celsia, dne prvního ledna dva tisíce dvacet čtyři."),
    "prefixed-scale-range": ("cs", None, "Stálo to $5 mil.–$10 mil. a $500 tis.–$1 mil.",
                             "Stálo to pět až deset milionů dolarů a pět set tisíc dolarů až milion dolarů."),
    "spaced-times": ("cs", None, "Rozměr 3  x  4 m, cvičím 3 x týdně, opakuj to 3 x.",
                     "Rozměr tři krát čtyři metry, cvičím tři krát týdně, opakuj to tři krát."),
    "multiplication-with-many-spaces": ("cs", None, "Rozměr 3    x 4 m.", "Rozměr tři krát čtyři metry."),
    "sk-thousand-range": ("sk", None, "Stálo to 2–3 tis. Kč a 2–3 mil. €.",
                          "Stálo to dvetisíc až tritisíc korún a dva až tri milióny eur."),
    "scale-in-capitals": ("cs", None, "Stálo to 5 TIS.–10 TIS. Kč a $5 MIL.",
                          "Stálo to pět tisíc až deset tisíc korun a pět milionů dolarů."),
    "scale-on-both-ends": ("cs", None, "Stálo to 5 tis. Kč–10 tis. Kč a 5 tis.–10 tis. Kč.",
                           "Stálo to pět tisíc korun až deset tisíc korun a pět tisíc až deset tisíc korun."),
    "two-digit-year": ("cs", None, "Narodil se 1.1.24 a 5.6.05.",
                       "Narodil se prvního ledna dvacet čtyři a pátého června nula pět."),
    "sk-two-digit-year": ("sk", None, "Stalo sa to 1.1.24.", "Stalo sa to prvého januára dvadsaťštyri."),
    "symbol-on-both-ends": ("cs", None, "Stojí $5–$10 a €5 – €10.", "Stojí pět až deset dolarů a pět až deset eur."),
    "signed-symbol-on-both-ends": ("cs", None, "Změna -$5–-$10 a −€5–−€10.",
                                   "Změna mínus pět až mínus deset dolarů a mínus pět až mínus deset eur."),
    "unit-on-both-ends": ("cs", None, "Ujel 5 km–10 m za 1,50 €–2,50 €.",
                          "Ujel pět kilometrů až deset metrů za jedno euro padesát centů až dvě eura padesát centů."),
    "dash-between-clauses": ("cs", None, "Ujel 5 km – 10 lidí ho sledovalo.", "Ujel pět kilometrů — deset lidí ho sledovalo."),
    "capital-one-letter-abbreviations": ("cs", None, "Č. 5 platí. R. 2024 byl dobrý, ale Č. Novák ne.",
                                         "Číslo pět platí. Roku dva tisíce dvacet čtyři byl dobrý, ale Č. Novák ne."),
    "sk-capital-one-letter-abbreviation": ("sk", None, "Č. 5 platí.", "Číslo päť platí."),
    "acronyms-in-capitals": ("cs", None, "TURNAJ ATP. HRÁL ZA TJ. SOKOL.", "TURNAJ ATP. HRÁL ZA TJ. SOKOL."),
    "capital-abbreviations-in-mixed-text": ("cs", None, "Viděl NAPŘ. Prahu, Brno ATD.",
                                            "Viděl například Prahu, Brno a tak dále."),
    "sk-party-acronym": ("sk", None, "Voliči ODS. sa radovali.", "Voliči ODS. sa radovali."),
    "letter-labels-not-roman": ("cs", None, "Velikost L. Příloha C. Varianta D. Kapitola byla dlouhá.",
                                "Velikost L. Příloha C. Varianta D. Kapitola byla dlouhá."),
    "roman-with-d-and-m": ("cs", None, "D. kapitola, ale Washington DC. a nové CD.",
                           "Pětistá kapitola, ale Washington DC. a nové CD."),
    "roman-range-without-preposition": ("cs", None, "XIX.–XX. století bylo bouřlivé. Od XIX.–XX. století.",
                                        "Devatenácté až dvacáté století bylo bouřlivé. Od devatenáctého až dvacátého století."),
    "unary-minus-after-plus": ("cs", None, "Platí 2+-3 = -1.", "Platí dva plus mínus tři rovná se mínus jedna."),
    "signed-operand-after-x": ("cs", None, "Platí 3x−4 a 3x-4.", "Platí tři krát mínus čtyři a tři krát mínus čtyři."),
    "plus-operand": ("cs", None, "Platí 3/+4 a 3×+4.", "Platí tři lomeno plus čtyři a tři krát plus čtyři."),
    "binary-unicode-minus": ("cs", None, "Platí 5 − 3 = 2 a 5−1 = 4.",
                             "Platí pět mínus tři rovná se dva a pět mínus jedna rovná se čtyři."),
    "signed-operands": ("cs", None, "Platí 3×−4 = −12 a 3/−4.",
                        "Platí tři krát mínus čtyři rovná se mínus dvanáct a tři lomeno mínus čtyři."),
    "spaced-two-digit-year": ("cs", None, "Narodil se 5. 6. 05, ne 5. 6. 24 lidí.",
                              "Narodil se pátého června nula pět, ne pátého června dvacet čtyři lidí."),
    "per-unit-on-both-ends": ("cs", None, "Stojí 5 Kč/kg–10 Kč/kg nebo $4/kg–$5/kg.",
                              "Stojí pět korun za kilogram až deset korun za kilogram nebo čtyři dolary za kilogram "
                              "až pět dolarů za kilogram."),
    "math-signs": ("sk", None, "Platí 3 × 4 = 12.", "Platí tri krát štyri rovná sa dvanásť."),
    "square-metres": ("cs", None, "Byt má 80 m² a sklep 2 m2.",
                      "Byt má osmdesát metrů čtverečních a sklep dva metry čtvereční."),
    "dot-thousands": ("cs", None, "Stálo to 10.000 Kč.", "Stálo to deset tisíc korun."),
    "time-range": ("cs", None, "Otevřeno 10:00–12:00.", "Otevřeno deset hodin až dvanáct hodin."),
    "label-cs": ("cs", None, "Kapitola 1 začíná.", "Kapitola jedna začíná."),
    "same-form-in-every-case": ("cs", None, "Pak začalo XXI. století.", "Pak začalo dvacáté první století."),
    "tagger-gender-fix": ("sk", None, "Potom vyšiel 2. diel.", "Potom vyšiel druhý diel."),
    "ordinal-range": ("cs", None, "Od 1.–5. ledna.", "Od prvního až pátého ledna."),
    "shared-noun": ("cs", None, "Přelom XIX. a XX. století.", "Přelom devatenáctého a dvacátého století."),
    "noun-after-tisíc": ("cs", None, "S 1 000 Kč vyrazil.", "S tisícem korun vyrazil."),
    "grouped-digits": ("cs", None, "Po 1 000 letech.", "Po tisíci letech."),
    "capitals-agreement": ("cs", None, "PŘIŠLA TZV. VELKÁ VODA.", "PŘIŠLA takzvaná VELKÁ VODA."),
    "capitals-saint": ("cs", None, "KOSTEL SV. VÁCLAVA", "KOSTEL svatého VÁCLAVA"),
    "capitals-roman-after-name": ("cs", None, "KAREL IV. ZALOŽIL UNIVERZITU.", "KAREL čtvrtý ZALOŽIL UNIVERZITU."),
    "capitals-clitic-before-verb": ("cs", None, "KAREL IV. SE NARODIL V PRAZE. VLÁDL KAREL IV. TO BYLO DÁVNO.",
                                    "KAREL čtvrtý SE NARODIL V PRAZE. VLÁDL KAREL čtvrtý. TO BYLO DÁVNO."),
    "capitals-conjunction-after-period": ("cs", None, "KAREL IV. A VÁCLAV IV. ZALOŽILI UNIVERZITY. VLÁDL KAREL IV. A POTOM ZEMŘEL.",
                                          "KAREL čtvrtý A VÁCLAV čtvrtý ZALOŽILI UNIVERZITY. VLÁDL KAREL čtvrtý A POTOM ZEMŘEL."),
    "capitals-conjunction-after-abbreviation": ("cs", None, "BYL TAM ATD. A POTOM ODEŠEL. JABLKA, HRUŠKY ATD. A MNOHO DALŠÍHO "
                                                "LEŽELO NA STOLE.",
                                                "BYL TAM a tak dále. A POTOM ODEŠEL. JABLKA, HRUŠKY a tak dále A MNOHO DALŠÍHO "
                                                "LEŽELO NA STOLE."),
    "capitals-sentence-ends": ("cs", None, "BYL TAM ATD. POTOM ODEŠEL. STALO SE TO 5. 6. POTOM ODEŠEL.",
                               "BYL TAM a tak dále. POTOM ODEŠEL. STALO SE TO pátého června. POTOM ODEŠEL."),
    "capitals-name-goes-on": ("cs", None, "VLÁDL KAREL IV. LUCEMBURSKÝ.", "VLÁDL KAREL čtvrtý LUCEMBURSKÝ."),
    "capitals-label-numbered-before-a-new-sentence": ("cs", None, "VYŠEL DÍL V. KNIHA O NĚM VYŠLA.",
                                                      "VYŠEL DÍL pátý. KNIHA O NĚM VYŠLA."),
    "capitals-initial-after-a-label": ("cs", None, "VYŠLA KNIHA V. HAVLA.", "VYŠLA KNIHA V. HAVLA."),
    "capitals-verb-after-a-full-clause": ("cs", None, "BYL TAM ATD. ODEŠEL DOMŮ. VLÁDL KAREL IV. ZALOŽIL UNIVERZITU. "
                                          "ČEKAL 5 MIN. ODEŠEL.",
                                          "BYL TAM a tak dále. ODEŠEL DOMŮ. VLÁDL KAREL čtvrtý. ZALOŽIL UNIVERZITU. "
                                          "ČEKAL pět minut. ODEŠEL."),
    "capitals-period-before-verb-untagged": ("cs", None, "JABLKA, HRUŠKY ATD. LEŽELY NA STOLE. DNE 5. 6. ODEŠEL DOMŮ.",
                                             "JABLKA, HRUŠKY a tak dále LEŽELY NA STOLE. DNE pátého června ODEŠEL DOMŮ."),
    "capitals-list-then-verb": ("cs", None, "PŘINESL JABLKA, HRUŠKY ATD. ODEŠEL. KDYŽ PŘIŠEL, KAREL IV. ZALOŽIL UNIVERZITU.",
                                "PŘINESL JABLKA, HRUŠKY a tak dále. ODEŠEL. KDYŽ PŘIŠEL, KAREL čtvrtý ZALOŽIL UNIVERZITU."),
    "capitals-list-after-subordinate-clause": ("cs", None, "KDYŽ PŘIŠEL, JABLKA, HRUŠKY ATD. LEŽELY NA STOLE.",
                                               "KDYŽ PŘIŠEL, JABLKA, HRUŠKY a tak dále LEŽELY NA STOLE."),
    "capitals-date-and-amount-in-own-clause": ("cs", None, "PŘIŠEL A DNE 5. 6. ODEŠEL. KDYŽ PŘIŠEL, ZA 5 MIN. ODEŠEL.",
                                               "PŘIŠEL A DNE pátého června ODEŠEL. KDYŽ PŘIŠEL, ZA pět minut ODEŠEL."),
    "capitals-page": ("cs", None, "NA STR. 45 SE PÍŠE.", "NA straně čtyřicet pět SE PÍŠE."),
    "capitals-units": ("cs", None, "CENA JE 5 KČ, JEL 50 KM/H A MĚŘÍ 5 CM.",
                       "CENA JE pět korun, JEL padesát kilometrů za hodinu A MĚŘÍ pět centimetrů."),
    "capitals-one-letter-units": ("cs", None, "VZDÁLENOST 100 M, VÁHA 5 G A MÁM 5 S SEBOU.",
                                  "VZDÁLENOST sto metrů, VÁHA pět gramů A MÁM pět S SEBOU."),
    "capitals-glued-letter-is-no-unit": ("cs", None, "TELEFON PODPORUJE 5G, JE TO IPHONE 5S. BYT MÁ 60M².",
                                         "TELEFON PODPORUJE pět G, JE TO IPHONE pět S. BYT MÁ šedesát metrů čtverečních."),
    "capitals-attached-power": ("cs", None, "BYT MÁ 60M2 A SKLEP 20M3, TEDY 5M.",
                                "BYT MÁ šedesát metrů čtverečních A SKLEP dvacet metrů krychlových, TEDY pět M."),
    "capitals-scale-and-currency": ("cs", None, "STÁLO TO 5 TIS. KČ.", "STÁLO TO pět tisíc korun."),
    "sk-capitals-currency": ("sk", None, "CENA JE 5 KČ.", "CENA JE päť korún."),
    "capital-acronym-before-number": ("cs", None, "MAX. 5 KG.", "Maximálně pět kilogramů."),
    "sk-capital-acronym-before-number": ("sk", None, "PODĽA § 5 ODS. 2 PLATÍ.",
                                         "PODĽA paragrafu päť odseku dva PLATÍ."),
    "capital-era-after-number": ("cs", None, "ROKU 300 N. L. VLÁDL.", "ROKU tři sta našeho letopočtu VLÁDL."),
    "invisible-characters": ("cs", None, "\ufeffRakousko\u2011Uhersko má 5\u201110 Kč, text\u00adový.",
                             "Rakousko-Uhersko má pět až deset korun, textový."),
    "zero-width-space-between-words": ("cs", None, "Ahoj\u200bsvěte.", "Ahoj světe."),
    "approximately": ("cs", None, "Je to ~5 km, tedy ≈5 000 m.",
                      "Je to přibližně pět kilometrů, tedy přibližně pět tisíc metrů."),
    "sk-approximately": ("sk", None, "Je to ~5 km.", "Je to približne päť kilometrov."),
    "approximately-after-preposition": ("cs", None, "Pracoval s ~5 kg a bez ≈5 lidí.",
                                        "Pracoval s přibližně pěti kilogramy a bez přibližně pěti lidí."),
    "sk-approximately-after-preposition": ("sk", None, "Pracoval s ~5 kg.", "Pracoval s približne piatimi kilogramami."),
    # after "a" and an approximator the amount shares the case of the one before; "pak" opens a clause of its own
    "approximately-after-conjunction": ("cs", None, "Šel s 5 lidmi a ~3 psy. Pracoval s 2 kg a asi 3 kg. "
                                                    "Přišel s 5 přáteli a pak 3 psi utekli.",
                                        "Šel s pěti lidmi a přibližně třemi psy. Pracoval se dvěma kilogramy a asi třemi "
                                        "kilogramy. Přišel s pěti přáteli a pak tři psi utekli."),
    "approximately-before-currency": ("cs", None, "Stálo to ≈$5, tedy ~ €5 a ≈ -$5.",
                                      "Stálo to přibližně pět dolarů, tedy přibližně pět eur a přibližně mínus pět dolarů."),
    "approximately-between-numbers": ("cs", None, "Platí 3≈4 a 3~4.", "Platí tři přibližně čtyři a tři přibližně čtyři."),
    "approximately-plus-minus": ("cs", None, "Je to ≈±5 °C a ~±5 °C.",
                                 "Je to přibližně plus minus pět stupňů Celsia a přibližně plus minus pět stupňů Celsia."),
    "asterisk-times": ("cs", None, "Spočítej 3*4 a 5 * 6.", "Spočítej tři krát čtyři a pět krát šest."),
    "asterisk-before-currency": ("cs", None, "Spočítej 3*$4 a 3 * + $4.",
                                 "Spočítej tři krát čtyři dolary a tři krát plus čtyři dolary."),
    "signed-operand-after-new-operators": ("cs", None, "Spočítej 3*-4, je to ~-5 °C.",
                                           "Spočítej tři krát mínus čtyři, je to přibližně mínus pět stupňů Celsia."),
    "range-with-az": ("cs", None, "Při 1 až 2 °C voda zamrzne. Stálo to $5 až 10.",
                      "Při jednom až dvou stupních Celsia voda zamrzne. Stálo to pět až deset dolarů."),
    "sk-range-with-az": ("sk", None, "Pri 1 až 2 °C voda zamrzne.", "Pri jednom až dvoch stupňoch Celzia voda zamrzne."),
    "range-with-az-before-sentence-end": ("cs", None, "Počítal s 1 až 2. Potom odešel.",
                                          "Počítal s jedním až dvěma. Potom odešel."),
    "capitals-ordinal-after-az": ("cs", None, "ZLEPŠENÍ NASTANE 5 AŽ 6. DEN.", "ZLEPŠENÍ NASTANE pět AŽ šestý DEN."),
    "capitals-verb-past-an-ordinal-after-az": ("cs", None, "DOSÁHL AŽ 3. DEN 5 BODŮ.", "DOSÁHL AŽ třetí DEN pěti BODŮ."),
    "ordinal-before-sentence-with-conjunction": ("cs", None, "Bylo jich 1. A 2. díl vyšel.",
                                                 "Bylo jich jeden. A druhý díl vyšel."),
    "capitals-ordinal-chain": ("cs", None, "ODE DNE 1. AŽ 5. LEDNA A 1. A 2. DÍL. BYLO JICH 5. A PAK ODEŠEL.",
                               "ODE DNE prvního AŽ pátého LEDNA A první A druhý DÍL. BYLO JICH pět. A PAK ODEŠEL."),
    # in capitals an ordinal after "A" leads its noun, as a lowercase "a" shows in mixed text; after a cardinal it
    # ends the list and the sentence
    "capitals-ordinal-after-conjunction": ("cs", None, "BYLO JICH 1. A 2. DÍL VYŠEL. VYŠEL 2. DÍL A 3. DÍL VYŠEL "
                                                       "POZDĚJI. BOJOVALI V LETECH 1914 A 1918. VÁLKA SKONČILA.",
                                           "BYLO JICH první A druhý DÍL VYŠEL. VYŠEL druhý DÍL A třetí DÍL VYŠEL "
                                           "POZDĚJI. BOJOVALI V LETECH tisíc devět set čtrnáct A tisíc devět set osmnáct. "
                                           "VÁLKA SKONČILA."),
    "ordinal-range-with-az": ("cs", None, "Od 1. až 5. ledna. Přečti 2. až 4. kapitolu. XIX. až XX. století bylo bohaté.",
                              "Od prvního až pátého ledna. Přečti druhou až čtvrtou kapitolu. Devatenácté až dvacáté "
                              "století bylo bohaté."),
    "az-before-time-or-word": ("cs", None, "Otevřeno 8 až 9.30 hod. a 8 až 9:30. Opakuj to 2 až 3krát, byl to 2 až 3letý "
                                           "chlapec a zlepšení nastane 5 až 6. den.",
                               "Otevřeno osm až devět třicet a osm až devět třicet. Opakuj to dva až třikrát, byl to dva až "
                               "tříletý chlapec a zlepšení nastane pět až šestý den."),
    "comma-list": ("cs", None, "Zvol 1,2,3 nebo 4,5,6.", "Zvol jedna, dva, tři nebo čtyři, pět, šest."),
    "comma-list-before-noun": ("cs", None, "Šel s 1,2,3 přáteli.", "Šel s jedním, dvěma, třemi přáteli."),
    "comma-list-with-unit": ("cs", None, "Zvol 1,2,3 kg nebo 1,2,3 Kč, šel s 1,2,3 kg.",
                             "Zvol jeden, dva, tři kilogramy nebo jedna, dvě, tři koruny, šel s jedním, dvěma, třemi kilogramy."),
    "comma-list-period-and-preposition": ("cs", None, "Balení 1,2,3 ks. Potom šel s 2,3,4 kg.",
                                          "Balení jeden, dva, tři kusy. Potom šel se dvěma, třemi, čtyřmi kilogramy."),
    "comma-list-with-rate": ("cs", None, "Tok 1,2,3 kg/s, cena 1,2,3 Kč/kg.",
                             "Tok jeden, dva, tři kilogramy za sekundu, cena jedna, dvě, tři koruny za kilogram."),
    "comma-list-with-scale": ("cs", None, "Stálo to 1,2,3 tis. Kč. Šel s 1,2,3 mil.",
                              "Stálo to jeden, dva, tři tisíce korun. Šel s jedním, dvěma, třemi miliony."),
    "sk-comma-list-with-thousands": ("sk", None, "Stálo to 1,2,3 tis. Kč.", "Stálo to tisíc, dvetisíc, tritisíc korún."),
    "verse-references": ("cs", {"verse_references": ["Jan", "Mt"]},
                         "Viz Jan 3,16, Mt 5,3–12, Mt 5,3–7,29 a Jan 3:16. Jan přišel v 5,5.",
                         "Viz Jan tři, šestnáct, Mt pět, tři až dvanáct, Mt pět, tři až sedm, dvacet devět a "
                         "Jan tři, šestnáct. Jan přišel v pět celých pět desetin."),
    "sk-verse-references": ("sk", {"verse_references": ["Ján"]}, "Pozri Ján 3,16.", "Pozri Ján tri, šestnásť."),
    "verse-references-with-more-verses": ("cs", {"verse_references": ["Jan"]},
                                          "Viz Jan 3,16.18 a Jan 3,16–18.20. Viz Jan 3,16. 18 lidí přišlo.",
                                          "Viz Jan tři, šestnáct a osmnáct a Jan tři, šestnáct až osmnáct a dvacet. "
                                          "Viz Jan tři, šestnáct. Osmnáct lidí přišlo."),
    "verse-reference-with-english-book": ("cs", {"verse_references": ["John"], "english": ["John"]}, "Viz John 3,16.",
                                          "Viz <en>John</en> tři, šestnáct."),
    "verse-references-off": ("cs", None, "Viz Jan 3,16.", "Viz Jan tři celé šestnáct setin."),  # "Jan" is a name too
    "numbered-verse-books": ("cs", {"verse_references": ["Jan", "Kor", "Sam"]}, "Viz 1 Jan 3,16, 2. Kor 5,17 a 2Sam 7,12.",
                             "Viz první Jan tři, šestnáct, druhý Kor pět, sedmnáct a druhá Sam sedm, dvanáct."),
    "number-before-a-verse-reference": ("cs", {"verse_references": ["Jan"]}, "Bylo jich 5. Jan 3,16 to říká.",
                                        "Bylo jich pět. Jan tři, šestnáct to říká."),
    "per-second-before-word": ("cs", None, "Jel 5 m / s a pak šel, tok měl 5 l / s a víc.",
                               "Jel pět metrů za sekundu a pak šel, tok měl pět litrů za sekundu a víc."),
    "sk-per-second-before-word": ("sk", None, "Išiel 5 m / s a potom zastal.",
                                  "Išiel päť metrov za sekundu a potom zastal."),
    "per-second-attached-before-word": ("cs", None, "Jel 5 km/s a dál, tok 5 l/s a víc, pak 5 KM/S a dál.",
                                        "Jel pět kilometrů za sekundu a dál, tok pět litrů za sekundu a víc, pak pět "
                                        "kilometrů za sekundu a dál."),
    "per-second-after-power": ("cs", None, "Plocha roste 5 m² / s a dál, pak 5 m2 / s a víc.",
                               "Plocha roste pět metrů čtverečních za sekundu a dál, pak pět metrů čtverečních za "
                               "sekundu a víc."),
    "per-second-before-word-in-range": ("cs", None, "Jel 5–10 m / s a pak šel. Jel 5 m / s – 10 m / s a stál. "
                                        "Letěl 5 tis. km / s a pak 5–10 tis. km / s a zmizel.",
                                        "Jel pět až deset metrů za sekundu a pak šel. Jel pět metrů za sekundu až deset "
                                        "metrů za sekundu a stál. Letěl pět tisíc kilometrů za sekundu a pak pět až deset "
                                        "tisíc kilometrů za sekundu a zmizel."),
    "capitals-per-second-before-word": ("cs", None, "JEL 5 M / S A PAK 5 KM / S A DÁL.",
                                        "JEL pět metrů za sekundu A PAK pět kilometrů za sekundu A DÁL."),
    "unit-slash-is-no-preposition": ("cs", None, "Jel 5 m/s a 5 km/h.",
                                     "Jel pět metrů za sekundu a pět kilometrů za hodinu."),
    # mezi, nad, pod, před, za: a masculine plural in -y is the same in the accusative and the instrumental
    "between-place": ("cs", None, "Stál mezi 2 stromy.", "Stál mezi dvěma stromy."),
    "between-place-verb-after": ("cs", None, "Mezi 2 stromy stála lavička.", "Mezi dvěma stromy stála lavička."),
    "between-direction": ("cs", None, "Postavil se mezi 2 stromy.", "Postavil se mezi dva stromy."),
    "behind-place": ("cs", None, "Stál za 2 stoly.", "Stál za dvěma stoly."),
    "behind-while-going": ("cs", None, "Celou cestu jel za 2 vozy, pak se schoval za 2 stromy.",
                           "Celou cestu jel za dvěma vozy, pak se schoval za dva stromy."),
    "ago-and-in": ("cs", None, "Před 2 roky odjel, přijel před 2 týdny a za 2 roky se vrátí.",
                   "Před dvěma roky odjel, přijel před dvěma týdny a za dva roky se vrátí."),
    "threshold-after-motion": ("cs", None, "Teplota klesla pod 5 °C, dnes je pod 5 °C.",
                               "Teplota klesla pod pět stupňů Celsia, dnes je pod pěti stupni Celsia."),
    "threshold-range-after-motion": ("cs", None,
                                     "Teplota klesla pod 1–2 °C, pak vystoupala nad 10–20 °C. Cena klesla pod 5–10 tis. Kč.",
                                     "Teplota klesla pod jeden až dva stupně Celsia, pak vystoupala nad deset až dvacet "
                                     "stupňů Celsia. Cena klesla pod pět až deset tisíc korun."),
    "threshold-list-after-motion": ("cs", None, "Teploty klesly pod 5 °C, 3 °C a 1 °C. Pak klesly pod 3,2,1 °C.",
                                    "Teploty klesly pod pět stupňů Celsia, tři stupně Celsia a jeden stupeň Celsia. "
                                    "Pak klesly pod tři, dva, jeden stupeň Celsia."),
    "genitive-verb-across-conjunction": ("cs", None, "Dosáhl cíle a 2. místa. Bál se tmy a 2. dílu.",
                                         "Dosáhl cíle a druhého místa. Bál se tmy a druhého dílu."),
    "genitive-verb-subject-conjunct": ("cs", None, "Cíle dosáhl Petr a 5 mužů. Dosáhl cíle a 5 bodů.",
                                       "Cíle dosáhl Petr a pět mužů. Dosáhl cíle a pěti bodů."),
    "present-genitive-verb-after-its-subject": ("cs", None, "Petr dosáhne 5 bodů. Teplota dosáhne 5 stupňů. "
                                                            "Hráči dosáhnou 5 bodů.",
                                                "Petr dosáhne pěti bodů. Teplota dosáhne pěti stupňů. "
                                                "Hráči dosáhnou pěti bodů."),
    # a singular inanimate subject, but no accusative of time: "každý rok" and "akce" are no subjects
    "present-genitive-verb-after-an-inanimate-subject": ("cs", None, "Auto dosáhne 5 metrů. Zákon se týká 50 lidí. "
                                                                     "Každý rok se akce zúčastní 50 lidí.",
                                                         "Auto dosáhne pěti metrů. Zákon se týká padesáti lidí. "
                                                         "Každý rok se akce zúčastní padesát lidí."),
    "present-genitive-verb-before-its-subject": ("cs", None, "Dosáhne 5 bodů právě Petr. Zúčastní se 50 lidí z Prahy.",
                                                 "Dosáhne pěti bodů právě Petr. Zúčastní se padesát lidí z Prahy."),
    "present-genitive-verb-subject-in-its-clause": ("cs", None, "Petr dosáhne vítězství a 5 bodů. Petr dosáhne cíle a 5 bodů.",
                                                    "Petr dosáhne vítězství a pěti bodů. Petr dosáhne cíle a pěti bodů."),
    # CAC tags a neuter in -í after 5 nominative singular, but it can only be a genitive plural there
    "one-form-neuter-after-five": ("cs", None, "Dosáhl 5 vítězství. Bál se 5 rozhodnutí. Má 5 stavení.",
                                   "Dosáhl pěti vítězství. Bál se pěti rozhodnutí. Má pět stavení."),
    "asking-verbs-take-the-genitive": ("cs", None, "Ptal se 5 mužů. Chtěl se zeptat 5 lidí. Ptal se na 2. kapitolu.",
                                       "Ptal se pěti mužů. Chtěl se zeptat pěti lidí. Ptal se na druhou kapitolu."),
    "genitive-verb-plain-subject-conjunct": ("cs", None, "Cíle dosáhla Eva a 5 žen. Cíle dosáhl Jiří a 5 mužů. "
                                                         "Cíle dosáhl Petr a asi 5 mužů.",
                                             "Cíle dosáhla Eva a pět žen. Cíle dosáhl Jiří a pět mužů. "
                                             "Cíle dosáhl Petr a asi pět mužů."),
    "genitive-verb-plural-subject-conjunct": ("cs", None, "Cíle dosáhli Petr a 5 mužů. Petr a 5 mužů dosáhli cíle. "
                                                          "Báli se otce a 5 mužů.",
                                              "Cíle dosáhli Petr a pět mužů. Petr a pět mužů dosáhli cíle. "
                                              "Báli se otce a pěti mužů."),
    "genitive-verb-object-put-first": ("cs", None, "Cíle dosáhl vůdce a 5 mužů. Cíle dosáhli 5 procent studentů.",
                                       "Cíle dosáhl vůdce a pět mužů. Cíle dosáhli pět procent studentů."),
    "genitive-verb-genitive-object-put-first": ("cs", None, "Úspěchu dosáhl vůdce a 5 mužů. "
                                                            "Voleb se zúčastnili 60 procent voličů.",
                                                "Úspěchu dosáhl vůdce a pět mužů. "
                                                "Voleb se zúčastnili šedesát procent voličů."),
    "genitive-verb-soft-genitive-object-put-first": ("cs", None, "Konce se dočkal vůdce a 5 rytířů.",
                                                     "Konce se dočkal vůdce a pět rytířů."),
    "genitive-verb-before-five-and-more": ("cs", None, "Dosáhl 5 bodů a zúčastnilo se 50 lidí. Chce dosáhnout 5 bodů. "
                                                       "5 mužů dosáhlo cíle. Vzdal 5 bodů.",
                                           "Dosáhl pěti bodů a zúčastnilo se padesát lidí. Chce dosáhnout pěti bodů. "
                                           "Pět mužů dosáhlo cíle. Vzdal pět bodů."),
    "number-before-colloquial-plural-verb": ("cs", None, "Pak 5 mužů dosáhli cíle. 5 dětí se bály tmy. 5 bodů dosáhl.",
                                             "Pak pět mužů dosáhli cíle. Pět dětí se bály tmy. Pěti bodů dosáhl."),
    "number-before-its-verb-and-subject": ("cs", None, "5 chyb si nikdo nevšimne. 5 bodů Petr nedosáhne. 5 psů se vojáci "
                                                       "báli. 5 bodů dosáhne Petr.",
                                           "Pěti chyb si nikdo nevšimne. Pěti bodů Petr nedosáhne. Pěti psů se vojáci "
                                           "báli. Pěti bodů dosáhne Petr."),
    "genitive-noun-after-one-to-four": ("cs", None, "Dosáhl 2 bodů, dosáhli 3 bodů a dosáhl 1 bodu. Vypil 2 piva a ve 2 "
                                                    "hodiny odešel.",
                                        "Dosáhl dvou bodů, dosáhli tří bodů a dosáhl jednoho bodu. Vypil dvě piva a ve dvě "
                                        "hodiny odešel."),
    # a noun shared from a later number is genitive by that number's own right after 5, and "piva" is tagged
    # a genitive singular
    "one-to-four-sharing-a-later-noun": ("cs", None, "Vypil 1 a 2 piva. Vypil 2 nebo 5 piv. Dosáhl 2 a 3 bodů. "
                                                     "Bál se 2 nebo 5 psů.",
                                         "Vypil jedno a dvě piva. Vypil dvě nebo pět piv. Dosáhl dvou a tří bodů. "
                                         "Bál se dvou nebo pěti psů."),
    # CAC tags "hodiny" a genitive plural here, a form no genitive plural has
    # a number with no noun takes the gender of the unit after a later one
    "unit-shared-across-a-link": ("cs", None, "Mělo to 1 nebo 2 %. Od 1 do 2 h. Bylo od 1 do 2 hodin. Mezi 1 a 2 h.",
                                  "Mělo to jedno nebo dvě procenta. Od jedné do dvou hodin. Bylo od jedné do dvou hodin. "
                                  "Mezi jednou a dvěma hodinami."),
    "noun-shared-past-an-approximator": ("cs", None, "Má 1 nebo asi 2 děti. Od 1 do asi 2 h.",
                                         "Má jedno nebo asi dvě děti. Od jedné do asi dvou hodin."),
    "amount-after-approximator-opening-a-clause": ("cs", None, "Experimentoval s 2 kg a asi 3 kg zůstaly.",
                                                   "Experimentoval se dvěma kilogramy a asi tři kilogramy zůstaly."),
    # a phrase put before its verb keeps the shared preposition: that verb is the clause's own, not a new clause's
    "fronted-amounts-after-an-approximator": ("cs", None, "S 2 kg a asi 3 kg pracoval. S 5 lidmi a asi 3 psy přišel.",
                                              "Se dvěma kilogramy a asi třemi kilogramy pracoval. S pěti lidmi a asi "
                                              "třemi psy přišel."),
    "approximator-inside-a-shared-preposition-chain": ("cs", None, "Pracoval s 2 kg a asi 3 kg a 4 kg.",
                                                       "Pracoval se dvěma kilogramy a asi třemi kilogramy a čtyřmi "
                                                       "kilogramy."),
    "infinitive-opens-no-clause": ("cs", None, "Nemohl s 2 kg a asi 3 kg pracovat. Chtěl s 5 lidmi a asi 3 psy odejít.",
                                   "Nemohl se dvěma kilogramy a asi třemi kilogramy pracovat. Chtěl s pěti lidmi a asi "
                                   "třemi psy odejít."),
    "auxiliary-is-not-the-clause-verb": ("cs", None, "Včera jsem s 2 kg a asi 3 kg pracoval. Ráno jsme s 5 lidmi a asi 3 psy "
                                                     "odešli. Pracovat s 2 kg a asi 3 kg je těžké.",
                                         "Včera jsem se dvěma kilogramy a asi třemi kilogramy pracoval. Ráno jsme s pěti "
                                         "lidmi a asi třemi psy odešli. Pracovat se dvěma kilogramy a asi třemi kilogramy "
                                         "je těžké."),
    "amount-opening-a-clause-without-an-approximator": ("cs", None, "Pracoval s 2 kg a 3 kg zůstaly. S 2 kg a 3 kg pracoval. "
                                                                    "Začal s 2 kg a 3 kg cvičit. Včera jsem s 2 kg a 3 kg "
                                                                    "pracoval.",
                                                        "Pracoval se dvěma kilogramy a tři kilogramy zůstaly. Se dvěma "
                                                        "kilogramy a třemi kilogramy pracoval. Začal se dvěma kilogramy a "
                                                        "třemi kilogramy cvičit. Včera jsem se dvěma kilogramy a třemi "
                                                        "kilogramy pracoval."),
    "amount-with-a-symbol-or-range-opening-a-clause": ("cs", None, "Pracoval s 2 € a 3 € zůstaly. "
                                                                   "Pracoval s 1 kg a 2–3 kg zůstaly.",
                                                       "Pracoval se dvěma eury a tři eura zůstaly. "
                                                       "Pracoval s jedním kilogramem a dva až tři kilogramy zůstaly."),
    "amount-with-a-period-opening-no-clause": ("cs", None, "Pracoval s 2 h a 3 h. Zůstaly mu.",
                                               "Pracoval se dvěma hodinami a třemi hodinami. Zůstaly mu."),
    "unit-shared-after-od-and-an-approximator": ("cs", None, "Od přibližně 1 do 2 h. Bylo od asi 1 do 2 hodin.",
                                                 "Od přibližně jedné do dvou hodin. Bylo od asi jedné do dvou hodin."),
    "unit-shared-across-a-link-in-capitals": ("cs", None, "TRVALO TO 1 NEBO 2 H.", "TRVALO TO jednu NEBO dvě hodiny."),
    "one-sharing-a-neuter-plural": ("cs", None, "Má 1 nebo 2 děti. Měl 1 nebo 2 oči zavřené.",
                                    "Má jedno nebo dvě děti. Měl jedno nebo dvě oči zavřené."),
    "one-to-four-before-a-mistagged-genitive": ("cs", None, "Hrál si s dětmi a skoro 2 hodiny tam zůstal. Bál se 2 žen.",
                                                "Hrál si s dětmi a skoro dvě hodiny tam zůstal. Bál se dvou žen."),
    "sk-genitive-noun-after-one-to-four": ("sk", None, "Bál sa 2 žien.", "Bál sa dvoch žien."),
    "sk-genitive-verb-object": ("sk", None, "Bál sa 5 psov. Týka sa to 5 ľudí. Báli sme sa 25 psov. "
                                "Chce sa zbaviť 5 kíl. Bál sa 5 až 6 psov. Bál sa 5 psov a 2 mačiek.",
                                "Bál sa piatich psov. Týka sa to piatich ľudí. Báli sme sa dvadsiatich piatich psov. "
                                "Chce sa zbaviť piatich kíl. Bál sa piatich až šiestich psov. "
                                "Bál sa piatich psov a dvoch mačiek."),
    "sk-genitive-verb-neuter-in-i": ("sk", None, "Bál sa 5 stretnutí.", "Bál sa piatich stretnutí."),
    "genitive-verbs": ("cs", None, "Dosáhli 5. místa, bál se 2. dílu a vzdal se 2. kola.",
                       "Dosáhli pátého místa, bál se druhého dílu a vzdal se druhého kola."),
    "accusative-without-clitic": ("cs", None, "Vzdal 2. kolo.", "Vzdal druhé kolo."),
    "negated-verbs": ("cs", None, "Nedosáhl 20. století, nevzpomínal na XX. století a nepostavil se mezi 2 stromy.",
                      "Nedosáhl dvacátého století, nevzpomínal na dvacáté století a nepostavil se mezi dva stromy."),
    "genitive-verbs-present": ("cs", None, "Budova dosahuje 2. podlaží. Obávají se 2. kola a dočkají se 2. dílu. "
                                           "Co se týče 2. století.",
                               "Budova dosahuje druhého podlaží. Obávají se druhého kola a dočkají se druhého dílu. "
                               "Co se týče druhého století."),
    "genitive-verb-with-subject-after": ("cs", None, "Cíle dosáhl 2. muž. Cíle dosáhli 2. muži. Cíle dosáhnou 2. muži.",
                                         "Cíle dosáhl druhý muž. Cíle dosáhli druzí muži. Cíle dosáhnou druzí muži."),
    "genitive-verb-with-neuter-subject": ("cs", None, "Cíle dosáhlo 2. sdružení. Dožil se 2. tisíciletí. Dosáhlo to 2. výročí. "
                                          "Cíle dosáhne 2. sdružení.",
                                          "Cíle dosáhlo druhé sdružení. Dožil se druhého tisíciletí. Dosáhlo to druhého výročí. "
                                          "Cíle dosáhne druhé sdružení."),
    "genitive-verb-subject-to": ("cs", None, "Dosáhlo to 5 bodů. Toto dosáhlo 5 bodů.",
                                 "Dosáhlo to pěti bodů. Toto dosáhlo pěti bodů."),
    "guard-genitive-verb-to-taking-no-part": ("cs", None, "Zúčastnilo se to 50 lidí.", "Zúčastnilo se to padesát lidí."),
    "genitive-verb-measures": ("cs", None, "Teplota dosáhla 30 °C. Dosáhl 5 Kč. Cena dosáhla 4,50 €. "
                                           "Cena dosáhla 0,50 €. Cena dosáhla 100 Kč/kg.",
                               "Teplota dosáhla třiceti stupňů Celsia. Dosáhl pěti korun. Cena dosáhla čtyř eur "
                               "padesáti centů. Cena dosáhla padesáti centů. Cena dosáhla sta korun za kilogram."),
    "genitive-verb-measure-ranges-and-lists": ("cs", None, "Cena dosáhla $5–10. Zbavil se 1,2,3 kg. Dosáhl 5 tis. Kč.",
                                               "Cena dosáhla pěti až deseti dolarů. "
                                               "Zbavil se jednoho, dvou, tří kilogramů. Dosáhl pěti tisíc korun."),
    "genitive-verb-measure-after-its-own-noun": ("cs", None, "Dosáhl rychlosti 120 km/h. "
                                                             "Teplota dosáhla během dne 30 °C. "
                                                             "Teplota dosáhla 30,5 °C.",
                                                 "Dosáhl rychlosti sto dvacet kilometrů za hodinu. "
                                                 "Teplota dosáhla během dne třiceti stupňů Celsia. "
                                                 "Teplota dosáhla třicet celých pět desetin stupně Celsia."),
    "genitive-verb-time-after-its-object": ("cs", None, "Dosáhl cíle 5 minut před ostatními. "
                                                        "Zbavil se bolesti 5 hodin po operaci.",
                                            "Dosáhl cíle pět minut před ostatními. "
                                            "Zbavil se bolesti pět hodin po operaci."),
    "zero-after-genitive-verb": ("cs", None, "Teplota dosáhla 0 °C. Teplota se dotkla 0 stupňů. Dosáhl 0 bodů.",
                                 "Teplota dosáhla nuly stupňů Celsia. Teplota se dotkla nuly stupňů. Dosáhl nuly bodů."),
    "genitive-verb-number-after-a-measure-and-a": ("cs", None, "Dosáhl 5 km a 10 bodů. Dosáhl 5 km a 6 km.",
                                                   "Dosáhl pěti kilometrů a deseti bodů. "
                                                   "Dosáhl pěti kilometrů a šesti kilometrů."),
    "genitive-verb-number-list": ("cs", None, "Dosáhl 5, 10 nebo 15 bodů. Dosáhl 2, 3 nebo 5 bodů.",
                                  "Dosáhl pěti, deseti nebo patnácti bodů. Dosáhl dvou, tří nebo pěti bodů."),
    "number-list-before-its-verb-and-after-it": ("cs", None, "5, 10 nebo 15 bodů Petr nedosáhne.",
                                                 "Pěti, deseti nebo patnácti bodů Petr nedosáhne."),
    "number-list-joined-to-a-subject": ("cs", None, "Cíle dosáhl Petr a 5 nebo 10 mužů.",
                                        "Cíle dosáhl Petr a pět nebo deset mužů."),
    "genitive-verb-measure-list": ("cs", None, "Dosáhl 5, 10 nebo 15 Kč.", "Dosáhl pěti, deseti nebo patnácti korun."),
    "measure-list-after-its-own-noun": ("cs", None, "Dosáhl rychlosti 120, 130 nebo 140 km/h.",
                                        "Dosáhl rychlosti sto dvacet, sto třicet nebo "
                                        "sto čtyřicet kilometrů za hodinu."),
    "preposition-before-a-list-subject": ("cs", None, "Dosáhl cíle pro 5, 10 nebo 15 hráčů.",
                                          "Dosáhl cíle pro pět, deset nebo patnáct hráčů."),
    "year-label-before-a-count": ("cs", None, "Dožil se roku 1990, 5 dní po svatbě.",
                                  "Dožil se roku tisíc devět set devadesát, pět dní po svatbě."),
    "comma-list-after-genitive-verb": ("cs", None, "Dosáhl 1,2,3 bodů. Bál se 2,3,4 psů.",
                                       "Dosáhl jednoho, dvou, tří bodů. Bál se dvou, tří, čtyř psů."),
    "comma-list-after-accusative-verb": ("cs", None, "Vypil 1,2,3 piva.", "Vypil jedno, dvě, tři piva."),
    "number-sharing-a-numeral-subject": ("cs", None, "Zúčastnilo se 2 nebo 50 lidí.",
                                         "Zúčastnilo se dva nebo padesát lidí."),
    "number-sharing-a-genitive-object": ("cs", None, "Bál se 2 nebo 5 psů.", "Bál se dvou nebo pěti psů."),
    "sk-number-sharing-a-genitive-verb-object": ("sk", None, "Bál sa 2 alebo 5 žien. Bál sa 3 alebo 5 mačiek. "
                                                 "Bál sa 2 alebo asi 5 žien.",
                                                 "Bál sa dvoch alebo piatich žien. Bál sa troch alebo piatich mačiek. "
                                                 "Bál sa dvoch alebo asi piatich žien."),
    "number-sharing-an-accusative-noun": ("cs", None, "Vypil 2 nebo 5 piv.", "Vypil dvě nebo pět piv."),
    "one-sharing-a-later-noun": ("cs", None, "Dosáhl 1 nebo 2 bodů.", "Dosáhl jednoho nebo dvou bodů."),
    "one-sharing-a-numeral-subject": ("cs", None, "Zúčastnilo se 1 nebo 50 lidí.",
                                      "Zúčastnilo se jeden nebo padesát lidí."),
    "one-sharing-a-subject-noun": ("cs", None, "Přišla 1 nebo 2 ženy.", "Přišla jedna nebo dvě ženy."),
    "preposition-of-a-list-of-numbers": (
        "cs", None, "Byl tam bez 5 nebo 10 lidí. Stál mezi 1 a 2 stromy. Šel bez 5, 10 nebo 15 lidí.",
        "Byl tam bez pěti nebo deseti lidí. Stál mezi jedním a dvěma stromy. Šel bez pěti, deseti nebo patnácti lidí."),
    "one-sharing-a-plural-tagged-singular": (
        "cs", None, "Trvalo to 1 nebo 2 hodiny. Stálo to 1 nebo 2 koruny. Spal 1 nebo 2 hodiny.",
        "Trvalo to jednu nebo dvě hodiny. Stálo to jednu nebo dvě koruny. Spal jednu nebo dvě hodiny."),
    "guard-one-sharing-a-plural-subject-or-predicate": (
        "cs", None, "Byla to 1 nebo 2 hodiny. Zbývá 1 nebo 2 minuty.",
        "Byla to jedna nebo dvě hodiny. Zbývá jedna nebo dvě minuty."),
    "zero-before-a-written-noun": (
        "cs", None, "S 0 body skončil. Při 0 stupních Celsia. Za 0 bodů. S 0 bodů.",
        "S nula body skončil. Při nula stupních Celsia. Za nula bodů. S nulou bodů."),
    "zero-genitive-preposition-before-a-written-noun": (
        "cs", None, "Bez 0 bodů. Mám 0 bodů.", "Bez nula bodů. Mám nula bodů."),
    "zero-before-a-unit": (
        "cs", None, "Při 0 °C voda mrzne. Koupil to za 0 Kč. Přijdu v 0 h. Po 0:00 odešel.",
        "Při nula stupních Celsia voda mrzne. Koupil to za nula korun. Přijdu v nula hodin. Po nula hodinách odešel."),
    "zero-genitive-preposition-before-a-unit": (
        "cs", None, "Do 0 °C. Bez 0 Kč.", "Do nula stupňů Celsia. Bez nula korun."),
    "unit-after-a-written-scale-word": (
        "cs", None, "Mám 5 milionů Kč, 2 tisíce € a 5 tisíc m².",
        "Mám pět milionů korun, dva tisíce eur a pět tisíc metrů čtverečních."),
    "guard-written-scale-word-before-a-word": ("cs", None, "Mám 5 tisíc s sebou.", "Mám pět tisíc s sebou."),
    "genitive-verb-list-of-noun-phrases": (
        "cs", None, "Dosáhl 5 bodů, 10 bodů nebo 15 bodů. Cena dosáhla 5 Kč, 10 Kč nebo 15 Kč. "
                    "Vrátil se z 5 měst, 10 vesnic a 15 osad.",
        "Dosáhl pěti bodů, deseti bodů nebo patnácti bodů. Cena dosáhla pěti korun, deseti korun nebo "
        "patnácti korun. Vrátil se z pěti měst, deseti vesnic a patnácti osad."),
    "guard-noun-phrase-then-a-time-or-a-clause": (
        "cs", None, "Dosáhl 5 bodů, 10 minut před koncem. Zůstal bez 5 korun a 10 lidí odešlo. "
                    "Dosáhl 5, 10 nebo 15 bodů, 3 jich odešlo.",
        "Dosáhl pěti bodů, deset minut před koncem. Zůstal bez pěti korun a deset lidí odešlo. "
        "Dosáhl pěti, deseti nebo patnácti bodů, tři jich odešlo."),
    "genitive-verb-list-of-counted-subjects": (
        "cs", None, "Cíle dosáhli 5 mužů, 10 žen a 15 dětí. 2 muži a 5 žen dosáhli cíle.",
        "Cíle dosáhli pět mužů, deset žen a patnáct dětí. Dva muži a pět žen dosáhli cíle."),
    "genitive-verb-list-of-counted-subjects-before-a-plural-verb": (
        "cs", None, "5 mužů, 10 žen a 15 dětí dosáhli cíle.", "Pět mužů, deset žen a patnáct dětí dosáhli cíle."),
    "guard-genitive-verb-list-of-counted-objects": (
        "cs", None, "5 bodů, 10 bodů a 15 bodů dosáhli hráči. Dosáhli 5 bodů, 10 bodů a 15 bodů.",
        "Pěti bodů, deseti bodů a patnácti bodů dosáhli hráči. Dosáhli pěti bodů, deseti bodů a patnácti bodů."),
    "na-with-accusative-verb": ("cs", None, "Vzpomínal na XX. století.", "Vzpomínal na dvacáté století."),
    "ordinal-before-capitalised-noun-in-case": ("cs", None, "V 5. Symfonii zazněl sbor, o 5. Symfonii psal.",
                                                "V páté Symfonii zazněl sbor, o páté Symfonii psal."),
    "sk-ordinal-before-capitalised-noun-in-case": ("sk", None, "V 5. Symfónii zaznel zbor.",
                                                   "V piatej Symfónii zaznel zbor."),
    "number-ends-sentence-after-preposition": ("cs", None, "Čekal na 2. Vlak přijel.", "Čekal na dva. Vlak přijel."),
    "roman-numbers-a-thing": ("cs", None, "Vyšel díl V. Kniha byla úspěšná.", "Vyšel díl pátý. Kniha byla úspěšná."),
    "roman-numbers-a-thing-before-adjective": ("cs", None, "Vyšel díl V. Nové vydání uspělo. Vyšel díl X. Toto vydání uspělo.",
                                               "Vyšel díl pátý. Nové vydání uspělo. Vyšel díl desátý. Toto vydání uspělo."),
    "roman-numbers-a-thing-before-pronoun": ("cs", None, "Vyšel díl V. On uspěl. Vyšel díl X. Poté kniha uspěla.",
                                             "Vyšel díl pátý. On uspěl. Vyšel díl desátý. Poté kniha uspěla."),
    "roman-numbers-a-thing-before-a-phrase": ("cs", None, "Vyšel díl V. Kniha o něm vyšla. Pan V. Kniha o něm vyšla.",
                                              "Vyšel díl pátý. Kniha o něm vyšla. Pan V. Kniha o něm vyšla."),
    "label-numbered-before-a-capitalised-noun": ("cs", None, "Kapitola V. Návrat domů.", "Kapitola pátá. Návrat domů."),
    "initial-before-adjective-surname": ("cs", None, "Firma V. Nový vznikla.", "Firma V. Nový vznikla."),
    "initial-after-a-woman": ("cs", None, "Paní V. Ona poté odešla. Vyšla část V. Kniha byla úspěšná.",
                              "Paní V. Ona poté odešla. Vyšla část pátá. Kniha byla úspěšná."),
    "initial-after-a-person": ("cs", None, "Autor V. Kovář napsal knihu.", "Autor V. Kovář napsal knihu."),
    "ruler-number-after-a-title": ("cs", None, "Císař Karel V. Kniha o něm vyšla. Vládl kníže Václav I. Kniha o něm vyšla.",
                                   "Císař Karel pátý. Kniha o něm vyšla. Vládl kníže Václav první. Kniha o něm vyšla."),
    "roman-numeral-before-a-sentence-led-by-a-name": ("cs", None, "Za vlády císaře Karla IV. Země vzkvétala.",
                                                      "Za vlády císaře Karla čtvrtého. Země vzkvétala."),
    "sk-roman-numeral-before-a-sentence-led-by-a-name": ("sk", None, "Vládol Karol IV. Kniha o ňom vyšla.",
                                                         "Vládol Karol štvrtý. Kniha o ňom vyšla."),
    "roman-numeral-before-an-epithet": ("cs", None, "Za Karla IV. Lucemburského vznikla univerzita.",
                                        "Za Karla čtvrtého Lucemburského vznikla univerzita."),
    "roman-numeral-before-an-agreeing-epithet-tagged-a-name": ("cs", None, "Karel IV. Lucemburský zemřel.",
                                                               "Karel čtvrtý Lucemburský zemřel."),
    "nearest-verb-governs": ("cs", None, "Dosáhl cíle a obsadil 2. místo, postavil se a zpíval mezi 2 stromy.",
                             "Dosáhl cíle a obsadil druhé místo, postavil se a zpíval mezi dvěma stromy."),
    "acronym-starts-sentence-in-mixed-text": ("cs", None, "Byl tam atd. USA zasáhly. Vládl Karel IV. NATO vzniklo později.",
                                              "Byl tam a tak dále. USA zasáhly. Vládl Karel čtvrtý. NATO vzniklo později."),
    "clitic-of-another-verb": ("cs", None, "Smál se a vzdal 2. kolo.", "Smál se a vzdal druhé kolo."),
    "capital-acronym-before-signed-amount": ("cs", None, "MAX. -5 KG, MAX. $5.",
                                             "Maximálně mínus pět kilogramů, maximálně pět dolarů."),
    "capital-acronym-before-approximate-amount": ("cs", None, "MAX. ≈5 KG, MAX. ±2 KG.",
                                                  "Maximálně přibližně pět kilogramů, maximálně plus minus dva kilogramy."),
    "capital-acronym-before-number-kept": ("cs", None, "TURNAJ ATP. 500 ZAČAL.", "TURNAJ ATP. pět set ZAČAL."),  # the tour
    "acronym-period-before-number": ("cs", None, "Turnaj ATP. 500 začal. Hrál na turnajích ATP. 500 lidí přišlo.",
                                     "Turnaj ATP. pět set začal. Hrál na turnajích ATP. Pět set lidí přišlo."),
    "acronym-period-before-number-across-a-line": ("cs", None, "Turnaj ATP.\n500 začal.", "Turnaj ATP.\npět set začal."),
    "capital-acronym-before-spaced-prefix": ("cs", None, "MAX. ≈ 5 KG, MAX. ± 2 KG, MAX. $ 5.",
                                             "Maximálně přibližně pět kilogramů, maximálně plus minus dva kilogramy, "
                                             "maximálně pět dolarů."),
    "vzdalit-is-not-vzdat": ("cs", None, "Vzdaluje se 2. kolo, vzdálilo se 3. kolo.",
                             "Vzdaluje se druhé kolo, vzdálilo se třetí kolo."),
    "conjunction-ends-the-verb-search": ("cs", None, "Cíle dosáhl a 2. místo nakonec obsadil.",
                                         "Cíle dosáhl a druhé místo nakonec obsadil."),
    "passive-is-a-place": ("cs", None, "Dům je postaven mezi 2 stromy.", "Dům je postaven mezi dvěma stromy."),
    "passive-of-placing": ("cs", None, "Dopis byl položen mezi 2 svazky. Obraz je pověšen nad 2 stoly. "
                                       "Mezi 2 svazky byl dopis položen.",
                           "Dopis byl položen mezi dva svazky. Obraz je pověšen nad dvěma stoly. "
                           "Mezi dva svazky byl dopis položen."),
    "verse-references-in-capitals": ("cs", {"verse_references": ["JAN"]}, "VIZ JAN 3,16. VÁHA 5 G.",
                                     "VIZ JAN tři, šestnáct. VÁHA pět gramů."),
    "english-phrase-with-hyphen-variant": ("cs", {"english": ["state\u2011of\u2011the\u2011art"]},
                                           "Je to state\u2011of\u2011the\u2011art.", "Je to <en>state-of-the-art</en>."),
    "sk-genitive-verb-unit-object": ("sk", None, "Zbavil sa 5 kg. Zbavila sa 3 kg. Chce sa zbaviť 5 kg. "
                                     "Týka sa to 5 %.",
                                     "Zbavil sa piatich kilogramov. Zbavila sa troch kilogramov. "
                                     "Chce sa zbaviť piatich kilogramov. Týka sa to piatich percent."),
    "genitive-verb-imperfective-imperative-and-infinitive": (
        "cs", None, "Zúčastňoval se 5 závodů. Neboj se 5 psů. Nemusíš se bát 5 psů. Všímal si 5 chyb.",
        "Zúčastňoval se pěti závodů. Neboj se pěti psů. Nemusíš se bát pěti psů. Všímal si pěti chyb."),
    "sk-genitive-verb-imperfective-imperative-and-infinitive": (
        "sk", None, "Zúčastňoval sa 5 pretekov. Neboj sa 5 psov. Nemusíš sa báť 5 psov.",
        "Zúčastňoval sa piatich pretekov. Neboj sa piatich psov. Nemusíš sa báť piatich psov."),
    "na-before-a-level-amount": (
        "cs", None, "Záleží na 5 %. Inflace zůstala na 5 %. Teplota se drží na 20 °C.",
        "Záleží na pěti procentech. Inflace zůstala na pěti procentech. Teplota se drží na dvaceti stupních Celsia."),
    "guard-na-before-a-target-or-a-span": (
        "cs", None, "Inflace vzrostla na 5 %. Zůstal na 2 h. Na 5 km² žije 100 lidí.",
        "Inflace vzrostla na pět procent. Zůstal na dvě hodiny. Na pěti kilometrech čtverečních žije sto lidí."),
    "po-before-an-amount-that-is-no-time": (
        "cs", None, "Po 5 km zastavil. Prodával je po 5 Kč.",
        "Po pěti kilometrech zastavil. Prodával je po pěti korunách."),
    "guard-po-before-a-time": ("cs", None, "Přijdu po 2 h.", "Přijdu po dvou hodinách."),
    "guard-po-until-an-amount": (
        "cs", None, "Voda stoupla až po 5 m. Šel od 2 km po 5 km.",
        "Voda stoupla až po pět metrů. Šel od dvou kilometrů po pět kilometrů."),
    "guard-time-subject-or-before-its-verb": (
        "cs", None, "Čekala 1 nebo 2 ženy. 1 h trvá 60 min.",
        "Čekala jedna nebo dvě ženy. Jedna hodina trvá šedesát minut."),
    "na-after-a-subject-noun-that-owns-the-amount": (
        "cs", None, "Zůstala mu faktura na 5 Kč. Zůstal mu dluh na 5 Kč. Zůstala objednávka na 5 ks.",
        "Zůstala mu faktura na pět korun. Zůstal mu dluh na pět korun. Zůstala objednávka na pět kusů."),
    "roman-numeral-before-the-verb-of-its-sentence": (
        "cs", None,
        "Za vlády Karla IV. Lucemburského země vzkvétala. Václavu II. Přemyslovci patřilo Polsko. "
        "Vládl Karel IV. Petr byl jeho synem.",
        "Za vlády Karla čtvrtého Lucemburského země vzkvétala. Václavu druhému Přemyslovci patřilo Polsko. "
        "Vládl Karel čtvrtý. Petr byl jeho synem."),
    "roman-numeral-before-an-epithet-ending-its-sentence": (
        "cs", None, "Patřilo to Václavu II. Přemyslovci.", "Patřilo to Václavu druhému Přemyslovci."),
    "sk-roman-numeral-before-a-new-sentence-led-by-a-name": (
        "sk", None, "Vládol Karol IV. Peter bol jeho synom.", "Vládol Karol štvrtý. Peter bol jeho synom."),
    "guard-roman-numeral-before-an-epithet-and-a-verbless-name": (
        "cs", None, "Jindřich VIII. Tudor zemřel. Zemřel Karel IV. Jeho syn vládl. Císař Karel V. Kniha o něm vyšla.",
        "Jindřich osmý Tudor zemřel. Zemřel Karel čtvrtý. Jeho syn vládl. Císař Karel pátý. Kniha o něm vyšla."),
    "guard-roman-numeral-after-an-oblique-name-before-a-sentence-of-its-own": (
        "cs", None, "Dal to Karlu IV. Země byla chudá. Poslal dopis Karlu IV. Praha.",
        "Dal to Karlu čtvrtému. Země byla chudá. Poslal dopis Karlu čtvrtému. Praha."),
    "guard-roman-numeral-before-a-subject-after-a-fronted-phrase": (
        "cs", None, "Za Karla IV. Velký požár vypukl v Praze. Za vlády Karla IV. Kniha vyšla.",
        "Za Karla čtvrtého. Velký požár vypukl v Praze. Za vlády Karla čtvrtého. Kniha vyšla."),
    "sk-unit-after-a-written-scale-word-in-cases": (
        "sk", None, "S 5 miliónmi € odišiel. O 5 miliónoch € sa nehovorí.",
        "S piatimi miliónmi eur odišiel. O piatich miliónoch eur sa nehovorí."),
    "sk-clock-span-to-an-afternoon-hour": (
        "sk", None, "Otvorené od 8 h do 17 h. Otvorené od 8 do 17 h. Po 18 h je zatvorené.",
        "Otvorené od ôsmej do sedemnástej. Otvorené od ôsmej do sedemnástej. Po osemnástej je zatvorené."),
    "sk-afternoon-hour-after-other-prepositions": (
        "sk", None, "Príde pred 20 h. Príde okolo 15 h. Príde k 17 h. Príde medzi 14 a 16 h.",
        "Príde pred dvadsiatou. Príde okolo pätnástej. Príde k sedemnástej. Príde medzi štrnástou a šestnástou."),
    "sk-afternoon-span-from-a-morning-hour": (
        "sk", None, "Príde medzi 9 a 17 h.", "Príde medzi deviatou a sedemnástou."),
    "sk-afternoon-span-in-capitals": ("sk", None, "PRÍDE OD 8 DO 17 H.", "PRÍDE OD ôsmej DO sedemnástej."),
    "dot-times-after-an-approximator": (
        "cs", None, "Přišel v 8.30, asi 9.30 a 10.30 hod.", "Přišel v osm třicet, asi devět třicet a deset třicet."),
    "sk-dot-times-after-an-approximator": (
        "sk", None, "Prídem o 8.30, asi 9.30 a 10.30 hod.",
        "Prídem o ôsmej tridsať, asi deviatej tridsať a desiatej tridsať."),
    "dot-time-before-a-bare-hour-with-hour-word": (
        "cs", None, "Otevřeno 8.30–17 h.", "Otevřeno osm třicet až sedmnáct hodin."),
    "dot-time-as-the-upper-end-of-a-dash-range": (
        "cs", None, "Otevřeno 8–17.30 hod.", "Otevřeno osm až sedmnáct třicet."),
    "colon-time-as-the-upper-end-of-a-dash-range": (
        "cs", None, "Otevřeno 8–17:30.", "Otevřeno osm až sedmnáct třicet."),
    "sk-colon-time-as-the-upper-end-of-a-dash-range": (
        "sk", None, "Otvorené 8–17:30.", "Otvorené osem až sedemnásť tridsať."),
    "sk-hours-in-a-list-after-o-each-logged": ("sk", None, "Prídem o 5 h a 6 h.", "Prídem o piatej a šiestej."),
    "sk-morning-span-before-an-afternoon-span": (
        "sk", None, "Otvorené od 9 do 12 h a od 13 do 17 h.",
        "Otvorené od deviatej do dvanástej a od trinástej do sedemnástej."),
    "sk-morning-dash-span-before-an-afternoon-span": (
        "sk", None, "Otvorené 8–12 h a 13–17 h.", "Otvorené od ôsmej do dvanástej a od trinástej do sedemnástej."),
    "sk-hours-of-a-feminine-genitive-tagged-locative": (
        "sk", None, "Po 20 h jazdy sme zastavili.", "Po dvadsiatich hodinách jazdy sme zastavili."),
    "sk-genitive-verb-after-a-time-genitive": (
        "sk", None, "Toho dňa sa bojí 5 psov.", "Toho dňa sa bojí piatich psov."),
    "cs-ordinal-hour-after-kolem": ("cs", None, "Přijdu kolem 13. h.", "Přijdu kolem třinácté hodiny."),
    "cs-ordinal-hour-abbreviation-before-a-word": (
        "cs", None, "Přijdu ve 13. hod. a odejdu.", "Přijdu ve třinácté hodině a odejdu."),
    "cs-ordinal-school-hour": ("cs", None, "V 5. h máme tělocvik.", "V páté hodině máme tělocvik."),
    "sk-ordinal-hour-after-okolo": ("sk", None, "Prídem okolo 13. h.", "Prídem okolo trinástej hodiny."),
    "sk-ordinal-hours-after-od-and-do": (
        "sk", None, "Od 8. h do 17. h je otvorené.", "Od ôsmej hodiny do sedemnástej hodiny je otvorené."),
    "cs-time-span-stays-until": ("cs", None, "Otevřeno 8:30–17:00.", "Otevřeno osm třicet až sedmnáct hodin."),
    "sk-time-span-after-o-stays-until": ("sk", None, "Prídem o 8:30–9:00.", "Prídem o ôsmej tridsať až deviatej."),
    "sk-two-hour-spans-before-denne": (
        "sk", None, "Otvorené od 9 do 12 h a od 13 do 17 h denne.",
        "Otvorené od deviatej do dvanástej a od trinástej do sedemnástej denne."),
    "sk-agreement-subject-of-a-neuter-verb-counts-persons-in-the-genitive": (
        "sk", {"num2words": {"construction": "agreement"}}, "Prišlo 5 mužov.", "Prišlo päť mužov."),
    "sk-agreement-subject-of-a-genitive-verb": (
        "sk", {"num2words": {"construction": "agreement"}}, "Zúčastnilo sa 5 mužov.", "Zúčastnilo sa päť mužov."),
    "guard-sk-agreement-object-agrees": (
        "sk", {"num2words": {"construction": "agreement"}}, "Videl 5 mužov.", "Videl piatich mužov."),
    "guard-sk-agreement-object-after-a-neuter-subject": (
        "sk", {"num2words": {"construction": "agreement"}}, "Dieťa videlo 5 mužov.", "Dieťa videlo piatich mužov."),
    "guard-sk-agreement-object-after-a-pronoun-subject": (
        "sk", {"num2words": {"construction": "agreement"}}, "To zrazilo 5 mužov.", "To zrazilo piatich mužov."),
    "guard-sk-agreement-object-before-a-subject-after-the-verb": (
        "sk", {"num2words": {"construction": "agreement"}}, "Včera videlo dieťa 5 mužov.",
        "Včera videlo dieťa piatich mužov."),
    "guard-sk-agreement-object-in-a-relative-clause": (
        "sk", {"num2words": {"construction": "agreement"}}, "Auto, ktoré zrazilo 5 mužov, ušlo.",
        "Auto, ktoré zrazilo piatich mužov, ušlo."),
    "sk-agreement-object-of-a-neuter-pronoun-subject": (
        "sk", {"num2words": {"construction": "agreement"}}, "Ono videlo 5 mužov.", "Ono videlo piatich mužov."),
    "cs-zero-before-counted-nouns-after-pred": (
        "cs", None, "Mluvili jsme před 0 body.", "Mluvili jsme před nula body."),
    "cs-zero-before-counted-nouns-after-mezi": ("cs", None, "Stál mezi 0 body.", "Stál mezi nula body."),
    "cs-zero-before-counted-nouns-after-nad": ("cs", None, "Stál nad 0 body.", "Stál nad nula body."),
    "guard-sk-zero-after-pred": ("sk", None, "Stál pred 0 bodmi.", "Stál pred nula bodmi."),
    "guard-sk-zero-after-medzi": ("sk", None, "Stál medzi 0 bodmi.", "Stál medzi nula bodmi."),
    "guard-cs-zero-opening-a-range-after-mezi": (
        "cs", None, "Stál mezi 0 a 5 body.", "Stál mezi nulou a pěti body."),
    "guard-cs-zero-after-a-verb-stays-declined": ("cs", None, "Dosáhl 0 bodů.", "Dosáhl nuly bodů."),
    "guard-cs-clock-zero-after-od": (
        "cs", None, "Pracuje od 0:00 do 6:00.", "Pracuje od nuly hodin do šesti hodin."),
    "cs-great-grandchildren-nominative-plural": ("cs", None, "Přišli 2 pravnuci.", "Přišli dva pravnuci."),
    "cs-great-great-grandchildren-nominative-plural": (
        "cs", None, "Přišli 2 prapravnuci.", "Přišli dva prapravnuci."),
    "guard-cs-great-grandchildren-accusative-plural": ("cs", None, "Viděl 2 pravnuky.", "Viděl dva pravnuky."),
    "sk-snimky-is-left-to-the-tagger": ("sk", None, "Má 2 snímky.", "Má dve snímky."),
    "roman-numbered-verse-books": (
        "cs", {"verse_references": ["Jan", "Kor", "Sam"]}, "Viz I. Jan 3,16, II. Kor 5,17 a II. Sam 7,12.",
        "Viz první Jan tři, šestnáct, druhý Kor pět, sedmnáct a druhá Sam sedm, dvanáct."),
    "roman-numeral-above-the-top-number-of-a-book-stays": (
        "cs", {"verse_references": ["Jan", "Kor", "Sam"]}, "Viz II. Kor 5,17 a III. Jan 1,1 a IV. Jan 3,16.",
        "Viz druhý Kor pět, sedmnáct a třetí Jan jedna, jedna a IV. Jan tři, šestnáct."),
    "sk-roman-numbered-verse-book": (
        "sk", {"verse_references": ["Ján"]}, "Pozri III. Ján 1,2.", "Pozri tretí Ján jeden, dva."),
    "guard-conjunction-before-a-verse-reference": (
        "cs", {"verse_references": ["Jan"]}, "I Jan 3,16 to potvrzuje.", "I Jan tři, šestnáct to potvrzuje."),
}
LOGGED = {  # id: (language, input, expected output, part of the WARNING)
    "fallback": ("cs", "Zbyl jen 1.", "Zbyl jen jeden.", "nominative masculine inanimate"),
    "count-ends-sentence-before-a-phrase": ("cs", "Bylo jich 5. Kniha o válce vyšla.",
                                            "Bylo jich pět. Kniha o válce vyšla.", "no governing noun"),
    "doubtful-tag": ("cs", "Vyšly 2. díly.", "Vyšly druhé díly.", "plural noun"),
    "adverb-before-noun": ("cs", "Vrátil 2 zpátky knihovně.", "Vrátil dva zpátky knihovně.",
                           "nominative masculine inanimate"),
    "decimal-comma-before-currency": ("cs", "Stálo to 1,234 USD.",
                                      "Stálo to jedna celá dvě stě třicet čtyři tisícin dolaru.", "not thousands"),
    "capital-acronym-kept": ("cs", "HRÁL ZA TJ. SOKOL.", "HRÁL ZA TJ. SOKOL.", "kept as written"),
    "capital-letter-glued-to-number": ("cs", "TELEFON PODPORUJE 5G.", "TELEFON PODPORUJE pět G.", "glued"),
    "comma-list-glued-to-word": ("cs", "Vyber 1,2,3G.", "Vyber jedna, dva, tři G.", "glued"),
    "range-glued-to-word": ("cs", "Váží 5–10G.", "Váží pět až deset G.", "glued"),
    "decimal-comma-before-noun": ("cs", "Přišlo 2,000 lidí.", "Přišlo dva lidí.", "not thousands"),
    "decimal-comma-before-scale": ("cs", "Stálo to 2,000 tis. Kč.", "Stálo to dva tisíce korun.", "not thousands"),
    "decimal-comma-in-unit-range": ("cs", "Ujel 1,234–2 km.", "Ujel jedna celá dvě stě třicet čtyři tisícin až dva kilometry.",
                                    "not thousands"),
    "decimal-comma-at-both-ends": ("cs", "Ujel 1,234 km–2,500 km.",
                                   "Ujel jedna celá dvě stě třicet čtyři tisícin kilometru až dvě celé pět desetin kilometru.",
                                   "not thousands"),
    # with no subject, a number after a present genitive verb is its object, but after "zúčastnit se" its subject
    "genitive-verb-present-tense": ("cs", "Dosáhne 5 bodů.", "Dosáhne pěti bodů.", "read as the object"),
    "sk-genitive-verb-present-tense": ("sk", "Bojí sa 5 psov.", "Bojí sa piatich psov.", "read as the object"),
    "genitive-verb-present-tense-concerning": ("cs", "Týká se 50 lidí.", "Týká se padesáti lidí.", "read as the object"),
    "genitive-verb-present-tense-taking-part": ("cs", "Zúčastní se 50 lidí.", "Zúčastní se padesát lidí.", "read as the subject"),
    "sk-genitive-verb-present-tense-taking-part": ("sk", "Zúčastní sa 50 ľudí.", "Zúčastní sa päťdesiat ľudí.",
                                                    "read as the subject"),
    # a verbal noun names an event, which is what one takes part in
    "genitive-verb-present-tense-taking-part-in-events": ("cs", "Zúčastní se 50 jednání.", "Zúčastní se padesáti jednání.",
                                                          "read as the object"),
    "genitive-verb-present-tense-taking-part-in-races": ("cs", "Zúčastní se 5 závodů.", "Zúčastní se pěti závodů.",
                                                         "read as the object"),
    "guard-genitive-verb-present-tense-taking-part-after-the-race": ("cs", "Závodu se zúčastní 5 jezdců.",
                                                                     "Závodu se zúčastní pět jezdců.",
                                                                     "read as the subject"),
    "genitive-verb-present-tense-taking-part-in-a-percent-of-votes": ("cs", "Zúčastní se 5 procent hlasování.",
                                                                      "Zúčastní se pěti procent hlasování.",
                                                                      "read as the object"),
    "guard-genitive-verb-present-tense-taking-part-with-to": ("cs", "Zúčastní se to 50 lidí.",
                                                              "Zúčastní se to padesát lidí.", "read as the subject"),
    "guard-genitive-verb-present-tense-to-after-a-question-word": ("cs", "Čeho se to bojí 5 psů?",
                                                                   "Čeho se to bojí pěti psů?", "read as the object"),
    "genitive-verb-present-tense-time": ("cs", "Bojí se 5 minut.", "Bojí se pěti minut.", "duration"),
    "genitive-verb-present-tense-after-conjunction": ("cs", "Dosáhne vítězství a 5 bodů.", "Dosáhne vítězství a pěti bodů.",
                                                      "subject or object"),
    # "akce" is also a genitive and "každý rok" an accusative, so neither is a subject before the number
    "genitive-verb-present-tense-after-object-and-time": ("cs", "Každý rok se akce zúčastní 50 lidí.",
                                                          "Každý rok se akce zúčastní padesát lidí.", "subject or object"),
    "guard-genitive-verb-present-tense-after-an-e-feminine": ("cs", "Inflace dosahuje 5 procent.",
                                                              "Inflace dosahuje pěti procent.", "read as the object"),
    "guard-genitive-verb-present-tense-after-a-night": ("cs", "Tu noc se bojí 5 dětí.", "Tu noc se bojí pěti dětí.",
                                                        "read as the object"),
    "guard-genitive-verb-present-tense-after-a-time": ("cs", "Příštího roku dosáhne 5 bodů.",
                                                       "Příštího roku dosáhne pěti bodů.", "read as the object"),
    # the subject of "přijde" is shared by "dosáhne", unless "a" opens a clause whose subject is the number
    "genitive-verb-present-tense-shared-subject": ("cs", "Petr přijde a dosáhne 5 bodů.", "Petr přijde a dosáhne pěti bodů.",
                                                   "shared across"),
    # a number before its verb opens a clause of its own, so "Petr" is not its verb's subject
    "genitive-verb-present-tense-number-first": ("cs", "Petr přišel a 5 mužů dosáhne cíle.",
                                                 "Petr přišel a pět mužů dosáhne cíle.", "subject or object"),
    "genitive-verb-present-tense-object-put-first": ("cs", "Úspěchu dosáhne 5 studentů.",
                                                     "Úspěchu dosáhne pět studentů.", "read as the subject"),
    "genitive-verb-present-tense-soft-genitive-put-first": ("cs", "Konce se dočká 5 lidí.",
                                                            "Konce se dočká pět lidí.", "read as the subject"),
    # a plural verb that agrees with the number is how people speak; a fronted object fits too
    "genitive-verb-colloquial-plural": ("cs", "5 mužů dosáhli cíle.", "Pět mužů dosáhli cíle.", "plural verb"),
    "sk-genitive-verb-number-before-plural-verb": ("sk", "5 žien sa báli psov.", "Päť žien sa báli psov.",
                                                    "subject of a plural verb, as people say"),
    "genitive-verb-number-before-a-neuter-plural-verb": ("cs", "5 procent voličů se zúčastnila voleb.",
                                                         "Pět procent voličů se zúčastnila voleb.",
                                                         "subject of a plural verb, as people say"),
    "unit-shared-without-a-preposition": ("cs", "Zbývá 1 nebo 2 h.", "Zbývá jedna nebo dvě hodiny.", "no preposition"),
    "unit-shared-past-an-approximator": ("cs", "Zbývá 1 nebo asi 2 h.", "Zbývá jedna nebo asi dvě hodiny.",
                                         "no preposition"),
    "capitals-ordinal-after-conjunction-before-a-plural": ("cs", "DOSÁHL CÍLE A 2. DÍLY VYŠLY.",
                                                           "DOSÁHL CÍLE A druhé DÍLY VYŠLY.", "plural noun"),
    "capitals-ordinal-after-az-before-a-time": ("cs", "BOJOVAL AŽ 3. DEN PADL.", "BOJOVAL AŽ třetí DEN PADL.",
                                                "ordinal after až"),
    "capitals-count-after-az-before-a-noun": ("cs", "PŘIŠLO JICH AŽ 5. MATKA ODEŠLA.", "PŘIŠLO JICH AŽ pět. MATKA ODEŠLA.",
                                              "no governing noun"),
    # a bare number after "v" is more often a clock time than an age ("Číst uměl v pěti")
    "bare-number-after-v": ("cs", "Přišel v 5.", "Přišel v pět.", "time (v pět) or age"),
    "guard-genitive-verb-children-of-a-masculine-verb": ("cs", "Děti se báli 5 psů.", "Děti se báli pěti psů.",
                                                         "plural verb"),
    "guard-genitive-verb-coordinated-subject": ("cs", "Petr a Eva se báli 5 psů.", "Petr a Eva se báli pěti psů.",
                                                "plural verb"),
    "genitive-verb-before-a-time": ("cs", "Dožil se 90 let.", "Dožil se devadesáti let.", "duration"),
    "sk-genitive-verb-time": ("sk", "Dožil sa 90 rokov.", "Dožil sa deväťdesiatich rokov.",
                              "a duration is the accusative"),
    "guard-genitive-verb-age-after-its-noun": ("cs", "Dosáhla věku 90 let.", "Dosáhla věku devadesáti let.", "duration"),
    "genitive-verb-age-after-its-noun-before-po": ("cs", "Dosáhl věku 90 let po operaci.",
                                                   "Dosáhl věku devadesáti let po operaci.", "duration"),
    # "vůdce" is a genitive as well, so its nominative tag cannot make the number a conjunct of the subject
    "genitive-verb-conjunct-or-object": ("cs", "Dočkali se vůdce a 5 rytířů.", "Dočkali se vůdce a pěti rytířů.",
                                         "conjunct of a nominative subject"),
    "genitive-verb-shared-unit-time": ("cs", "Týká se 2 nebo 3 h.", "Týká se dvou nebo tří hodin.",
                                       "a duration is the accusative"),
    "genitive-verb-two-ended-range": ("cs", "Dosáhne 30 °C–35 °C.",
                                      "Dosáhne třiceti stupňů Celsia až třiceti pěti stupňů Celsia.", "read as the object"),
    "genitive-verb-scale-amount": ("cs", "Dosáhne $1 mil.", "Dosáhne milionu dolarů.", "as 'milionu'"),
    "zero-after-an-la-verb": ("cs", "Dosáhla 0 bodů.", "Dosáhla nuly bodů.", "may agree with zero, read as the object"),
    "zero-events-after-an-la-verb": ("cs", "Zúčastnila se 0 závodů.", "Zúčastnila se nuly závodů.",
                                     "may agree with zero, read as the object"),
    "neuter-unit-of-events-after-an-la-verb": ("cs", "Zúčastnila se 2 % soutěží.", "Zúčastnila se dvou procent soutěží.",
                                               "may be neuter plural, read as the object"),
    "zero-subject-of-an-la-verb": ("cs", "Zúčastnila se 0 lidí.", "Zúčastnila se nula lidí.",
                                   "may agree with zero, read as the subject"),
    "genitive-verb-cents-range": ("cs", "Dosáhne 1,50–2,50 €.",
                                  "Dosáhne jednoho eura padesáti centů až dvou eur padesáti centů.", "read as the object"),
    "genitive-verb-neuter-plural-unit-subject": ("cs", "Voleb se zúčastnila 2 % voličů.",
                                                 "Voleb se zúčastnila dvě procenta voličů.",
                                                 "may be neuter plural, read as the subject"),
    "genitive-verb-neuter-plural-unit-object": ("cs", "Inflace dosáhla 3 %.", "Inflace dosáhla tří procent.",
                                                "may be neuter plural, read as the object"),
    "genitive-verb-unit-of-a-plural-verb": ("cs", "5 % studentů dosáhli cíle.", "Pět procent studentů dosáhli cíle.",
                                            "subject of a plural verb, as people say"),
    "genitive-verb-percent-of-a-plural-verb": ("cs", "5 procent studentů dosáhli cíle.",
                                               "Pět procent studentů dosáhli cíle.",
                                               "subject of a plural verb, as people say"),
    "genitive-verb-percent-sign-before-a-neuter-plural-verb": ("cs", "5 % voličů se zúčastnila voleb.",
                                                               "Pět procent voličů se zúčastnila voleb.",
                                                               "subject of a plural verb, as people say"),
    # a clause with a verb of its own does not share the one before "a"; the tagger's plural is logged
    "own-verb-after-conjunction": ("cs", "Dosáhl cíle a 2. díly vyšly.", "Dosáhl cíle a druhé díly vyšly.", "plural noun"),
    "roman-between-label-and-genitive": ("cs", "Vyšel díl V. knihy.", "Vyšel díl páté knihy.", "may number it"),
    "initial-or-ruler-after-a-name": ("cs", "Karel V. Kniha o něm vyšla.", "Karel V. Kniha o něm vyšla.",
                                      "initial or a ruler's number"),
    "initial-or-ruler-read-as-the-number": ("cs", "Karel V. byl zadržen.", "Karel pátý byl zadržen.",
                                            "initial or a ruler's number"),
    "wait-on-one-form-noun": ("cs", "Čekal na 2. náměstí.", "Čekal na druhé náměstí.", "locative"),
    "going-behind-or-after": ("cs", "Šel za 2 stromy a tam se zastavil.", "Šel za dvěma stromy a tam se zastavil.",
                              "place behind"),
    "comma-list-object-or-subject": ("cs", "Viděl 1,2,5 mužů.", "Viděl jeden, dva, pět mužů.", "accusative"),
    "sk-genitive-verb-shared-unit-time": ("sk", "Týka sa 2 alebo 3 h.", "Týka sa dvoch alebo troch hodín.",
                                          "a duration is the accusative"),
    "sk-genitive-verb-unit-time": ("sk", "Bál sa 5 min.", "Bál sa piatich minút.", "a duration is the accusative"),
    "sk-genitive-verb-neuter-plural-unit-subject": ("sk", "Volieb sa zúčastnili 2 % voličov.",
                                                    "Volieb sa zúčastnili dve percentá voličov.",
                                                    "may be neuter plural, read as the subject"),
    "sk-genitive-verb-unit-of-a-plural-verb-after": ("sk", "Zúčastnili sa 60 % voličov.",
                                                     "Zúčastnili sa šesťdesiatich percent voličov.",
                                                     "subject of a plural verb, as people say"),
    "sk-afternoon-dash-span": ("sk", "Otvorené 8–17 h.", "Otvorené od ôsmej do sedemnástej.", "hour of the day or a span"),
    "sk-afternoon-dash-span-after-od": ("sk", "Otvorené od 8–17 h.", "Otvorené od ôsmej do sedemnástej.",
                                        "hour of the day or a span"),
    "sk-genitive-verb-plural-verb-feminine-subject": ("sk", "Zúčastnili sa 5 žien.", "Zúčastnili sa piatich žien.",
                                                      "subject of a plural verb, as people say"),
    "genitive-verb-present-tense-taking-part-imperfective": ("cs", "Zúčastňuje se 50 lidí.",
                                                             "Zúčastňuje se padesát lidí.", "read as the subject"),
    "sk-afternoon-hour-after-do": ("sk", "Otvorené do 17 h.", "Otvorené do sedemnástej.", "read as the hour"),
    "sk-dot-time-before-do-an-afternoon-hour": (
        "sk", "Otvorené od 8.30 do 17 h.", "Otvorené od ôsmej tridsať do sedemnástej.", "read as the hour"),
    "sk-hour-after-a-dotted-time-before-do": (
        "sk", "Otvorené od 8.30 do 12 h.", "Otvorené od ôsmej tridsať do dvanástej.", "read as the hour"),
    "sk-hour-after-a-colon-time-before-do": (
        "sk", "Spal od 22:00 do 6 h.", "Spal od dvadsiatej druhej do šiestej.", "read as the hour"),
    "sk-hour-after-o-before-a-genitive-time-noun": (
        "sk", "Prídem o 5 h rána.", "Prídem o piatej rána.", "read as the hour"),
    "sk-afternoon-hour-after-po-before-a-genitive-time-noun": (
        "sk", "Prídem po 18 h večera.", "Prídem po osemnástej večera.", "read as the hour"),
    "sk-hour-after-o": ("sk", "Prídem o 5 h.", "Prídem o piatej.", "read as the hour"),
    "sk-hour-abbreviation-after-o": (
        "sk", "Stretnutie bude o 10 hod. v kancelárii.", "Stretnutie bude o desiatej v kancelárii.",
        "read as the hour"),
    "sk-hours-in-a-list-after-o": ("sk", "Prídem o 5 alebo 6 h.", "Prídem o piatej alebo šiestej.", "read as the hour"),
    "sk-hours-in-a-dash-span-after-o": ("sk", "Prídem o 5–6 h.", "Prídem o piatej až šiestej.", "read as the hour"),
    "sk-dot-time-and-an-hour-after-o": (
        "sk", "Prídem o 8.30 a 9 h.", "Prídem o ôsmej tridsať a deviatej.", "read as the hour"),
    "sk-clock-hour-before-an-adverb-tagged-locative": (
        "sk", "Pracoval po 14 h denne.", "Pracoval po štrnástej denne.", "read as the hour"),
    "sk-clock-hour-before-nonstop-tagged-locative": (
        "sk", "Pracoval po 14 h nonstop.", "Pracoval po štrnástej nonstop.", "read as the hour"),
    "sk-hour-before-dna-and-a-date": (
        "sk", "Prídem o 17 h dňa 5. júna.", "Prídem o sedemnástej dňa piateho júna.", "read as the hour"),
    "sk-hour-before-dopoludnia": (
        "sk", "Prídem o 10 h dopoludnia.", "Prídem o desiatej dopoludnia.", "read as the hour"),
    "sk-ordinal-hour-with-no-preposition": (
        "sk", "Začiatok: 13. h.", "Začiatok: trinásta hodina.", "no preposition; nominative"),
    "sk-time-span-without-od": (
        "sk", "Otvorené 8:30–17:00.", "Otvorené od ôsmej tridsať do sedemnástej.", "read as the hour"),
    "sk-dotted-time-and-hour-span-without-od": (
        "sk", "Otvorené 8.30–17 h.", "Otvorené od ôsmej tridsať do sedemnástej.", "read as the hour"),
    "sk-colon-time-and-hour-span-without-od": (
        "sk", "Otvorené 8:30–17 h.", "Otvorené od ôsmej tridsať do sedemnástej.", "read as the hour"),
    "sk-two-times-span-ending-before-six": (
        "sk", "Spal 22:00–6:00.", "Spal od dvadsiatej druhej do šiestej.", "read as the hour"),
    "sk-afternoon-span-without-od-after-a-morning-span": (
        "sk", "Otvorené od 9 do 12 a 13 do 17 h.", "Otvorené od deviatej do dvanástej a trinástej do sedemnástej.",
        "read as the hour"),
    "sk-spans-with-the-unit-once-at-the-end": (
        "sk", "Otvorené od 9 do 12 a od 13 do 17 h.",
        "Otvorené od deviatej do dvanástej a od trinástej do sedemnástej.", "read as the hour"),
    "guard-neuter-verb-of-time-passing-with-1-h-is-logged": (
        "cs", "Uplynulo 1 h.", "Uplynulo jedna hodina.", "no preposition; nominative"),
    "guard-neuter-verb-of-time-passing-and-a-list-is-logged": (
        "cs", "Uplynulo 1 nebo 2 h.", "Uplynulo jedna nebo dvě hodiny.", "no preposition; nominative"),
    "guard-neuter-verb-of-remaining-with-1-h-is-logged": (
        "cs", "Zbylo 1 h.", "Zbylo jedna hodina.", "no preposition; nominative"),
    "guard-masculine-verb-of-time-passing-with-1-h-is-logged": (
        "cs", "Uplynul 1 h.", "Uplynul jedna hodina.", "no preposition; nominative"),
    "guard-masculine-verb-of-remaining-with-1-h-is-logged": (
        "cs", "Zbyl 1 h.", "Zbyl jedna hodina.", "no preposition; nominative"),
    "guard-masculine-verb-of-ending-with-1-h-is-logged": (
        "cs", "Skončil 1 h.", "Skončil jedna hodina.", "no preposition; nominative"),
    "guard-sk-verb-of-time-passing-with-1-h-is-logged": (
        "sk", "Uplynulo 1 h.", "Uplynulo jedna hodina.", "no preposition; nominative"),
    "s-after-a-number-before-a-noun-is-seconds": (
        "cs", "Po 3 s ticha odešel.", "Po třech sekundách ticha odešel.", "read as seconds and not as the preposition"),
    "s-before-a-name-without-an-instrumental-ending-is-seconds-logged": (
        "cs", "Přišli 3 s Kate.", "Přišli tři sekundy Kate.", "read as seconds and not as the preposition"),
    "guard-s-before-an-instrumental-is-no-unit": (
        "cs", "Dostal 2 s plusem.", "Dostal dva s plusem.", "no governing noun"),
    "guard-s-before-a-dual-instrumental-is-no-unit": (
        "cs", "Šli 2 s dvěma psy.", "Šli dva s dvěma psy.", "no governing noun"),
    "sk-hour-span-before-denne-is-the-hour": (
        "sk", "Otvorené od 8 do 17 h denne.", "Otvorené od ôsmej do sedemnástej denne.", "read as the hour"),
    "sk-dash-span-before-denne-is-the-hour": (
        "sk", "Otvorené 8–17 h denne.", "Otvorené od ôsmej do sedemnástej denne.", "read as the hour"),
    "sk-o-after-a-verb-of-speaking-is-the-hour": (
        "sk", "Premiér bude hovoriť o 18 h.", "Premiér bude hovoriť o osemnástej.", "read as the hour"),
    "guard-sk-o-after-a-past-copula-and-no-word-of-change-is-the-hour": (
        "sk", "Stretnutie bolo o 5 h.", "Stretnutie bolo o piatej.", "read as the hour"),
    "cs-plural-instrumental-after-an-ordinal": (
        "cs", "Dotkl se 2. prsty stolu.", "Dotkl se druhými prsty stolu.", "ordinal agrees with a plural noun"),
    "cs-plural-subject-of-a-genitive-verb-after-an-ordinal": (
        "cs", "Soutěže se zúčastnily 2. ročníky.", "Soutěže se zúčastnily druhé ročníky.",
        "ordinal agrees with a plural noun"),
}
UNLOGGED = {  # id: (language, input, expected output); readings that need no review
    "sk-morning-dash-span": ("sk", "Otvorené 8–12 h.", "Otvorené osem až dvanásť hodín."),
    "sk-dash-span-of-work": ("sk", "Bolo to 8–17 h práce.", "Bolo to osem až sedemnásť hodín práce."),
    "duration-verb-with-a-neuter-subject": ("cs", "Jednání trvalo 1 nebo 2 hodiny.",
                                            "Jednání trvalo jednu nebo dvě hodiny."),
    "duration-verb-present-with-a-subject": ("cs", "Film trvá 1 nebo 2 hodiny.", "Film trvá jednu nebo dvě hodiny."),
    "na-enough-for-after-zustat-with-a-dative": ("cs", "Zůstalo mu jen na 5 Kč.", "Zůstalo mu jen na pět korun."),
    "sk-genitive-verb-plural-verb-plural-subject": ("sk", "Ženy sa báli 5 psov.", "Ženy sa báli piatich psov."),
    "sk-genitive-verb-plural-imperative": ("sk", "Nebojte sa 5 psov.", "Nebojte sa piatich psov."),
    "spaced-two-digit-year": ("cs", "Dne 5. 6. 24 v Praze.", "Dne pátého června dvacet čtyři v Praze."),
    "sk-spaced-two-digit-year": ("sk", "Dňa 5. 6. 24 v Prahe.", "Dňa piateho júna dvadsaťštyri v Prahe."),
    "genitive-verb-century": ("cs", "Dosáhli jsme XXI. století.", "Dosáhli jsme dvacátého prvního století."),
    "label-across-a-line": ("cs", "Viz č.\n5 a str.\n7.", "Viz číslo\npět a strana\nsedm."),
    "genitive-verb-first-person-auxiliary": ("cs", "Báli jsme se 5 psů.", "Báli jsme se pěti psů."),
    "genitive-verb-number-list-before-a-plural-verb": ("cs", "Petr a 5, 10 nebo 15 mužů dosáhli cíle.",
                                                       "Petr a pět, deset nebo patnáct mužů dosáhli cíle."),
    "genitive-verb-measure-list-subject": ("cs", "Voleb se zúčastnilo 50, 60 nebo 70 % voličů.",
                                           "Voleb se zúčastnilo padesát, šedesát nebo sedmdesát procent voličů."),
    "genitive-verb-present-tense-subject-to": ("cs", "Týká se to 5 lidí.", "Týká se to pěti lidí."),
    "sk-genitive-verb-present-tense-subject-to": ("sk", "Týka sa to 5 ľudí.", "Týka sa to piatich ľudí."),
    "guard-genitive-verb-to-after-a-question-word": ("cs", "Čeho se to bálo 5 psů?", "Čeho se to bálo pět psů?"),
    "genitive-verb-colloquial-plural-after": ("cs", "Cíle dosáhli 5 mužů.", "Cíle dosáhli pět mužů."),
    "genitive-verb-unit-subject": ("cs", "Voleb se zúčastnilo 60 % voličů.",
                                   "Voleb se zúčastnilo šedesát procent voličů."),
    "genitive-verb-unit-range-after-its-subject": ("cs", "Teplota dosáhne 30–35 °C.",
                                                   "Teplota dosáhne třiceti až třiceti pěti stupňů Celsia."),
    "genitive-verb-unit-first-person": ("cs", "Dosáhli jsme 30 °C.", "Dosáhli jsme třiceti stupňů Celsia."),
    "genitive-verb-percent-and-percent": ("cs", "Dosáhl 5 % a 10 %.", "Dosáhl pěti procent a deseti procent."),
    "guard-genitive-verb-percent-of-a-singular": ("cs", "Stroje dosáhly 5 procent výkonu.",
                                                  "Stroje dosáhly pěti procent výkonu."),
    "zero-subject-of-genitive-verb": ("cs", "Zúčastnilo se 0 lidí.", "Zúčastnilo se nula lidí."),
    "sk-genitive-verb-zero-subject": ("sk", "Zúčastnilo sa 0 ľudí.", "Zúčastnilo sa nula ľudí."),
    "genitive-verb-neuter-unit-after-a-first-person-la": ("cs", "Dosáhla jsem 3 %.", "Dosáhla jsem tří procent."),
    "guard-genitive-verb-object-put-before-a-feminine-verb": ("cs", "5 vítězství dosáhla už v prvním roce.",
                                                              "Pěti vítězství dosáhla už v prvním roce."),
    "guard-genitive-verb-number-after-a-feminine-verb": ("cs", "Dosáhla 5 procent.", "Dosáhla pěti procent."),
    "guard-genitive-verb-object-put-before-a-feminine-verb-and-a-time": ("cs", "5 vítězství dosáhla minulého roku.",
                                                                         "Pěti vítězství dosáhla minulého roku."),
    "guard-genitive-verb-percent-put-before-a-feminine-verb-and-a-time": ("cs", "5 procent dosáhla toho roku.",
                                                                          "Pěti procent dosáhla toho roku."),
    "genitive-verb-unit-after-a-feminine-subject": ("cs", "Strana dosáhla 4 %.", "Strana dosáhla čtyř procent."),
    "genitive-verb-feminine-subject-in-a-consonant": ("cs", "Nezaměstnanost dosahuje 5 procent. Daň se týká 5 lidí.",
                                                      "Nezaměstnanost dosahuje pěti procent. Daň se týká pěti lidí."),
    "guard-genitive-verb-loanword-subject-in-e": ("cs", "Google dosáhne 5 miliard uživatelů.",
                                                  "Google dosáhne pěti miliard uživatelů."),
    "sk-genitive-verb-unit": ("sk", "Teplota dosiahla 30 °C.", "Teplota dosiahla tridsať stupňov Celzia."),
    "sk-genitive-verb-object-of-a-subject": ("sk", "Zákon sa dotkne 50 ľudí. Eva sa bála 5 psov.",
                                             "Zákon sa dotkne päťdesiatich ľudí. Eva sa bála piatich psov."),
    "sk-genitive-verb-subject": ("sk", "Zúčastnilo sa 50 ľudí. Celé leto sa bálo 5 detí.",
                                 "Zúčastnilo sa päťdesiat ľudí. Celé leto sa bálo päť detí."),
    "sk-genitive-verb-invariable-numeral": ("sk", "Týka sa to 100 ľudí.", "Týka sa to sto ľudí."),
    "sk-accusative-verbs": ("sk", "Dosiahol 5 bodov. Všimol si 5 chýb.", "Dosiahol päť bodov. Všimol si päť chýb."),
    "sk-genitive-verb-genitive-object-put-first": ("sk", "Úspechu sa dočkal vodca a 5 mužov. Cieľa sa dočkalo 5 ľudí.",
                                                   "Úspechu sa dočkal vodca a päť mužov. Cieľa sa dočkalo päť ľudí."),
    "sk-neuter-in-i-after-an-accusative-verb": ("sk", "Mal 5 stretnutí. Dosiahol 5 vysvedčení.",
                                                "Mal päť stretnutí. Dosiahol päť vysvedčení."),
    "sk-ordinal-after-genitive-verb": ("sk", "Bál sa 3. vlny.", "Bál sa tretej vlny."),
    "sk-ordinal-genitive-after-a": ("sk", "Bál sa 2. dielu a 3. dielu.", "Bál sa druhého dielu a tretieho dielu."),
    "sk-ordinal-instrumental-after-genitive-verb": ("sk", "Dotkol sa 2. prstom stola.", "Dotkol sa druhým prstom stola."),
    "sk-ordinal-after-other-verbs": ("sk", "Dosiahli sme 21. storočie. Myslel na 2. diel.",
                                     "Dosiahli sme dvadsiate prvé storočie. Myslel na druhý diel."),
    "sk-number-sharing-a-subject-or-an-accusative": ("sk", "Bálo sa 2 alebo 5 detí. Vypil 2 alebo 5 pív.",
                                                     "Bálo sa dve alebo päť detí. Vypil dve alebo päť pív."),
    "number-after-v-before-a-day-time": ("cs", "Ve 2 ráno vstal a ve 3 v noci usnul. Vyhrál v 5 z 10 případů.",
                                         "Ve dvě ráno vstal a ve tři v noci usnul. Vyhrál v pěti z deseti případů."),
    "hour-unit-after-v-is-a-clock-time": ("cs", "Přijdu ve 2 h.", "Přijdu ve dvě hodiny."),
    "spaced-multiplication-sign": ("cs", "Spočítej 3x 4.", "Spočítej tři krát čtyři."),
    "approximately-between-labels": ("cs", "Platí 1≈2 a 2~1.", "Platí jedna přibližně dva a dva přibližně jedna."),
    "approximately-before-currency-operand": ("cs", "Platí 1≈$2.", "Platí jedna přibližně dva dolary."),
    "capitals-ordinal-after-az-without-a-verb": ("cs", "ZLEPŠENÍ NASTANE AŽ 3. DEN.", "ZLEPŠENÍ NASTANE AŽ třetí DEN."),
    "sk-genitive-verb-unit-subject-of-a-neuter-verb": ("sk", "Zúčastnilo sa 60 % voličov. Zúčastnilo sa 5 % voličov.",
                                                       "Zúčastnilo sa šesťdesiat percent voličov. "
                                                       "Zúčastnilo sa päť percent voličov."),
    "sk-tagger-gender-fix-points": ("sk", "Získal 2 body. Dosiahol 2 body. Chýbali mu 2 body.",
                                    "Získal dva body. Dosiahol dva body. Chýbali mu dva body."),
    "sk-multiplier-after-a-genitive-verb": ("sk", "Vzdal sa 5 krát. Dotkol sa jej 3 krát.",
                                            "Vzdal sa päť krát. Dotkol sa jej tri krát."),
    "genitive-verb-imperative-ordinal": ("cs", "Nebojte se 2. kola.", "Nebojte se druhého kola."),
    "guard-fighting-takes-no-genitive": ("cs", "Bojoval 5 let. Bojovali s 5 nepřáteli.",
                                         "Bojoval pět let. Bojovali s pěti nepřáteli."),
    "guard-taking-part-imperfective-number-sharing-a-subject": ("cs", "Zúčastňovalo se 2 nebo 50 lidí.",
                                                                "Zúčastňovalo se dva nebo padesát lidí."),
    "time-after-a-verb-of-duration": (
        "cs", "Trvá 1 nebo 2 hodiny. Čekala 1 nebo 2 hodiny. Trvá to 1 h.",
        "Trvá jednu nebo dvě hodiny. Čekala jednu nebo dvě hodiny. Trvá to jednu hodinu."),
    "unit-shared-after-a-verb-of-duration": ("cs", "Trvá to 1 nebo 2 h. Trvá to 1 nebo asi 2 h.",
                                             "Trvá to jednu nebo dvě hodiny. Trvá to jednu nebo asi dvě hodiny."),
    "sk-unit-after-a-written-scale-word": ("sk", "Mám 5 miliónov €, 2 milióny Kč a 5 miliárd $.",
                                           "Mám päť miliónov eur, dva milióny korún a päť miliárd dolárov."),
    "sk-one-oclock": ("sk", "Príde o 1:00. Od 1:00 do 2:00 spí. Pred 1:00 odišiel.",
                      "Príde o jednej. Od jednej do druhej spí. Pred jednou odišiel."),
    "guard-sk-afternoon-oclock": ("sk", "Prišiel o 13:00. Príde o 0:00.", "Prišiel o trinástej. Príde o nultej."),
    "guard-sk-hours-that-are-a-count": (
        "sk", "Od 1 do 2 h. Doručíme do 24 h. Po 15 h práce odišiel. Otvorené od 8 do 12 h.",
        "Od jednej do dvoch hodín. Doručíme do dvadsiatich štyroch hodín. Po pätnástich hodinách práce odišiel. "
        "Otvorené od ôsmich do dvanástich hodín."),
    "guard-sk-hours-before-an-adjective-and-noun-stay-counts": (
        "sk", "Batéria vydrží do 20 h nepretržitej prevádzky.",
        "Batéria vydrží do dvadsiatich hodín nepretržitej prevádzky."),
    "guard-sk-decimal-span-stays-a-count": (
        "sk", "Príde od 8,5 do 17 h.", "Príde od osem celých päť desatín do sedemnástich hodín."),
    "guard-sk-o-before-a-comparative-keeps-the-span": ("sk", "Prídem o 2 h neskôr.", "Prídem o dve hodiny neskôr."),
    "guard-sk-o-after-a-shift-verb-keeps-the-span": ("sk", "Posunuli to o 1 h.", "Posunuli to o jednu hodinu."),
    "guard-sk-o-after-a-verb-with-an-object-keeps-the-span": ("sk", "Prosil o 2 h.", "Prosil o dve hodiny."),
    "guard-sk-o-after-ide-keeps-the-span": ("sk", "Ide o 5 h.", "Ide o päť hodín."),
    "guard-sk-o-before-a-genitive-noun-keeps-the-count": ("sk", "Prídem o 5 h práce.", "Prídem o päť hodín práce."),
    "guard-sk-o-with-a-decimal-keeps-the-count": ("sk", "Prídem o 5,5 h.", "Prídem o päť celých päť desatín hodiny."),
    "guard-sk-o-after-25-keeps-the-count": ("sk", "Prídem o 25 h.", "Prídem o dvadsaťpäť hodín."),
    "sk-afternoon-hours-before-a-genitive-in-u-are-a-count": (
        "sk", "Po 18 h spánku sa zobudil.", "Po osemnástich hodinách spánku sa zobudil."),
    "guard-sk-o-with-minutes-keeps-the-span": ("sk", "Prídem o 2 h 30 min.", "Prídem o dve hodiny tridsať minút."),
    "guard-sk-o-with-and-minutes-keeps-the-span": (
        "sk", "Vrátim sa o 2 h a 30 min.", "Vrátim sa o dve hodiny a tridsať minút."),
    "guard-sk-o-with-and-a-half-keeps-the-span": ("sk", "Prídem o 2 h a pol.", "Prídem o dve hodiny a pol."),
    "cs-hours-of-operation-after-v-are-no-clock-time": (
        "cs", "Hotel je ve 24 h provozu.", "Hotel je ve dvaceti čtyřech hodinách provozu."),
    "guard-cs-clock-time-before-a-time-genitive": ("cs", "Přijdu v 8 h večera.", "Přijdu v osm hodin večera."),
    "guard-cs-clock-time-before-a-weekday-genitive": (
        "cs", "Sejdeme se ve 14 h pátku.", "Sejdeme se ve čtrnáct hodin pátku."),
    "sk-hour-before-do-and-a-time": ("sk", "Otvorené od 8 do 17:30.", "Otvorené od ôsmej do sedemnástej tridsať."),
    "sk-spans-with-the-long-hour-word-stay-counts": (
        "sk", "Otvorené od 8 do 12 a 13 do 17 hodín.",
        "Otvorené od ôsmich do dvanástich a trinástich do sedemnástich hodín."),
    "sk-spans-before-a-later-span-of-work-stay-counts": (
        "sk", "Pracoval od 9 do 12 h a od 13 do 17 h práce.",
        "Pracoval od deviatich do dvanástich hodín a od trinástich do sedemnástich hodín práce."),
    "sk-dash-spans-before-a-later-span-of-work-stay-counts": (
        "sk", "Pracoval 8–12 h a 13–17 h práce.", "Pracoval osem až dvanásť hodín a trinásť až sedemnásť hodín práce."),
    "sk-spans-with-the-unit-once-before-work-stay-counts": (
        "sk", "Pracoval od 9 do 12 a 13 do 17 h práce.",
        "Pracoval od deviatich do dvanástich a trinástich do sedemnástich hodín práce."),
    "po-before-a-range-end-is-the-accusative": ("cs", "Měří od 1 m po 5 m.", "Měří od jednoho metru po pět metrů."),
    "po-after-od-and-a-pronoun-is-each": ("cs", "Dostal od něj po 5 Kč.", "Dostal od něj po pěti korunách."),
    "po-with-minutes-is-after": ("cs", "Po 30 min se vrátil.", "Po třiceti minutách se vrátil."),
    "po-with-hours-is-after": ("cs", "Po 12 h nastal klid.", "Po dvanácti hodinách nastal klid."),
    "guard-po-until-a-length-is-the-accusative": ("cs", "Ujel až po 5 km.", "Ujel až po pět kilometrů."),
    "guard-po-until-a-time-is-after": ("cs", "Až po 2 h odešel.", "Až po dvou hodinách odešel."),
    "guard-po-between-two-percents-is-the-accusative": ("cs", "Od 5 % po 10 %.", "Od pěti procent po deset procent."),
    "na-after-dohodnout-se-is-the-locative": ("cs", "Dohodli se na 5 %.", "Dohodli se na pěti procentech."),
    "na-after-trvat-is-the-locative": ("cs", "Trvali na 5 kg.", "Trvali na pěti kilogramech."),
    "na-after-zit-is-the-locative": ("cs", "Žijí na 50 m².", "Žijí na padesáti metrech čtverečních."),
    "na-before-bydlet-is-the-locative": (
        "cs", "Na 50 m² bydlí 5 lidí.", "Na padesáti metrech čtverečních bydlí pět lidí."),
    "sk-na-after-zostat-is-the-locative": ("sk", "Zostala na 5 %.", "Zostala na piatich percentách."),
    "sk-na-after-zalezat-is-the-locative": ("sk", "Záleží na 2 h.", "Záleží na dvoch hodinách."),
    "sk-na-after-dohodnut-sa-is-the-locative": ("sk", "Dohodli sa na 5 %.", "Dohodli sa na piatich percentách."),
    "guard-na-after-dohodnout-a-thing-is-the-accusative": (
        "cs", "Dohodli cenu na 500 Kč.", "Dohodli cenu na pět set korun."),
    "guard-na-after-dohodnout-se-a-time-is-the-accusative": ("cs", "Dohodli se na 2 h.", "Dohodli se na dvě hodiny."),
    "guard-na-after-trvat-a-time-is-the-accusative": ("cs", "Trvalo to na 2 h.", "Trvalo to na dvě hodiny."),
    "guard-sk-na-after-zostat-a-time-is-the-accusative": ("sk", "Zostal tam na 2 h.", "Zostal tam na dve hodiny."),
    "o-after-a-noun-of-delay-is-the-difference": (
        "cs", "Zpoždění o 2 h se nikomu nelíbilo.", "Zpoždění o dvě hodiny se nikomu nelíbilo."),
    "o-before-a-reflexive-delay-verb-is-the-difference": ("cs", "O 2 h se zpozdil.", "O dvě hodiny se zpozdil."),
    "o-before-a-comparative-is-the-difference": ("cs", "Mluvil o 2 h déle.", "Mluvil o dvě hodiny déle."),
    "o-after-a-verb-of-speaking-is-the-topic": ("cs", "Mluvili o 5 %.", "Mluvili o pěti procentech."),
    "o-after-a-verb-of-negotiating-is-the-topic": ("cs", "Jednali o 5 kg.", "Jednali o pěti kilogramech."),
    "sk-o-after-a-noun-of-delay-is-the-difference": ("sk", "Meškanie o 2 h.", "Meškanie o dve hodiny."),
    "sk-o-after-a-noun-of-increase-is-the-difference": (
        "sk", "Zvýšenie o 5 % schválili.", "Zvýšenie o päť percent schválili."),
    "sk-o-after-informovat-is-the-topic": ("sk", "Informovali o 5 %.", "Informovali o piatich percentách."),
    "guard-o-after-jednat-se-is-the-accusative": ("cs", "Jednalo se o 5 %.", "Jednalo se o pět procent."),
    "guard-o-after-a-noun-before-a-topic-verb-is-no-topic": (
        "cs", "Informovali o růstu o 5 %.", "Informovali o růstu o pět procent."),
    "guard-o-after-a-verb-of-growth-is-the-accusative": ("cs", "Vzrostl o 5 %.", "Vzrostl o pět procent."),
    "guard-o-after-prosit-is-the-accusative": ("cs", "Prosil o 5 Kč.", "Prosil o pět korun."),
    "time-after-a-verb-with-its-own-subject-is-the-accusative": (
        "cs", "Mluvil 1 h a čekal 2 h.", "Mluvil jednu hodinu a čekal dvě hodiny."),
    "time-after-an-impersonal-verb-is-the-accusative": ("cs", "Pršelo 1 h.", "Pršelo jednu hodinu."),
    "sk-time-after-a-duration-verb-is-the-accusative": ("sk", "Trvalo to 1 h.", "Trvalo to jednu hodinu."),
    "s-after-a-number-before-a-verb-is-seconds": ("cs", "Po 10 s začalo pršet.", "Po deseti sekundách začalo pršet."),
    "s-after-a-number-before-a-conjunction-is-seconds": (
        "cs", "Počkej 10 s než se spustí.", "Počkej deset sekund než se spustí."),
    "s-after-na-and-a-number-is-seconds": ("cs", "Na 5 s zavřel oči.", "Na pět sekund zavřel oči."),
    "sek-is-seconds": ("cs", "Po 10 sek začalo pršet.", "Po deseti sekundách začalo pršet."),
    "sek-with-a-period-is-seconds": ("cs", "Trvalo to 10 sek.", "Trvalo to deset sekund."),
    "sk-s-after-a-number-before-a-verb-is-seconds": (
        "sk", "Po 10 s začalo pršať.", "Po desiatich sekundách začalo pršať."),
    "guard-sek-in-capitals-is-the-krona": ("cs", "Stálo to 5 SEK.", "Stálo to pět SEK."),
    "sk-po-with-seconds-is-after": ("sk", "Po 10 sek začalo pršať.", "Po desiatich sekundách začalo pršať."),
    "sk-po-with-minutes-is-after": ("sk", "Po 30 min sa vrátil.", "Po tridsiatich minútach sa vrátil."),
    "sk-po-before-a-range-end-is-the-accusative": (
        "sk", "Meral od 1 m po 5 m.", "Meral od jedného metra po päť metrov."),
    "sk-po-until-a-length-is-the-accusative": ("sk", "Došiel až po 5 km.", "Došiel až po päť kilometrov."),
    "guard-sk-po-with-hours-is-after": ("sk", "Po 2 h nastal pokoj.", "Po dvoch hodinách nastal pokoj."),
    "guard-sk-po-with-kilometres-is-after": ("sk", "Po 5 km odbočil.", "Po piatich kilometroch odbočil."),
    "sk-o-before-a-time-noun-keeps-the-span": ("sk", "Ušetril o 2 h času.", "Ušetril o dve hodiny času."),
    "sk-o-after-a-noun-of-delay-and-a-copula-keeps-the-span": (
        "sk", "Meškanie vlaku je o 2 h.", "Meškanie vlaku je o dve hodiny."),
    "sk-o-after-a-noun-of-delay-and-a-future-copula-keeps-the-span": (
        "sk", "Meškanie vlaku bude o 2 h.", "Meškanie vlaku bude o dve hodiny."),
    "sk-o-after-a-verb-of-growth-keeps-the-span": ("sk", "Vzrástol o 2 h.", "Vzrástol o dve hodiny."),
    "sk-o-after-a-noun-of-difference-and-a-copula-keeps-the-span": (
        "sk", "Rozdiel je o 2 h.", "Rozdiel je o dve hodiny."),
    "o-after-a-news-noun-and-a-copula-is-the-topic": ("cs", "Zpráva je o 5 %.", "Zpráva je o pěti procentech."),
    "sk-o-after-a-comparative-adverb-keeps-the-span": ("sk", "Bolo to dlhšie o 2 h.", "Bolo to dlhšie o dve hodiny."),
    "sk-o-after-a-comparative-adjective-keeps-the-span": ("sk", "Je dlhší o 2 h.", "Je dlhší o dve hodiny."),
    "sk-o-after-viac-keeps-the-span": ("sk", "Zostalo viac o 2 h.", "Zostalo viac o dve hodiny."),
    "guard-sk-o-after-a-comparative-and-a-count-keeps-the-accusative": (
        "sk", "Je starší o 5 rokov.", "Je starší o päť rokov."),
    "sk-po-after-a-verb-of-reaching-is-up-to": (
        "sk", "Voda siahala po 50 cm.", "Voda siahala po päťdesiat centimetrov."),
    "po-after-a-verb-of-reaching-is-up-to": ("cs", "Voda sahala po 50 cm.", "Voda sahala po padesát centimetrů."),
    "po-after-an-adjective-of-depth-is-up-to": (
        "cs", "Sníh byl hluboký po 20 cm.", "Sníh byl hluboký po dvacet centimetrů."),
    "guard-po-with-no-word-of-reaching-is-after": ("cs", "Po 5 km zastavil.", "Po pěti kilometrech zastavil."),
    "guard-sk-po-after-a-verb-is-each": ("sk", "Dostal po 5 kg.", "Dostal po piatich kilogramoch."),
    "guard-po-after-a-tall-subject-is-each": (
        "cs", "Vysoký muž dostal po 5 Kč.", "Vysoký muž dostal po pěti korunách."),
    "sk-o-after-a-noun-of-delay-and-a-past-copula-keeps-the-span": (
        "sk", "Meškanie vlaku bolo o 2 h.", "Meškanie vlaku bolo o dve hodiny."),
    "sk-personal-nominative-plural-from-five": ("sk", "Prišli 5 muži.", "Prišli piati muži."),
    "sk-personal-nominative-plural-over-twenty": ("sk", "Prišli 22 študenti.", "Prišli dvadsiati dvaja študenti."),
    "sk-personal-nominative-plural-in-a-list": (
        "sk", "Prišli 5 alebo 6 chlapci.", "Prišli piati alebo šiesti chlapci."),
    "sk-personal-nominative-plural-in-a-comma-list": ("sk", "Prišli 5,6,7 muži.", "Prišli piati, šiesti, siedmi muži."),
    "guard-sk-personal-genitive-plural-stays": ("sk", "Prišlo 5 mužov.", "Prišlo päť mužov."),
    "guard-sk-personal-nominative-plural-from-two-to-four": ("sk", "Prišli 3 muži.", "Prišli traja muži."),
    "guard-sk-animal-plural-counts-with-the-cardinal": ("sk", "Prišli 5 psi.", "Prišli päť psi."),
    "guard-sk-year-before-a-nominative-plural-keeps-the-cardinal": (
        "sk", "V roku 2020 muži vyhrali.", "V roku dvetisícdvadsať muži vyhrali."),
    "guard-sk-years-in-a-list-before-a-nominative-plural-keep-the-cardinal": (
        "sk", "V rokoch 2010 a 2011 študenti protestovali.",
        "V rokoch dvetisícdesať a dvetisícjedenásť študenti protestovali."),
    "sk-zero-subject-of-a-neuter-verb": ("sk", "Prišlo 0 ľudí.", "Prišlo nula ľudí."),
    "sk-zero-subject-of-a-copula": ("sk", "Bolo tam 0 ľudí.", "Bolo tam nula ľudí."),
    "guard-sk-zero-things-in-the-genitive": ("sk", "Mal 0 bodov.", "Mal nula bodov."),
    "cs-times-word-takes-the-masculine-count": ("cs", "Viděl ho 2 krát.", "Viděl ho dva krát."),
    "guard-cs-times-word-from-five": ("cs", "Viděl ho 5 krát.", "Viděl ho pět krát."),
    "guard-cs-times-glued-to-the-digit": ("cs", "Viděl ho 2krát.", "Viděl ho dvakrát."),
    "cs-scaled-zero-after-a-preposition": ("cs", "Za 0 tis. Kč.", "Za nula korun."),
    "cs-scaled-zero-of-millions-after-a-preposition": ("cs", "Za 0 mil. Kč.", "Za nula korun."),
    "guard-cs-scaled-amount-after-a-preposition": ("cs", "Za 5 tis. Kč.", "Za pět tisíc korun."),
    "guard-cs-scaled-zero-after-a-genitive-verb": ("cs", "Dosáhl 0 tis. bodů.", "Dosáhl nuly bodů."),
    "cs-great-grandson-is-masculine-animate": ("cs", "Má 1 pravnuka.", "Má jednoho pravnuka."),
    "cs-great-grandsons-are-masculine-animate": ("cs", "Má 2 pravnuky.", "Má dva pravnuky."),
    "cs-great-grandfather-is-masculine-animate": ("cs", "Má 1 pradědečka.", "Má jednoho pradědečka."),
    "sk-great-grandson-is-masculine-animate": ("sk", "Má 1 pravnuka.", "Má jedného pravnuka."),
    "guard-cs-great-grandsons-in-the-instrumental": ("cs", "S 2 pravnuky.", "Se dvěma pravnuky."),
    "guard-cs-great-grandsons-in-the-genitive-plural": ("cs", "Má 5 pravnuků.", "Má pět pravnuků."),
    "cs-genitive-verb-instrumental-ordinal": ("cs", "Dotkl se 2. prstem stolu.", "Dotkl se druhým prstem stolu."),
    "cs-genitive-verb-instrumental-ordinal-of-a-person": ("cs", "Bál se 2. mužem.", "Bál se druhým mužem."),
    "guard-cs-genitive-verb-genitive-ordinal": ("cs", "Dotkl se 2. prstu.", "Dotkl se druhého prstu."),
    "guard-cs-genitive-verb-dative-looking-ordinal": ("cs", "Bál se 2. dílu.", "Bál se druhého dílu."),
    "sk-times-word-before-an-adverb": ("sk", "Bol tam 2 krát týždenne.", "Bol tam dva krát týždenne."),
    "sk-times-word-before-a-comparative": ("sk", "Je to 3 krát lacnejšie.", "Je to tri krát lacnejšie."),
    "guard-sk-times-word-after-a-verb": ("sk", "Prišiel 2 krát.", "Prišiel dva krát."),
    "sk-great-grandchildren-share-the-genitive-verb": (
        "sk", "Dočkal sa 5 vnúčat a 3 pravnúčat.", "Dočkal sa piatich vnúčat a troch pravnúčat."),
    "cs-great-grandchildren-share-the-genitive-verb": (
        "cs", "Dočkal se 5 vnoučat a 3 pravnoučat.", "Dočkal se pěti vnoučat a tří pravnoučat."),
    "sk-great-grandchildren-after-a-verb": ("sk", "Má 5 pravnúčat.", "Má päť pravnúčat."),
    "cs-great-uncle-is-masculine-animate": ("cs", "Má 1 prastrýca.", "Má jednoho prastrýca."),
    "sk-great-uncle-after-a-verb-is-masculine-animate": ("sk", "Videl 1 prastrýca.", "Videl jedného prastrýca."),
    "guard-sk-great-uncle-is-read-right-already": ("sk", "Má 1 prastrýca.", "Má jedného prastrýca."),
    "sk-misgendered-masculine-plural": ("sk", "Má 2 hrady.", "Má dva hrady."),
    "sk-misgendered-masculine-plural-of-things": ("sk", "Kúpil 2 mobily.", "Kúpil dva mobily."),
    "sk-misgendered-feminine-plural": ("sk", "Má 2 zľavy.", "Má dve zľavy."),
    "sk-misgendered-feminine-plural-of-money": ("sk", "Dostal 2 miliardy dolárov.", "Dostal dve miliardy dolárov."),
    "guard-sk-masculine-plural-tagged-right": ("sk", "Má 2 stromy.", "Má dva stromy."),
    "guard-sk-feminine-plural-tagged-right": ("sk", "Má 2 ženy.", "Má dve ženy."),
    "guard-sk-genitive-plural-of-a-listed-word": ("sk", "Má 5 hradov.", "Má päť hradov."),
    "guard-sk-genitive-singular-after-one-keeps-its-gender": (
        "sk", "Cena 1 kocky je 5 eur.", "Cena jednej kocky je päť eur."),
    "guard-sk-genitive-singular-of-money-after-one": ("sk", "Hodnota 1 miliardy eur.", "Hodnota jednej miliardy eur."),
    "guard-sk-feminine-homograph-of-a-masculine-plural-is-left-alone": (
        "sk", "Cena 1 roboty je 5 eur.", "Cena jednej roboty je päť eur."),
    "cs-great-great-grandson-is-masculine-animate": ("cs", "Má 1 prapravnuka.", "Má jednoho prapravnuka."),
    "cs-great-great-grandsons-are-masculine-animate": ("cs", "Má 2 prapravnuky.", "Má dva prapravnuky."),
    "sk-great-great-grandson-is-masculine-animate": ("sk", "Má 1 prapravnuka.", "Má jedného prapravnuka."),
    "cs-instrumental-hand-after-an-ordinal": (
        "cs", "Chytil se 2. rukou za zábradlí.", "Chytil se druhou rukou za zábradlí."),
    "guard-cs-genitive-plural-hands-stay-genitive": ("cs", "Chytil se z obou rukou.", "Chytil se z obou rukou."),
    "guard-cs-counted-hands-after-a-preposition": ("cs", "Bez 2 rukou.", "Bez dvou rukou."),
    "guard-cs-genitive-singular-hand-after-an-ordinal": ("cs", "Dotkl se 2. ruky.", "Dotkl se druhé ruky."),
    "guard-cs-preposition-kolem-before-an-ordinal": ("cs", "Šel kolem 2. domu.", "Šel kolem druhého domu."),
    "guard-cs-indeclinable-show-after-a-preposition": ("cs", "Bez 1 šou to nejde.", "Bez jedné šou to nejde."),
    "guard-cs-indeclinable-show-after-a-verb": ("cs", "Moderoval 1 šou.", "Moderoval jednu šou."),
    "sk-zero-object-of-a-verb": ("sk", "Videl 0 mužov.", "Videl nula mužov."),
    "sk-zero-object-of-a-verb-with-a-subject-clitic": ("sk", "Pozvali sme 0 ľudí.", "Pozvali sme nula ľudí."),
    "guard-cs-zero-object-of-a-verb": ("cs", "Viděl 0 mužů.", "Viděl nula mužů."),
    "sk-zero-after-s-before-a-written-noun": (
        "sk", "Tím odišiel z turnaja s 0 bodmi.", "Tím odišiel z turnaja s nula bodmi."),
    "sk-zero-after-za-before-a-unit": ("sk", "Kúpil to za 0 eur.", "Kúpil to za nula eur."),
    "sk-zero-after-bez-before-a-written-noun": ("sk", "Odišiel bez 0 bodov.", "Odišiel bez nula bodov."),
    "sk-zero-after-o-before-a-written-noun": ("sk", "Hovorili o 0 bodoch.", "Hovorili o nula bodoch."),
    "sk-zero-after-k-before-a-written-noun": ("sk", "Teplota klesla k 0 stupňom.", "Teplota klesla k nula stupňom."),
    "sk-zero-after-na-before-a-unit": ("sk", "Teplota klesla na 0 °C.", "Teplota klesla na nula stupňov Celzia."),
    "cs-zero-after-bez-before-a-written-noun": ("cs", "Odešel bez 0 bodů.", "Odešel bez nula bodů."),
    "guard-sk-bare-zero-after-na-is-declined": ("sk", "Teplota klesla na 0.", "Teplota klesla na nulu."),
    "guard-sk-bare-zero-after-k-is-declined": ("sk", "Teplota klesla k 0.", "Teplota klesla k nule."),
    "guard-sk-bare-zero-after-s-is-declined": ("sk", "Skončili s 0.", "Skončili s nulou."),
    "guard-cs-bare-zero-after-na-is-declined": ("cs", "Teplota klesla na 0.", "Teplota klesla na nulu."),
    "guard-cs-bare-zero-after-k-is-declined": ("cs", "Teplota klesla k 0.", "Teplota klesla k nule."),
    "guard-cs-bare-zero-after-s-is-declined": ("cs", "Skončili s 0.", "Skončili s nulou."),
    "sk-zero-before-a-unit-after-a-verb": ("sk", "Teplota dosiahla 0 °C.", "Teplota dosiahla nula stupňov Celzia."),
    "sk-zero-hours-after-a-verb": ("sk", "Trvalo to 0 h.", "Trvalo to nula hodín."),
    "guard-sk-zero-sharing-its-noun-with-a-later-number-is-declined": (
        "sk", "Medzi 0 a 5 bodmi bol rozdiel.", "Medzi nulou a piatimi bodmi bol rozdiel."),
    "guard-cs-zero-opening-a-range-keeps-the-genitive": ("cs", "Od 0 do 5 bodů.", "Od nuly do pěti bodů."),
    "roman-numeral-before-an-epithet-with-more-after-it": (
        "cs", "Patřilo to Václavu II. Přemyslovci i jeho synovi.",
        "Patřilo to Václavu druhému Přemyslovci i jeho synovi."),
    "roman-numeral-before-an-epithet-after-a-coordinated-name": (
        "cs", "Dali to Janu a Václavu II. Přemyslovci.", "Dali to Janu a Václavu druhému Přemyslovci."),
    "roman-numeral-before-an-epithet-and-a-year": (
        "cs", "Dali to Janu Lucemburskému a Václavu II. Přemyslovci v roce 1300.",
        "Dali to Janu Lucemburskému a Václavu druhému Přemyslovci v roce tisíc tři sta."),
    "guard-roman-numeral-before-coordinated-subjects": (
        "cs", "Dal to Karlu IV. Dům a zahrada byly prázdné.", "Dal to Karlu čtvrtému. Dům a zahrada byly prázdné."),
    "guard-roman-numeral-before-the-subject-of-a-new-sentence": (
        "cs", "Dal to Karlu IV. Země byla chudá.", "Dal to Karlu čtvrtému. Země byla chudá."),
    "guard-roman-numeral-before-a-capitalised-adjective-and-a-verb": (
        "cs", "Za Karla IV. Velký požár vypukl v Praze.", "Za Karla čtvrtého. Velký požár vypukl v Praze."),
    "roman-numeral-before-a-new-sentence-led-by-the-same-name": (
        "cs", "Vzpomínal na Karla IV. Karel IV. byl velký panovník.",
        "Vzpomínal na Karla čtvrtého. Karel čtvrtý byl velký panovník."),
    "roman-numeral-before-a-new-sentence-led-by-a-longer-name": (
        "cs", "Znal Karla IV. Jan Lucemburský byl jeho otec.", "Znal Karla čtvrtého. Jan Lucemburský byl jeho otec."),
    "sk-roman-numeral-before-a-new-sentence-led-by-the-same-name": (
        "sk", "Spomínal na Karola IV. Karol IV. bol veľký panovník.",
        "Spomínal na Karola štvrtého. Karol štvrtý bol veľký panovník."),
    "guard-roman-numeral-before-an-oblique-name": (
        "cs", "Řekl to Karlu IV. Petrovi.", "Řekl to Karlu čtvrtému Petrovi."),
    "guard-roman-numeral-before-an-accusative-name": ("cs", "Viděl Karla IV. Jana.", "Viděl Karla čtvrtého Jana."),
    "guard-roman-numeral-before-an-oblique-surname-tagged-nominative": (
        "cs", "Navštívil Jindřicha VIII. Tudora.", "Navštívil Jindřicha osmého Tudora."),
    "guard-roman-numeral-before-an-oblique-surname-and-a-verb-after-a-conjunction": (
        "cs", "Navštívil Jindřicha VIII. Tudora a odešel.", "Navštívil Jindřicha osmého Tudora a odešel."),
    "guard-roman-numeral-before-an-oblique-surname-and-a-relative-clause": (
        "cs", "Navštívil Jindřicha VIII. Tudora, který ho přijal.",
        "Navštívil Jindřicha osmého Tudora, který ho přijal."),
}
HEADINGS = ("# Kapitola 5\n\nPetr koupil 5\njablek.\n\n\n# 2. Kapitola\n\nBylo 8:00.\n",
            "# Kapitola pět\n\nPetr koupil pět\njablek.\n\n\n# Druhá Kapitola\n\nBylo osm hodin.\n")
TEST2_INPUTS = [  # every input of the Test 2 block above
    ("cs", "Dne 1.1.2024 v 14:30 zaplatil 100 Kč, tj. cca 4 € (20 %)."), ("cs", "Mám 5 jablek."),
    ("cs", "Je mi 25 let."), ("sk", "Mám 25 rokov."), ("cs", "Šel s 5 přáteli."), ("cs", "1. ledna 2024"),
    ("cs", "15.3.2024"), ("cs", "Karel IV."), ("cs", "XXI. století"), ("cs", "III. díl"), ("cs", "např. toto"),
    ("sk", "atď."), ("cs", "A pak – nic..."), ("cs", "Máš 5 jablek?"), ("cs", ""), ("cs", "Toto je věta bez čísel."),
]
G2P = {"cs": CzechSlovakPhonemizer("cs"), "sk": CzechSlovakPhonemizer("sk")}


@pytest.mark.parametrize("language,config,text,expected", EXAMPLES.values(), ids=list(EXAMPLES))
def test_examples(language, config, text, expected):
    assert TextNormalizer(language, config).normalize(text) == expected


@pytest.mark.parametrize("language,text,expected,warning", LOGGED.values(), ids=list(LOGGED))
def test_reading_is_logged_for_review(caplog, language, text, expected, warning):
    with caplog.at_level(logging.WARNING, logger="src.text.normalizer"):
        assert TextNormalizer(language).normalize(text) == expected
    records = [r for r in caplog.records if r.name == "src.text.normalizer"]
    assert [r.levelno for r in records] == [logging.WARNING]
    assert warning in records[0].getMessage()


@pytest.mark.parametrize("language,text,expected", UNLOGGED.values(), ids=list(UNLOGGED))
def test_reading_is_not_logged(caplog, language, text, expected):
    with caplog.at_level(logging.WARNING, logger="src.text.normalizer"):
        assert TextNormalizer(language).normalize(text) == expected
    assert not [r for r in caplog.records if r.name == "src.text.normalizer"]


def test_line_structure_and_headings(cs):
    text, expected = HEADINGS
    assert cs.normalize(text) == expected


def test_long_list_of_noun_phrases_is_read_quickly(cs):  # the time grew with the cube of the items: 12 s for 150
    cs.normalize("Mám 5 jablek.")
    start = time.time()
    out = cs.normalize("Dosáhl " + ", ".join(f"{n} bodů" for n in range(5, 150)) + " nebo 150 bodů.")
    assert time.time() - start < 3
    assert out.startswith("Dosáhl pěti bodů, šesti bodů, sedmi bodů, ") and out.endswith(" nebo sto padesáti bodů.")


@pytest.mark.parametrize("text", ["Četl <en>Apollo 13</en>.", "Firma <en>R&D</en>.",
                                  "Četl <en>state\u2011of\u2011the\u2011art</en>."])  # spans stay as written
def test_digit_or_symbol_in_span_raises(cs, text):
    with pytest.raises(ValueError):
        cs.normalize(text)


@pytest.mark.parametrize("text", ["Cena v Kč/kg.", "Skóre #výhra.", "Cena 100 Kč / s DPH.", "Je to ≈ fajn."])
def test_symbol_without_reading_raises(cs, text):
    with pytest.raises(ValueError):
        cs.normalize(text)


def test_dates_need_no_tagger(monkeypatch):
    monkeypatch.setattr("src.text.normalizer._tagger", lambda language: pytest.fail("tagger loaded"))
    assert (TextNormalizer("cs").normalize("Dne 2024-01-15 a 1. 1.")
            == "Dne patnáctého ledna dva tisíce dvacet čtyři a prvního ledna.")


def test_verse_references_leave_other_forms_alone():  # "3,16a", "3,16–18a": read and logged as without the option
    for text in ("Viz Jan 3,16a.", "Viz Jan 3,16–18a."):
        assert TextNormalizer("cs", {"verse_references": ["Jan"]}).normalize(text) == TextNormalizer("cs").normalize(text)


def test_unknown_num2words_variant_raises():
    with pytest.raises(ValueError):
        TextNormalizer("sk", {"num2words": {"inverted": True}})  # a Czech-only keyword


@pytest.mark.parametrize("language,variants", [
    ("cs", {"construction": "agreemnt"}), ("cs", {"oblique_style": "ful"}), ("cs", {"ordinal_style": "Mixed"}),
    ("cs", {"inverted": "yes"}), ("cs", {"inverted": 1}), ("sk", {"construction": "inverted"}),
    ("sk", {"declined": "no"}), ("sk", {"codified": 0})])
def test_bad_num2words_variant_value_raises(language, variants):
    with pytest.raises(ValueError, match=next(iter(variants))):
        TextNormalizer(language, {"num2words": variants})


def test_every_num2words_variant_value_is_taken():
    for variants in ({"construction": "inverted", "oblique_style": "partial", "ordinal_style": "all", "inverted": True},
                     {"construction": "agreement", "oblique_style": "full", "inverted": False}):
        TextNormalizer("cs", {"num2words": variants})
    TextNormalizer("sk", {"num2words": {"construction": "agreement", "declined": False, "codified": True}})


def test_numbered_verse_book_raises():  # the digit of "1 Jan" would stay as written; "Jan" leaves it to be read
    with pytest.raises(ValueError, match="verse_references"):
        TextNormalizer("cs", {"verse_references": ["Jan", "1 Jan"]})


@pytest.mark.parametrize("config", [{"verse_references": "Jan"}, {"verse_references": True},
                                    {"verse_references": ["Jan", 3]}, {"english": "Harry Potter"}])
def test_book_config_options_must_be_lists_of_strings(config):
    with pytest.raises(ValueError, match="must be a list of strings"):
        TextNormalizer("cs", config)


# inputs that raise have no output to tokenize; the heading example keeps "# ", which the G2P rejects
@pytest.mark.parametrize("language,config,text", [(lang, None, text) for lang, text in TEST2_INPUTS]
                         + [example[:3] for example in EXAMPLES.values()]
                         + [(language, None, text) for language, text, _, _ in LOGGED.values()]
                         + [(language, None, text) for language, text, _ in UNLOGGED.values()])
def test_output_is_tokenizable(language, config, text):
    G2P[language].tokenize(TextNormalizer(language, config).normalize(text))
