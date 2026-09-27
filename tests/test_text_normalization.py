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
    "sk-preposition-before-time-range": ("sk", None, "Otvorené od 2:00–3:00.", "Otvorené od druhej až tretej."),
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
    "mixed-time-range-with-hour-word": ("cs", None, "Otevřeno 8.30–9:30 hod.", "Otevřeno osm třicet až devět třicet."),
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
    "capitals-page": ("cs", None, "NA STR. 45 SE PÍŠE.", "NA straně čtyřicet pět SE PÍŠE."),
    "capitals-units": ("cs", None, "CENA JE 5 KČ, JEL 50 KM/H A MĚŘÍ 5 CM.",
                       "CENA JE pět korun, JEL padesát kilometrů za hodinu A MĚŘÍ pět centimetrů."),
    "capitals-one-letter-units": ("cs", None, "VZDÁLENOST 100 M, VÁHA 5 G A MÁM 5 S SEBOU.",
                                  "VZDÁLENOST sto metrů, VÁHA pět gramů A MÁM pět S SEBOU."),
    "capitals-scale-and-currency": ("cs", None, "STÁLO TO 5 TIS. KČ.", "STÁLO TO pět tisíc korun."),
    "sk-capitals-currency": ("sk", None, "CENA JE 5 KČ.", "CENA JE päť korún."),
    "capital-acronym-before-number": ("cs", None, "MAX. 5 KG.", "Maximálně pět kilogramů."),
    "sk-capital-acronym-before-number": ("sk", None, "PODĽA § 5 ODS. 2 PLATÍ.",
                                         "PODĽA paragrafu päť odseku dva PLATÍ."),
    "capital-era-after-number": ("cs", None, "ROKU 300 N. L. VLÁDL.", "ROKU tři sta našeho letopočtu VLÁDL."),
    "invisible-characters": ("cs", None, "\ufeffRakousko\u2011Uhersko má 5\u201110 Kč, text\u00adový.",
                             "Rakousko-Uhersko má pět až deset korun, textový."),
    "approximately": ("cs", None, "Je to ~5 km, tedy ≈5 000 m.",
                      "Je to přibližně pět kilometrů, tedy přibližně pět tisíc metrů."),
    "sk-approximately": ("sk", None, "Je to ~5 km.", "Je to približne päť kilometrov."),
    "asterisk-times": ("cs", None, "Spočítej 3*4 a 5 * 6.", "Spočítej tři krát čtyři a pět krát šest."),
    "comma-list": ("cs", None, "Zvol 1,2,3 nebo 4,5,6.", "Zvol jedna, dva, tři nebo čtyři, pět, šest."),
    "verse-references": ("cs", {"verse_references": ["Jan", "Mt"]},
                         "Viz Jan 3,16, Mt 5,3–12, Mt 5,3–7,29 a Jan 3:16. Jan přišel v 5,5.",
                         "Viz Jan tři, šestnáct, Mt pět, tři až dvanáct, Mt pět, tři až sedm, dvacet devět a "
                         "Jan tři, šestnáct. Jan přišel v pět celých pět desetin."),
    "sk-verse-references": ("sk", {"verse_references": ["Ján"]}, "Pozri Ján 3,16.", "Pozri Ján tri, šestnásť."),
    "verse-references-off": ("cs", None, "Viz Jan 3,16.", "Viz Jan tři celé šestnáct setin."),  # "Jan" is a name too
    "per-second-before-word": ("cs", None, "Jel 5 m / s a pak šel, tok měl 5 l / s a víc.",
                               "Jel pět metrů za sekundu a pak šel, tok měl pět litrů za sekundu a víc."),
    "sk-per-second-before-word": ("sk", None, "Išiel 5 m / s a potom zastal.",
                                  "Išiel päť metrov za sekundu a potom zastal."),
    "unit-slash-is-no-preposition": ("cs", None, "Jel 5 m/s a 5 km/h.",
                                     "Jel pět metrů za sekundu a pět kilometrů za hodinu."),
    # mezi, nad, pod, před, za: a masculine plural in -y is the same in the accusative and the instrumental
    "between-place": ("cs", None, "Stál mezi 2 stromy.", "Stál mezi dvěma stromy."),
    "between-place-verb-after": ("cs", None, "Mezi 2 stromy stála lavička.", "Mezi dvěma stromy stála lavička."),
    "between-direction": ("cs", None, "Postavil se mezi 2 stromy.", "Postavil se mezi dva stromy."),
    "behind-place": ("cs", None, "Stál za 2 stoly.", "Stál za dvěma stoly."),
    "ago-and-in": ("cs", None, "Před 2 roky odjel, přijel před 2 týdny a za 2 roky se vrátí.",
                   "Před dvěma roky odjel, přijel před dvěma týdny a za dva roky se vrátí."),
    "threshold-after-motion": ("cs", None, "Teplota klesla pod 5 °C, dnes je pod 5 °C.",
                               "Teplota klesla pod pět stupňů Celsia, dnes je pod pěti stupni Celsia."),
    "genitive-verbs": ("cs", None, "Dosáhli 5. místa, bál se 2. dílu a vzdal se 2. kola.",
                       "Dosáhli pátého místa, bál se druhého dílu a vzdal se druhého kola."),
    "accusative-without-clitic": ("cs", None, "Vzdal 2. kolo.", "Vzdal druhé kolo."),
    "na-with-accusative-verb": ("cs", None, "Vzpomínal na XX. století.", "Vzpomínal na dvacáté století."),
    "ordinal-before-capitalised-noun-in-case": ("cs", None, "V 5. Symfonii zazněl sbor, o 5. Symfonii psal.",
                                                "V páté Symfonii zazněl sbor, o páté Symfonii psal."),
    "sk-ordinal-before-capitalised-noun-in-case": ("sk", None, "V 5. Symfónii zaznel zbor.",
                                                   "V piatej Symfónii zaznel zbor."),
    "number-ends-sentence-after-preposition": ("cs", None, "Čekal na 2. Vlak přijel.", "Čekal na dva. Vlak přijel."),
    "roman-numbers-a-thing": ("cs", None, "Vyšel díl V. Kniha byla úspěšná.", "Vyšel díl pátý. Kniha byla úspěšná."),
    "initial-after-a-person": ("cs", None, "Autor V. Kovář napsal knihu.", "Autor V. Kovář napsal knihu."),
    "nearest-verb-governs": ("cs", None, "Dosáhl cíle a obsadil 2. místo, postavil se a zpíval mezi 2 stromy.",
                             "Dosáhl cíle a obsadil druhé místo, postavil se a zpíval mezi dvěma stromy."),
    "acronym-starts-sentence-in-mixed-text": ("cs", None, "Byl tam atd. USA zasáhly. Vládl Karel IV. NATO vzniklo později.",
                                              "Byl tam a tak dále. USA zasáhly. Vládl Karel čtvrtý. NATO vzniklo později."),
}
LOGGED = {  # id: (language, input, expected output, part of the WARNING)
    "fallback": ("cs", "Zbyl jen 1.", "Zbyl jen jeden.", "nominative masculine inanimate"),
    "doubtful-tag": ("cs", "Vyšly 2. díly.", "Vyšly druhé díly.", "plural noun"),
    "adverb-before-noun": ("cs", "Vrátil 2 zpátky knihovně.", "Vrátil dva zpátky knihovně.",
                           "nominative masculine inanimate"),
    "decimal-comma-before-currency": ("cs", "Stálo to 1,234 USD.",
                                      "Stálo to jedna celá dvě stě třicet čtyři tisícin dolaru.", "not thousands"),
    "capital-acronym-kept": ("cs", "HRÁL ZA TJ. SOKOL.", "HRÁL ZA TJ. SOKOL.", "kept as written"),
    "decimal-comma-before-noun": ("cs", "Přišlo 2,000 lidí.", "Přišlo dva lidí.", "not thousands"),
    "roman-between-label-and-genitive": ("cs", "Vyšel díl V. knihy.", "Vyšel díl páté knihy.", "may number it"),
}
UNLOGGED = {  # id: (language, input, expected output); readings that need no review
    "spaced-two-digit-year": ("cs", "Dne 5. 6. 24 v Praze.", "Dne pátého června dvacet čtyři v Praze."),
    "sk-spaced-two-digit-year": ("sk", "Dňa 5. 6. 24 v Prahe.", "Dňa piateho júna dvadsaťštyri v Prahe."),
    "genitive-verb-century": ("cs", "Dosáhli jsme XXI. století.", "Dosáhli jsme dvacátého prvního století."),
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


@pytest.mark.parametrize("text", ["Četl <en>Apollo 13</en>.", "Firma <en>R&D</en>."])
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


def test_unknown_num2words_variant_raises():
    with pytest.raises(ValueError):
        TextNormalizer("sk", {"num2words": {"inverted": True}})  # a Czech-only keyword


# inputs that raise have no output to tokenize; the heading example keeps "# ", which the G2P rejects
@pytest.mark.parametrize("language,config,text", [(lang, None, text) for lang, text in TEST2_INPUTS]
                         + [example[:3] for example in EXAMPLES.values()]
                         + [(language, None, text) for language, text, _, _ in LOGGED.values()]
                         + [(language, None, text) for language, text, _ in UNLOGGED.values()])
def test_output_is_tokenizable(language, config, text):
    G2P[language].tokenize(TextNormalizer(language, config).normalize(text))
