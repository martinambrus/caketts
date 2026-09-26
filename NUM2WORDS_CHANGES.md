# num2words v2 for Slovak and Czech: what changed, why, and what to check

*25 September 2026. Covers `num2words_sk.py` and `num2words_cs.py`.*

Both modules were rewritten from published grammar rather than patched:

- **Slovak sources:** Morfológia slovenského jazyka (1966), the Pravidlá slovenského pravopisu (1991 and 1998, quoted by Jarošová 2021), Navrátil (2003), Encyclopaedia Beliana, and language columns by JÚĽŠ SAV linguists.
- **Czech sources:** the Internetová jazyková příručka of ÚJČ AV ČR (IJP), Czech Wikipedia, Nový encyklopedický slovník češtiny, Český rozhlas and the National Library's answer service.

Every expected value in the test suites is a form quoted from one of these sources, and each test names its source. Where the sources were silent or allowed variants, you decided in the review of 25 September (§4). There are 94 tests: 44 Slovak and 50 Czech.

The main problem in v1 was the same in both languages. Wherever a cardinal had to decline, v1 produced an **ordinal** form: 21 G "dvadsiatehojedného" / "dvacátéhojednoho", 100 G "stého", 1000 G "tisíceho" / "tisícího". Czech also had wrong spelling on top of that: words were run together ("dvoutisícdvacetčtyři" for 2024) and 2000 was "dvoutisíc".

To see the scale of the change, I ran v1 and v2 over the numbers from your notebook, in every gender and case:

- **Slovak:** 2,413 of 3,798 forms changed.
- **Czech:** 2,135 of 3,240 forms changed.

Every changed form is listed in `docs/NUM2WORDS_v1_v2_diff.csv`, tagged with the rule that changed it.

---

## 1. Check these first: forms you asked for in December that v2 reverses

In the December 2025 session you corrected the Slovak outputs for 1000 and for the millions. The forms you asked for there are **ordinal** forms, according to every Slovak source I could reach. v2 therefore goes back to the cardinal forms:

| number, case | v1 (your December request) | v2 | why |
|---|---|---|---|
| 1000 G / D, f. G | tisíceho / tisícemu / tisícej | **tisíc** (does not change) | "tisíc ľudí (ľuďom, ľuďoch, ľuďmi)": *tisíc* does not decline with a counted noun (Navrátil 2003; Beliana). *tisíceho* is the ordinal *tisíci*: "tisíceho návštevníka" = "the thousandth visitor". |
| 1 000 000 G / D / I | miliónteho / milióntemu / milióntym | **milióna / miliónu / miliónom** | *milión* is a noun declined like *dub* (Navrátil; Beliana). *miliónteho* is the ordinal *miliónty*, as in "26-miliónteho návštevníka". |
| 2 000 000 G | dvamiliónteho | **dvoch miliónov** | same rule |
| 15 000 000 G | pätnásťmiliónteho | **pätnástich miliónov** | same rule |
| 10⁹ G, 2·10⁹ G | miliardtého, dvemiliardtého | **miliardy, dvoch miliárd** | *miliarda* is a noun declined like *žena* |
| 2 000 000 N, 2·10⁹ N | dvamilióny, dvemiliardy | **dva milióny, dve miliardy** | Your plural forms and the feminine *dve* are kept. Only the spacing changes: "milióny a miliardy sa píšu vždy oddelene" (Jarošová 2021, fn. 18). |

Your December correction of *milióneho* was right: that form does not exist. But the cardinal it should have become is *milióna*, not *miliónteho*.

If your ear disagrees with the sources on any of these rows, tell me. Adding the old forms back as an option is a small change.

Slovak also has a rarer standard variant in which *tisíc* declines like *päť*: *s tisícimi ľuďmi*, *k tisícim ľuďom* (Navrátil). It is not implemented; say if you want it.

---

## 2. Slovak: what changed

| rule | forms changed | v1 → v2 | source |
|---|---:|---|---|
| *tisíc* and every *-tisíc* compound stay the same in every case | 600 | 1000 G tisíceho → tisíc; 2000 G dvetisíceho → dvetisíc | Navrátil: "Zložené číslovky s komponentom -tisíc sa neskloňujú" |
| *sto, dvesto … deväťsto* stay the same in every case and gender | 396 | 100 G stého → sto; 100 f. stá → sto; 200 n. dvesté → dvesto | Beliana: "Číslovky sto a tisíc v spojení s počítaným predmetom sa neskloňujú"; MSJ p. 317 |
| *milión, miliarda* are nouns: separate words, and they decline, also before a smaller part | 360 | 10⁶ G miliónteho → milióna; 2·10⁶ G dvamiliónteho → dvoch miliónov; 2 999 999 G → dvoch miliónov deväťstodeväťdesiatdeväťtisícdeväťstodeväťdesiatich deviatich | Navrátil; Beliana; Jarošová fn. 18; your review |
| spacing only (same sounds) | 309 | 202 G dvestodvoch → dvesto dvoch; ordinal 21 dvadsiatyprvý → dvadsiaty prvý | PSP 1991 p. 44 (declined units written apart); Beliana (ordinals 21–99 are two words) |
| numbers ending in *-jeden* never change, and use *jeden* for every gender | 289 | 21 G dvadsiatehojedného → dvadsaťjeden; 21 f. dvadsaťjedna → dvadsaťjeden | Páleníková: "21 žien = dvadsaťjeden žien"; MSJ p. 326 |
| compound ordinals keep cardinal *sto-/tisíc-* prefixes | 200 | 101. stýprvý → stoprvý; 2024. dvojtisícidvadsiatyštvrtý → dvetisícdvadsiaty štvrtý | PSP 1998 p. 24 ("stoprvý, dvetisícdruhý"); PSP 1991 p. 44 |
| 22–99 decline both parts as **cardinals** | 115 | 25 G dvadsiatehopiateho → dvadsiatich piatich | MSJ p. 326: "dvadsiatich dvoch žiakov, dvadsiatim dvom žiakom" |
| bare 200th, 2000th, 2 000 000th: the frequent *dvestý, dvetisíci* | 72 | 200. dvojstý → dvestý; 2000. dvojtisíci → dvetisíci | your review; Šrámeková 2023: more frequent than the codified *dvojstý, dvojtisíci* (`codified=True`) |
| masculine accusative is inanimate by default | 55 | 1 A jedného → jeden; 3 A troch → tri (the old forms: `animacy="animate"` / `"personal"`) | MSJ pp. 318–323 |
| *miliónty*, *miliardtý* ordinal paradigms | 10 | 10⁶. milióny → miliónty; G milióntého → miliónteho | rhythmic law; usage "miliónty návštevník" |
| *miliarda* accusative | 7 | miliarda → miliardu | declined like *žena* |

These changes fall outside the notebook grid:

- **22 feminine:** dvadsaťdve → dvadsaťdva. MSJ p. 326: "jeden a dva má pri všetkých podstatných menách mužský tvar".
- **1000th feminine / neuter:** tisícia / tisície → tisíca / tisíce (SSSJ, quoted by Šrámeková 2023).
- **0** now declines: nula, nuly, nule, nulu, nulou.
- **Decimals:** jeden celých päť → **jedna celá päť desatín**; tri celých štrnásť → **tri celé štrnásť stotín** (Duchková, JÚĽŠ).

---

## 3. Czech: what changed

| rule | forms changed | v1 → v2 | source |
|---|---:|---|---|
| spacing only (same sounds) | 475 | 29 dvacetdevět → dvacet devět; ordinal 21 dvacátýprvní → dvacátý první | IJP: "Vypisujeme-li číslovky slovy, píšeme každé slovo zvlášť" |
| *tisíc* declines as a noun | 412 | 1000 G tisícího → tisíce; I tisícím → tisícem | IJP dictionary: tisíc |
| *sto* declines as a noun, with no gender forms | 240 | 100 G stého → sta; 100 f. stá → sto; 200 G dvoustého → dvou set | IJP: sto (like *město*); before *tisíc* too: 100 000 G stotisícího → sta tisíc (your review) |
| 2000 is *dva tisíce*, 5000 *pět tisíc* | 178 | 2000 dvoutisíc → dva tisíce; 2024 dvoutisícdvacetčtyři → dva tisíce dvacet čtyři | IJP: "dva tisíce dvě stě třicet jedna" |
| compounds decline as **cardinals** | 168 | 25 G dvacátéhopátého → dvaceti pěti | IJP: "k dvaceti sedmi stupňům" |
| compound ordinals: leading hundreds and thousands stay cardinal, the rest is ordinal and declines; separate words | 180 | 101. G stýprvního → sto prvního; 2222. G dvoutisícídvoustýdvacátéhodruhého → dva tisíce dvě stě dvacátého druhého | your review (years); Chlumská quoting Sedláček: "sto první, sto padesátý osmý"; cs.wikipedia (Brus 1877): "roku tisíc osm set sedmdesátého pátého". `ordinal_style="all"` gives IJP's all-ordinal type, "tisící devítistý padesátý šestý" |
| numbers ending in 1: *dvacet jedna* in the nominative/accusative, *jednadvaceti* in other cases | 133 | 21 dvacetjeden → dvacet jedna; 21 G dvacátéhojednoho → jednadvaceti | IJP: "Tvary 2. p. jsou běžnější a přirozenější" |
| *milion*, *miliarda* are nouns | 132 | 10⁶ G miliontého → milionu; 2·10⁶ dvamiliony → dva miliony | IJP dictionary: milion (spelled *milion*, *milión* allowed) |
| long numbers in oblique cases (see §4) | 126 | 202 G dvěstědvou → dvou set dvou | IJP chapter 791 |
| masculine accusative is inanimate by default | 66 | 1 A jednoho → jeden (old form: `animacy="animate"`) | IJP dictionary: jeden |
| multiplied ordinals use the combining form | 18 | 10000. desettisící → desetitisící | cs.wikipedia (*desetimiliontý*) |
| tens + 2: *dvacet dva* for every gender | 4 | 202 f. dvěstědvě → dvě stě dva | IJP chapter 792; cs.wikipedia: "dvacet jedna/dva mužů, žen, dětí" |
| *miliarda* accusative | 3 | miliarda → miliardu | IJP dictionary: miliarda |

These changes fall outside the grid:

- **0** now declines like the Slovak one.
- **Decimals:** jeden celých pět → **jedna celá pět desetin**; tři celých čtrnáct → **tři celé čtrnáct setin**; nula celých → **nula celá** (IJP chapter 791).

---

## 4. Decisions made in your review (25 September 2026)

On these points the sources were silent or allowed more than one form. You decided each one, and the tests check every decision. No questions remain open.

**Slovak**

| question | decision | v2 output |
|---|---|---|
| 22 000 | masculine unit, as in 22 | **dvadsaťdvatisíc** |
| a bare 2 after *sto-*, *tisíc-*, *milión* | agrees like a simple *dve* | **stodve** (knihy), 2002 f. **dvetisícdve**, 1 000 002 f. **milión dve**; masculine **stodva**. After tens the masculine stays: 122 f. **stodvadsaťdva**. |
| *milióny / miliardy* after a hundreds number ending in a bare 2–4 | nominative plural, as in *stodve knihy* | **stodva milióny**, **stodve miliardy**. With 5 and up, and after tens, the genitive plural: *stopäť miliónov*, *dvadsaťdva miliárd*. |
| instrumental of 2 and 3 | depends on the noun's gender | **dvoma, troma** with masculine and neuter nouns (*dvoma stromami*, *dvoma mestami*, *troma mužmi*). **dvomi, tromi** with feminine nouns (*dvomi stenami*, *tromi ženami*), also inside compounds and before *miliarda*. |
| millions followed by a smaller part, in other cases | both parts decline | 2 000 005 G **dvoch miliónov piatich**; 2 500 000 D **dvom miliónom päťstotisíc** |
| 200th and 2000th | the frequent forms | **dvestý**, **dvetisíci**, likewise **tisícdvestý**, **dvemiliónty**, **dvemiliardtý**. `codified=True` gives *dvojstý*, *dvojtisíci*. |
| 300th, 400th, 3000th, 4000th | the attested forms | **trojstý**, **štvorstý**, **trojtisíci**, **štvortisíci** |
| other inferred ordinals | confirmed | **milión prvý**, **päťtisíci**, **dvadsaťjedentisíci** |
| spelling of declined units | the 1991 rules: units written apart | 202 G **dvesto dvoch** |
| decimals in other cases | always the nominative reading | G **dve celé päť desatín** |

**Czech**

| question | decision | v2 output |
|---|---|---|
| 101 and 1001 in other cases | *jedna* does not change | G **sto jedna**, **tisíce jedna** |
| ordinals, including years | the mixed type | **tisíc devět set devadesátý první**, G **tisíc osm set sedmdesátého pátého**, **dva tisíce dvacátý čtvrtý**, **sto první**, **tři sta šedesátý pátý**. A number of one part stays a single ordinal: **dvoutisící**. `ordinal_style="all"` gives *tisící devítistý devadesátý první*. |
| the agreement type (*dvacet jeden muž*) in other cases | not acceptable | `construction="agreement"` changes only the nominative and accusative; other cases use **jednadvaceti**, **dvou set jedna** |
| 21,3 | *celá* agrees with the last word | **dvacet jedna celá tři desetiny** |
| 100 000 in other cases | *sto* declines | G **sta tisíc**, D **stu tisícům**, I **stem tisíci**, L **stu tisících**; likewise G **sta milionů** |
| 21 000th | like cs.wikipedia's *pětadvacetitisící* | **jednadvacetitisící** |
| 10¹² | *bilion* | **bilion** |
| how much of a long number declines | everything up to two parts; only the last part from three parts up | G **dvou tisíc tří set**; G **tisíc osm set čtyřiceti sedmi**. `oblique_style` overrides. |
| decimals in other cases | always the nominative reading | G **dvě celé pět desetin** |

*stodve knihy* also shows how the counted noun behaves: after sto-/tisíc- plus a bare 2–4, it agrees as after a simple *dve*. The normalizer (plan Step 2.2) needs this rule, because num2words never outputs the noun.

---

## 5. API

v2 is compatible with v1. `num2words(n, to=..., gender=..., case=...)` works as before, and so does the old `animate=True` flag.

| argument | values (default first) | language | effect |
|---|---|---|---|
| `animacy` | `inanimate`, `animate`, `personal` | both | Masculine only. `animate` gives the accusative *jedného / jednoho* and *prvého / prvního*. `personal` (Slovak) also gives *dvaja, traja, štyria* and A = G. |
| `construction` | `genitive`, `agreement` (+ `inverted` in Czech) | both | Slovak `agreement`: *dvadsiati piati žiaci*, *stodvadsiati štyria*. Czech: `agreement` gives *dvacet jeden muž* and *dvacet dvě ženy* (nominative and accusative only); `inverted` gives *jednadvacet*. |
| `declined` | `True`, `False` | sk | `False` leaves every compound unchanged in every case: *s dvadsaťdva žiakmi*, *so stodva*. |
| `codified` | `False`, `True` | sk | the frequent *dvestý / dvetisíci* or the codified *dvojstý / dvojtisíci* |
| `oblique_style` | `auto`, `full`, `partial` | cs | how much of a long number declines (§4) |
| `ordinal_style` | `mixed`, `all` | cs | *sto první*, *tisíc devět set devadesátý první* or *stý první*, *tisící devítistý devadesátý první* |
| `inverted` | `False`, `True` | cs, ordinals | *jednadvacátý* |
| `plural` | `False`, `True` | both, ordinals | *prví / druzí*, *prvé / druhé* |

Decimals are accepted as a float, a `Decimal`, or a string with "," or "." (for example `"68,50"`). They are read with *celá / celé / celých* and a denominator word.

---

## 6. Cross-check against Unicode CLDR

`scripts/cldr_crosscheck.py` compares both modules with the CLDR spell-out rules:

- **Czech:** 721 of 4,590 comparable forms differ.
- **Slovak:** 1,154 of 4,446 comparable forms differ.

I reviewed every group of differences. **None of them is a v2 bug.** Either CLDR disagrees with the sources above, or the form is one of your review decisions (§4). CLDR is not a normative source; its Slovak data even writes 2000 as *dve tisíce*.

- **Czech:**
  - CLDR uses the agreement type in every case (*dvacet jeden*, *dvaceti jednoho*). Your review rejected the oblique forms, so `construction="agreement"` reproduces only *dvacet jeden*.
  - CLDR spells *milión*; IJP prefers *milion*.
  - CLDR declines *sto* before another numeral (*sta pěti*), where IJP chapter 792 keeps *sto*.
  - CLDR reads 0 as *žádný*.
  - Czech ordinals: all 1,782 forms that ICU 74 could render agree with v2. Another 1,008 use rule syntax that ICU 74 cannot render, so they were not compared.
- **Slovak:** CLDR has these forms, where the sources have the second one:
  - *dve tisíce* for dvetisíc
  - *tretom* for treťom
  - *tisícieho* for tisíceho (rhythmic law)
  - *milióntý* for miliónty
  - *stý prvý* for PSP's *stoprvý*
  - *tristý*, *tritisíci* for *trojstý*, *trojtisíci* (Šrámeková: the *tri-* forms are unattested)
  - *dvadsaťjedna* for Páleníková's *dvadsaťjeden*
  - *piatich tisíc* for Navrátil's invariable *päťtisíc*
  - *sto jedného* for *stojeden* (Navrátil's -jeden rule)
  - *dvoma*, *troma* with feminine nouns, where your review chose *dvomi*, *tromi*
  - A few cells where ICU renders CLDR's rule badly (120 G comes out as "sto ")

---

## 7. Files

| file | what it is |
|---|---|
| `src/text/num2words_sk.py`, `src/text/num2words_cs.py` | the v2 modules |
| `tests/test_num2words_sk.py`, `tests/test_num2words_cs.py` | 44 + 50 tests; run `uv run pytest tests/test_num2words_sk.py tests/test_num2words_cs.py` |
| `scripts/validate_all_sk.py`, `scripts/validate_all_cs.py` | print every form for your notebook's numbers. Extra rows (animate, personal, agreement, inverted, undeclined, codified, all-ordinal) appear only where they differ from the default, followed by a decimals section. |
| `docs/NUM2WORDS_v1_v2_diff.csv` | every v1 → v2 change, with its rule; filter the `change` column |
| `docs/research/num2words_spec_sk.md`, `docs/research/num2words_spec_cs.md` | the research notes behind the modules, with every quote and URL. They were written before the review, so §4 takes precedence. |
| the repository's first commit | the v1 modules, tests and validation scripts |

---

## 8. Sources

**Slovak**
- MSJ — Ružička (ed.), *Morfológia slovenského jazyka*, SAV 1966, pp. 300–327: https://www.juls.savba.sk/ediela/msj/pdf100/msj300-399.pdf
- NAV — Navrátil, "Neohybnosť v ohybných slovných druhoch", *Kultúra slova* 37 (2003) 6: https://www.juls.savba.sk/ediela/ks/2003/6/ks2003-6.html
- BEL — *Encyclopaedia Beliana*, entry "číslovky": https://beliana.sav.sk/heslo/cislovky
- JAR — Jarošová, "Zložené číslovky so zreteľom na písanie spolu a oddelene", *Slovenská reč* 86 (2021) 2. Quotes PSP 1991 p. 44 and PSP 1998 p. 24: https://www.sav.sk/journals/uploads/12101101zlozene-cislovky-so-zretelom-na-pisanie-spolu-a-oddelene.pdf
- SRA — Šrámeková, "K potenciálnej variantnosti číslovkových tvarov dvojstý, dvestý", *Kultúra slova* 57 (2023) 5. Quotes SSSJ: https://www.sav.sk/journals/uploads/11281051268-284-k-potencialnej-variantnosti-cislovkovych-tvarov-dvojsty-dvesty.pdf
- PAL — Páleníková (JÚĽŠ), "Číslovky dvadsaťjeden, tridsaťjeden, štyridsaťjeden": https://www.quark.sk/cislovky-dvadsatjeden-tridsatjeden-styridsatjeden/
- POV18 — Považaj (JÚĽŠ), on sedem/osem/sedemdesiat/osemdesiat: https://www.quark.sk/pravopis-rozlicnych-tvarov-cisloviek-sedem-osem-sedemdesiat-osemdesiat/
- DUCH — Duchková (JÚĽŠ), "Ako čítame a skloňujeme desatinné čísla": https://www.quark.sk/ako-citame-a-sklonujeme-desatinne-cisla/
- SNK11 — Slovenský národný korpus, quiz answers 2011: https://korpus.juls.savba.sk/promo(2f)NocVyskumnikov2011(2f)Kviz.html
- TASR — "Ako písať jednoslovné a viacslovné číslovky" (2018): https://www.teraz.sk/import/slovencina-jednoslovne-viacslovne-cislov/343961-clanok.html

**Czech**
- IJP chapters: https://prirucka.ujc.cas.cz/?id=670 (dva), ?id=671 (tři, čtyři), ?id=791 (spelling, long numbers, decimals), ?id=792 (the counted noun)
- IJP dictionary entries (sto, tisíc, milion, miliarda, nula, jeden, dva, tři, čtyři, pět, devět, dvacet): https://prirucka.ujc.cas.cz/?slovo=sto and so on
- cs.wikipedia: https://cs.wikipedia.org/wiki/České_číslovky and https://cs.wikipedia.org/wiki/Řadová_číslovka
- NESČ: https://www.czechency.org/slovnik/ČÍSLOVKA, …/ZÁKLADNÍ%20ČÍSLOVKA, …/ŘADOVÁ%20ČÍSLOVKA
- Český rozhlas: https://plzen.rozhlas.cz/sklonovani-cislovek-6799885 and https://informace.rozhlas.cz/sklonovani-cislovek-8146797
- Národní knihovna, *Ptejte se knihovny*: https://www.ptejteseknihovny.cz/dotazy/sklonovani-cislovek
- Chlumská, *Řadové číslovky v současné češtině* (FF UK 2009): https://dspace.cuni.cz/bitstream/handle/20.500.11956/30337/DPTX_2008_2_11210_0_128342_0_75052.pdf

**Unreachable, worth checking directly:**
- slovnik.juls.savba.sk (KSSJ, PSP dictionary, SSSJ); it blocks automated access.
- The *Naše řeč* archive.
- IJP's question database.
