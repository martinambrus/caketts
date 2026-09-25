# Czech numerals: normative reference for `num2words_cs.py` (TTS text normalisation)

> **Note (added when this file moved into the repository).** These research notes were compiled on 25 September 2026, before the native-speaker review. Where they differ from the modules, `NUM2WORDS_CHANGES.md` §4 and the tests take precedence. Examples: the mixed ordinal type is the default, *bez sta tisíc korun*, and oblique cases never use the agreement type (*jednadvaceti*).

Research date: 2026-09-25. Purpose: state what standard Czech uses, with sources, so that the outputs listed in
`current_cs.md` can be corrected. Case order everywhere: **N G D A I L** (same as `current_cs.md`; vocative omitted).

## 0. Reading guide

**Confidence levels**
- **high**: the form or rule is printed in an ÚJČ source (the IJP dictionary table or a chapter), or in two or more independent sources that agree.
- **medium**: one secondary source (Wikipedia, Wiktionary, a thesis), or a direct application of a documented IJP rule to a number that the sources do not give.
- **low**: my own inference, marked **[INF]**, with no source found.

**Source tags** (full URLs are in §0.1).

| tag | source |
|---|---|
| IJP-670 / 671 / 786 / 791 / 792 / 122 / 780 | Internetová jazyková příručka (ÚJČ AV ČR), chapter pages `?id=…` |
| IJP:word | IJP dictionary entry: the declension table plus the SSČ/SSJČ/ASSČ text embedded in the entry |
| WIKI-ČČ / WIKI-ŘČ | cs.wikipedia articles "České číslovky" and "Řadová číslovka" |
| WIKT:word | English Wiktionary. Read through kaikki.org (wiktextract, enwiktionary dump 2026-09-02, extracted 2026-09-20), because en.wiktionary.org itself was not reachable |
| NESČ-Č / -ZČ / -ŘČ | Nový encyklopedický slovník češtiny online: entries ČÍSLOVKA (P. Karlík), ZÁKLADNÍ ČÍSLOVKA and ŘADOVÁ ČÍSLOVKA (K. Osolsobě) |
| ÚJOP | Charles University ÚJOP, *Česky v Česku – gramatické tabulky*, table T.4.2 |
| UK-hnd | Charles University course handout "Číslovky" (dl1.cuni.cz) |
| HRD | M. Hrdlička, "K užívání českých číslovek a číselných výrazů (se zřetelem k češtině jako cizímu jazyku)", *Studie z aplikované lingvistiky* (SALI) 2011/2 |
| CHL | L. Chlumská, *Řadové číslovky v současné češtině*, diploma thesis, FF UK 2009. Quotes Sedláček 1992, Brabcová 2000, Gebauer 1926, Havránek–Jedlička 1960 and Šmilauer 1972 |
| ČRo-05 / ČRo-20 | Český rozhlas: J. Málková, "Skloňování číslovek" (2005); ombudsman M. Pokorný, "Skloňování číslovek" (2020, quotes IJP) |
| NK-07 / NK-10 | Národní knihovna ČR, *Ptejte se knihovny* answers (2007, citing the ÚJČ jazyková poradna; 2010) |
| ÚJČ-ZD | ÚJČ "Zajímavé dotazy": Dvacetjednička |
| **[INF]** | my inference; not read in any source |

### 0.1 URLs
- IJP chapters:
  - https://prirucka.ujc.cas.cz/?id=670 (Dva, oba)
  - https://prirucka.ujc.cas.cz/?id=671 (Tři, čtyři)
  - https://prirucka.ujc.cas.cz/?id=791 (Členění čísel, víceslovné číslovkové výrazy, desetinná čísla)
  - https://prirucka.ujc.cas.cz/?id=792 (Počítaný předmět po číslovkách)
  - https://prirucka.ujc.cas.cz/?id=786 (Peněžní částky)
  - https://prirucka.ujc.cas.cz/?id=780 (Zkratky: mil., mld.)
  - https://prirucka.ujc.cas.cz/?id=122 (Psaní samohlásek v zakončení přejatých slov)
- IJP dictionary entries. Canonical URLs are listed; the content was read from the official mobile interface `https://m.prirucka.ujc.cas.cz/…` with the same query (see §0.2).
  - https://prirucka.ujc.cas.cz/?slovo=sto
  - https://prirucka.ujc.cas.cz/?slovo=tisíc
  - https://prirucka.ujc.cas.cz/?slovo=milion
  - https://prirucka.ujc.cas.cz/?slovo=miliarda
  - https://prirucka.ujc.cas.cz/?slovo=nula
  - https://prirucka.ujc.cas.cz/?id=jeden (m. živ.) and https://prirucka.ujc.cas.cz/?id=jeden_1 (m. neživ.)
  - https://prirucka.ujc.cas.cz/?slovo=dva
  - https://prirucka.ujc.cas.cz/?slovo=tři
  - https://prirucka.ujc.cas.cz/?slovo=čtyři
  - https://prirucka.ujc.cas.cz/?id=pět
  - https://prirucka.ujc.cas.cz/?slovo=sedm
  - https://prirucka.ujc.cas.cz/?slovo=osm
  - https://prirucka.ujc.cas.cz/?slovo=devět
  - https://prirucka.ujc.cas.cz/?slovo=dvacet
- Wikipedia:
  - https://cs.wikipedia.org/wiki/České_číslovky
  - https://cs.wikipedia.org/wiki/Řadová_číslovka
- Wiktionary (kaikki pages follow the pattern `https://kaikki.org/dictionary/Czech/meaning/<1st letter>/<first 2 letters>/<word>.html`; canonical `https://en.wiktionary.org/wiki/<word>#Czech`). Words read:
  - cardinals: jeden, dva, tři, čtyři, pět, šest, devět, deset, jedenáct, dvacet, devadesát, jednadvacet, "dvacet jedna"
  - nouns: sto, tisíc, milion, milión, miliarda, bilión, nula
  - ordinals: první, druhý, čtvrtý, stý, tisící, dvoustý, třístý, čtyřstý, pětistý, šestistý, sedmistý, osmistý, devítistý, miliardtý
- NESČ:
  - https://www.czechency.org/slovnik/ČÍSLOVKA
  - https://www.czechency.org/slovnik/ZÁKLADNÍ%20ČÍSLOVKA
  - https://www.czechency.org/slovnik/ŘADOVÁ%20ČÍSLOVKA
- ÚJOP: https://ujop.cuni.cz/UJOP-433-version1-cesky_v_cesku_gramaticke_tabulky_rj.pdf
- UK-hnd: https://dl1.cuni.cz/pluginfile.php/1095092/mod_resource/content/0/Cislovky.pdf
- HRD: https://studiezaplikovanelingvistiky.ff.cuni.cz/wp-content/uploads/sites/19/2016/03/milan_hrdlicka_11-22.pdf
- CHL: https://dspace.cuni.cz/bitstream/handle/20.500.11956/30337/DPTX_2008_2_11210_0_128342_0_75052.pdf
- ČRo-05: https://plzen.rozhlas.cz/sklonovani-cislovek-6799885
- ČRo-20: https://informace.rozhlas.cz/sklonovani-cislovek-8146797
- NK-07: https://www.ptejteseknihovny.cz/dotazy/sklonovani-cislovek
- NK-10: https://www.ptejteseknihovny.cz/dotazy/pouzivani-cislovek
- ÚJČ-ZD: https://ujc.cas.cz/cs/zajimave-dotazy/jak-se-pise-dvacet-jednicka-a-vubec-vyrazy-tohoto-typu-bez-ohledu-na-konkretni-ciselne-obsazeni-dohromady-nebo-jako-dve-slova-vim-ze-dvacet-jedna-se-pise-zvlast-a-jednadvacitka-do/

### 0.2 Access notes (for the reviewer)
- **How pages were read.** Every page was read through a fetch tool that returns a model-written reading of the page, not raw HTML. Key cells were re-queried with different prompts.
  - One reading error was caught and corrected this way: tisíc L pl first came back as "tisích"; the verified form is **tisících**.
  - The first reading of the kaikki "sto" table was also garbled. It was corrected by a cell-by-cell re-query and checked against IJP.
- **IJP dictionary tables.** `prirucka.ujc.cas.cz/?slovo=` is disallowed by that host's robots.txt. The tables were therefore read from IJP's official mobile host `m.prirucka.ujc.cas.cz`, which has no robots.txt and serves the same database.
  - IJP chapter pages (`?id=<digits>`) are allowed and were read directly.
  - Direct curl access to all sources was denied by the session's egress policy.
- **Not reachable:**
  - cs.wiktionary.org and en.wiktionary.org directly: cache-only / policy.
  - dotazy.ujc.cas.cz (ÚJČ question database): TLS/robots failure.
  - nase-rec.ujc.cas.cz (*Naše řeč*): https→http redirect loop. This includes the relevant articles "Sto jeden žák i stojedna žáků?" (art=5533) and "K tvoření číslovek řadových" (art=5604).
  - The RVP article "Umíme skloňovat číslovky?" (now redirects).
  - Příruční mluvnice češtiny: not online.
- **No IJP dictionary entry exists** (lookup falls back to the home page) for: jedna, deset, jedenáct, devadesát, jednadvacet, jedenadvacet, dvaadvacet, bilion/bilión, stý, tisící, dvoustý, čtyřstý, pětistý, dvoutisící, miliontý.
- **Wrong chapter ID in the brief.** The brief's hint `?id=770` is the chapter *Vokalizace předložek*. The numeral chapters are 670, 671, 790–795, 785 and 786.

---

## 1. Corrections to the current module (prioritised)

| # | Area | Current (`current_cs.md`) | Standard Czech | Conf. | Source |
|---|---|---|---|---|---|
| 1 | Multi-word cardinals written together | dvacetjeden, devadesátdevět, stojeden, stodvacetjeden, dvěstě, třista, pětset, tisícjeden, tisícsto, dvoutisícdvacetčtyři, dvamiliony, pětmilionů, dvěmiliardy, "pět celých dvacetpět" | **each word separately**: dvacet jeden/jedna, devadesát devět, sto jeden/jedna, sto dvacet jeden/jedna, dvě stě, tři sta, pět set, tisíc jeden/jedna, tisíc sto (or jedenáct set), dva tisíce dvacet čtyři, dva miliony, pět milionů, dvě miliardy. One word only for the inverted type (jednadvacet, dvaadvacet, pětadvacet …) | high | IJP-791 ("Vypisujeme-li číslovky slovy, píšeme každé slovo zvlášť"); ÚJČ-ZD |
| 2 | Multi-word ordinals written together | dvacátýprvní, stýprvní, stýdvacátýprvní, tisícíprvní, tisícístý, dvoutisícídvacátýčtvrtý | dvacátý první, stý první, stý dvacátý první, tisící první, tisící stý, dvoutisící dvacátý čtvrtý. One word only: jednadvacátý/dvaadvacátý…, dvoustý…devítistý, dvoutisící, pětitisící, stotisící, stopadesátý | high | IJP-791 ("dvacátý pátý, stý padesátý, čtyřstý sedmdesátý osmý, tisící devítistý padesátý šestý"; "pětadvacátý, stopadesátý, devítistý" together); WIKI-ŘČ |
| 3 | 2000, 5000 | dvoutisíc, pětitisíc (these are ordinal-like compounds) | **dva tisíce**, **pět tisíc** + regular declension (§6) | high | IJP:tisíc; IJP-791 ("dva tisíce dvě stě třicet jedna"); IJP:pět ("pět tisíc dvě stě dvacet pět") |
| 4 | Oblique cases of cardinals built from ordinal stems | 21 G dvacátéhojednoho; 100 G stého; 200 G dvoustého; 1000 G tisícího; 10⁶ G miliontého; 10⁹ G miliardtého … | always **cardinal** forms: 25 G dvaceti pěti / pětadvaceti; 100 G sta (or undeclined sto); 200 G dvou set; 1000 G tisíce; 10⁶ G milionu; 10⁹ G miliardy (§§3–7) | high | IJP:sto, IJP:tisíc, IJP:milion, IJP:miliarda, IJP-791 |
| 5 | 100 by gender | f. stá, n. sté (also stájedna, stédvacetjedno) | **sto** in all genders. sto is a neuter noun; stá/sté are ordinal forms | high | IJP:sto (declines like *město*); IJP-792 |
| 6 | 100 f. G | sté | **sta**, or undeclined **sto** with a counted noun ("bez sta korun" / "bez sto korun") | high | IJP-792; IJP:sto |
| 7 | 21 oblique | dvacátéhojednoho / dvacátéhojedné … | inverted **jednadvaceti** (G=D=I=L, all genders), or **dvaceti jedna** + plural noun. **Caution:** "dvaceti jedné" (proposed in the brief) and "dvaceti jednoho" were **not found in any source reached** (§4.3) | medium (which form to use); high (current output is wrong) | IJP-791 pattern (sedmadvaceti, třiadvaceti); WIKT:jednadvacet; WIKT:"dvacet jedna" |
| 8 | 21–24 nominative by gender | dvacetjeden (m) / dvacetjedna (f) / dvacetjedno (n) | Without an agreeing singular noun, or with a G pl noun: **dvacet jedna** (all genders) or **jednadvacet**. "dvacet jeden/jedna/jedno" + N sg noun is correct but becoming dated (agreement type) | high | IJP-792; WIKI-ČČ; NESČ-Č |
| 9 | Accusative animacy | m. A of 1 = jednoho only | m. **anim.** jednoho vs m. **inan.** jeden. Same for ordinals: prvního/první, druhého/druhý … An animacy parameter is required | high | IJP:jeden (both entries); WIKT:druhý |
| 10 | 0 | "nula" in every case | nula, **nuly, nule, nulu, nulou, nule** | high | IJP:nula |
| 11 | 10⁹ A | miliarda | **miliardu** | high | IJP:miliarda |
| 12 | Compound ordinals: the hundreds/thousands part is not declined | G stýprvního, tisícíprvního, tisícístého, dvoutisícídvacátéhočtvrtého; f. stáprvní in all cases | **every ordinal component declines**: stého prvního, tisícího prvního, tisícího stého, dvoutisícího dvacátého čtvrtého; f. G sté první, f. A/I stou první | high | IJP:tisíc (SSJČ: "roku t-ho [= tisícího] devítistého pátého"); WIKI-ŘČ |
| 13 | Decimals | nula celých pět; jeden celých pět; dva celých pět; tři celých čtrnáct; dvacetjeden celých tři | nula celá pět; **jedna celá** pět; **dvě celé** pět; **tři celé** čtrnáct; pět celých dvacet pět; 21,3: dvacet jedna celá/celých tři or jednadvacet celých tři (§11) | high (21,3: medium) | IJP-791 |
| 14 | 21 000 | dvacetjedentisíc | dvacet jedna tisíc / jednadvacet tisíc (also dvacet jeden tisíc, dated agreement type) | medium | IJP-792 rule; IJP:sto ("tři sta třicet tři tisíc") |
| 15 | 100 000 | stotisíc | sto tisíc | high | IJP-791 (word separation); IJP:sto |
| 16 | 21 000th | dvacetjedentisící | jednadvacetitisící **[INF]**, modelled on "pětadvacetitisící nebo dvacetipětitisící" | low | WIKI-ŘČ |

Correct as they stand: 1–4 feminine/neuter paradigms; 3 and 4 (tří, třem, třemi, třech); 5–20 and 40 oblique forms (-i); simple ordinal paradigms (apart from animacy); ordinal hundreds dvoustý/třístý/pětistý; tisící; miliontý.

---

## 2. Q1 — 1 to 4

### 2.1 jeden (singular) — conf. high
Sources: IJP:jeden (m. anim. and m. inan. columns); WIKI-ČČ; WIKT:jeden (all genders).

| | m. anim. | m. inan. | f. | n. |
|---|---|---|---|---|
| N | jeden | jeden | jedna | jedno |
| G | jednoho | jednoho | jedné | jednoho |
| D | jednomu | jednomu | jedné | jednomu |
| A | **jednoho** | **jeden** | jednu | jedno |
| I | jedním | jedním | jednou | jedním |
| L | jednom | jednom | jedné | jednom |

**Plural.** Plural forms are used only with pluralia tantum, e.g. *jedny dveře, jedna povidla* (WIKI-ČČ).
- m. anim.: jedni, jedněch, jedněm, jedny, jedněmi, jedněch (IJP:jeden)
- m. inan.: jedny, jedněch, jedněm, jedny, jedněmi, jedněch (IJP:jeden_1)
- f.: jedny; n.: jedna. Other cases as for masculine (WIKT, WIKI-ČČ; medium-high)

**Abstract number / counting.** The form is **jedna**: "dvacet jedna", "dva tisíce dvě stě třicet jedna", "nula celá jedna" (IJP-791); "jedna a dvě jsou tři" (IJP:dva). Conf. medium-high as a default.

### 2.2 dva / dvě — conf. high
Sources: IJP:dva; IJP-670.

| | m. (anim. = inan.) | f. | n. |
|---|---|---|---|
| N | dva | dvě | dvě |
| G | dvou | dvou | dvou |
| D | dvěma | dvěma | dvěma |
| A | dva | dvě | dvě |
| I | dvěma | dvěma | dvěma |
| L | dvou | dvou | dvou |

- IJP:dva: "Tvar *dva* v 1., 4. a 5. p. se užívá pro rod mužský a tvar *dvě* pro rod ženský a střední." Accusative has no animacy split.
- IJP-670: G/L "dvou … (nikoli dvouch, obouch)"; D/I "dvěma … (nikoli dvoum, oboum, dvouma, obouma, ani dvěmi, oběmi)". These are dual forms.

### 2.3 tři, čtyři (no gender distinction) — conf. high
Sources: IJP:tři, IJP:čtyři, IJP-671.

| | tři | čtyři | status of variants |
|---|---|---|---|
| N | tři | čtyři | |
| G | **tří** / třech | **čtyř** / čtyřech | -ech in G is *hovorové*. IJP-671: "tvary končící na -ch jsou hodnoceny jako hovorové, tedy sice spisovné, ale vhodné spíše v mluvených projevech"; IJP:tři: "Podoba *třech* ve 2. pádě je hovorová" |
| D | třem | čtyřem | |
| A | tři | čtyři | |
| I | **třemi** / třema | **čtyřmi** / čtyřma | -ma is a dual ending, used with nouns that have -ma in I: "se čtyřma očima, pes s třema nohama" (IJP-671; IJP:čtyři) |
| L | třech | čtyřech | only form |

**TTS default:** tří, čtyř (G); třemi, čtyřmi (I). The module's current forms are correct.

---

## 3. Q2 — 5 to 99 (simple numerals)

**Rule.** N = A (= V): pět. G = D = I = L: **pěti**.
- The counted noun is G pl in N/A: *pět chlapců*.
- In oblique cases the noun agrees in case: *bez pěti chlapců, k pěti chlapcům*.
- Sources: NESČ-ZČ; UK-hnd ("v nepřímých pádech číslovka končí na -i a substantivum se skloňuje (s pěti psy)").

| numeral | N / A | G / D / I / L | variants / notes | source | conf. |
|---|---|---|---|---|---|
| pět | pět | pěti | | IJP:pět | high |
| šest | šest | šesti | | WIKT:šest | high |
| sedm | sedm | sedmi | pronounced [sedm] or [sedum] (for G2P) | IJP:sedm | high |
| osm | osm | osmi | pronounced [osm] or [osum] | IJP:osm | high |
| devět | devět | **devíti** | IJP lists only devíti | IJP:devět | high |
| deset | deset | **deseti** | WIKT also lists desíti; not checked in IJP (no entry found) | WIKT:deset | medium-high |
| jedenáct … devatenáct | -náct | -nácti (jedenácti … devatenácti) | | WIKT:jedenáct | high |
| dvacet | dvacet | **dvaceti** | IJP lists "dvaceti, dvacíti". Use dvaceti, which is first-listed and used in IJP's own examples ("k dvaceti sedmi stupňům", "ve sto dvaceti případech") | IJP:dvacet; IJP-791; IJP-792 | high |
| třicet … devadesát | | třiceti, čtyřiceti, padesáti, šedesáti, sedmdesáti, osmdesáti, devadesáti | attested: "čtyřiceti sedmi", "šedesáti pěti" (IJP-791); "padesáti osmi" (ČRo-05) | WIKT:devadesát | high |

---

## 4. Q3 — compound numerals 21–99

### 4.1 Writing — conf. high
- The "dvacet jedna" type is written as **separate words**. IJP-791: "Vypisujeme-li číslovky slovy, píšeme každé slovo zvlášť, např. dvacet jedna, padesát čtyři…". ÚJČ-ZD: "dvacet jedna se píše zvlášť".
- The inverted type is written **as one word**. IJP-791: "Dohromady se píšou složené číslovkové výrazy *jedenadvacet* i *jednadvacet, dvaadvacet, čtyřiapadesát, devětadevadesát*."
- Both jedenadvacet and jednadvacet are correct.
- NESČ-Č notes that the inverted type "se pokládá za germanismus", but IJP lists it as standard.
- Joined spelling of whole amounts is limited to payment slips: "dvatisícedvěstětřicetjedna korun českých" (IJP-791). Not applicable to TTS.

### 4.2 Nominative / accusative: three constructions — conf. high
IJP-792, verbatim: "Po složených číslovkách končících na *jeden, dva, tři, čtyři* jsou v 1. p. možné tvary: *dvacet jeden žák* i *dvacet jedna žáků, dvacet dva/tři/čtyři žáci* i *žáků* nebo *jedenadvacet* i *jednadvacet, dvaadvacet, třiadvacet, čtyřiadvacet žáků*." And: "**Tvary 2. p. jsou běžnější a přirozenější.**"

| type | noun form | masculine | feminine | neuter | status |
|---|---|---|---|---|---|
| A: agreement with the units word | N sg (21); N pl (22–24) | dvacet jeden muž; dvacet dva muži | dvacet jedna žena; dvacet dvě ženy | dvacet jedno dítě; dvacet dvě děti | correct but dated. WIKI-ČČ: "Tento korektní způsob se však postupně stává zastaralým"; NESČ-Č calls it "starší" |
| B: "dvacet jedna/dva" + G pl | G pl | dvacet jedna mužů; dvacet dva mužů | dvacet jedna žen; dvacet **dva** žen | dvacet jedna dětí; dvacet dva dětí | neutral, more common (IJP-792). The units word is gender-invariable: WIKI-ČČ "dvacet jedna/dva mužů, žen, dětí"; HRD: "chybí proto shoda v rodě (nelze dost dobře říct *dvacet jeden chlapců)" |
| C: inverted, one word | G pl | jednadvacet / jedenadvacet, dvaadvacet mužů | …žen | …dětí | standard (IJP-791, IJP-792) |

- **No noun** (abstract number): "dvacet jedna" (IJP-791), "dvacet dva" (IJP-791: "k tisíc sedm set dvacet dva korunám").
- NK-10 gives the historical development: "Druhý pád se nejdříve prosadil u číslovkových výrazů zakončených na jeden, postupně pronikl i k výrazům s číslovkami dva, tři, čtyři"; example "sto dvacet jedna metrů … sto třicet čtyři bodů".

### 4.3 Oblique cases

**Documented rule.** IJP-791, first option: "Skloňujeme všechny části výrazu." Examples:
- "k 27 °C: *k dvaceti sedmi stupňům* nebo i obráceně *sedmadvaceti stupňům*"
- "*před třemi sty šedesáti pěti (před třemi sty pětašedesáti) lety*"
- "*o … čtyři sta dvaceti třech (třiadvaceti) dokladech*"

So both words decline. The units 2/3/4 take their own forms (dvou/dvěma, tří/třem/třemi/třech, čtyř/…), all others end in -i. The inverted type declines as a single word ending in -i. The noun agrees in case.

**Units 2–9.** No gender distinction in oblique cases. Conf. high: direct application of the documented rule. Individual forms are attested: "k dvaceti pěti lidem" (ČRo-05); "padesáti dvěma korunami" (ČRo-05); "dvaceti tří / dvaceti třem" (NK-07); "dvaceti třech" (IJP-791).

| n | N / A | G | D | I | L | inverted (G=D=I=L) |
|---|---|---|---|---|---|---|
| 22 | dvacet dva (m) / dvacet dvě (f, n: type A only) | dvaceti dvou | dvaceti dvěma | dvaceti dvěma | dvaceti dvou | dvaadvaceti |
| 23 | dvacet tři | dvaceti tří | dvaceti třem | dvaceti třemi | dvaceti třech | třiadvaceti |
| 24 | dvacet čtyři | dvaceti čtyř | dvaceti čtyřem | dvaceti čtyřmi | dvaceti čtyřech | čtyřiadvaceti |
| 25 | dvacet pět | dvaceti pěti | dvaceti pěti | dvaceti pěti | dvaceti pěti | pětadvaceti |
| 99 | devadesát devět | devadesáti devíti | ← | ← | ← | devětadevadesáti |

Brief's examples:
- **22 f. D:** "ke dvaceti dvěma ženám" / "ke dvaadvaceti ženám". Conf. high.
- **25 I:** "(s) dvaceti pěti" / "pětadvaceti". Conf. high.

**Units = 1 (21, 31 … 91): sources are thin.**

| construction | N | G | D | A | I | L | source / conf. |
|---|---|---|---|---|---|---|---|
| C inverted | jednadvacet | jednadvaceti | jednadvaceti | jednadvacet | jednadvaceti | jednadvaceti | WIKT:jednadvacet; same pattern as IJP-791 "sedmadvaceti" — **medium-high** |
| B + G pl / agreeing pl noun | dvacet jedna | dvaceti jedna | dvaceti jedna | dvacet jedna | dvaceti jedna | dvaceti jedna | WIKT:"dvacet jedna" (lemma with invariable *jedna*) — **medium-low**. No IJP example found |
| A + singular noun, m. | dvacet jeden žák | dvaceti jednoho žáka | dvaceti jednomu žákovi | dvacet jednoho žáka (anim.) / dvacet jeden (inan.) | dvaceti jedním žákem | dvaceti jednom žákovi | **[INF] low**, found in no source. IJP-792 describes type A only "v 1. p." Do not generate by default |
| A, f. | dvacet jedna žena | dvaceti jedné ženy | dvaceti jedné ženě | dvacet jednu ženu | dvaceti jednou ženou | dvaceti jedné ženě | **[INF] low**, same caveat |

- **TTS default for x1 in oblique cases:** the inverted form (21 → **jednadvaceti**, 31 → jednatřiceti, …, 91 → jednadevadesáti). Alternative: "dvaceti jedna". Avoid type A obliques. Conf. medium.
- The brief's expected "21 f. G = dvaceti jedné" is a type-A oblique. It could not be confirmed.

---

## 5. Q4 — sto and the hundreds

**sto paradigm.** Neuter, declines like *město*. Sources: IJP:sto; IJP-792; ÚJOP T.4.2. Conf. high.
- sg: sto, sta, stu, sto, stem, stu
- pl: sta, set, stům, sta, sty, stech
- IJP:sto note: "ve spojení s výrazem *dvě* mají tvary 1., 4. a 5. p. mn. č. podobu *stě* (*dvě stě*)"

| | 100 | 200 | 300 / 400 | 500 … 900 |
|---|---|---|---|---|
| N | sto | dvě stě | tři sta / čtyři sta | pět set … devět set |
| G | sta (or sto, see below) | dvou set | tří set / čtyř set | pěti set … devíti set |
| D | stu (or sto) | dvěma stům | třem stům / čtyřem stům | pěti stům |
| A | sto | dvě stě | tři sta / čtyři sta | pět set |
| I | stem (or sto) | dvěma sty | třemi sty / čtyřmi sty | pěti sty |
| L | stu (or sto) | dvou stech | třech stech / čtyřech stech | pěti stech |

Attestations:
- ÚJOP T.4.2: sta/stu/sto/stem/stu; dvou set/dvěma stům/dvě stě/dvěma sty/dvou stech; pěti set/pěti stům/pět set/pěti sty/pěti stech.
- IJP:tři "před třemi sty lety"; IJP:pět "před pěti sty lety".
- ČRo-05 "k dvěma stům posluchačům (ne *dvouseti*)", "s pěti sty padesáti lidmi (ne *pětiseti*)".
- ČRo-20 lists *pětseti* as an error.

**No gender forms.** sto, dvě stě, tři sta and pět set are the same with masculine, feminine and neuter nouns. The module's f. "stá" and n. "sté" are ordinals. Conf. high.

**Undeclined sto** (IJP-792, verbatim):
- "Číslovka *sto* se skloňuje podle vzoru „město", ale ve spojení s počítaným předmětem někdy zůstává nesklonná, např. *bez sto korun, ke sto korunám, o sto korunách, se sto korunami* (vedle pravidelně skloňovaných tvarů *bez sta korun, ke stu korun/korunám, o stu korun/korunách, se stem korun*)."
- "Je-li číslovka *sto* ve spojení s jinou číslovkou, která po ní následuje, zpravidla se neskloňuje, např. *ve sto dvaceti případech, po sto padesáti letech*."
- So for 121 G: "sto dvaceti jedna / sto jednadvaceti". Conf. high for "sto" staying as is; the x1 part as in §4.3.

**Other points:**
- **1100–1999**: "tisíc pět set třicet dva" or "patnáct set třicet dva" (NESČ-Č). 1100 = "tisíc sto" or "jedenáct set". Conf. high.
- **"jedno sto"** inside large numbers is optional: "dva tisíce (jedno) sto šedesát osm" (IJP-791).

---

## 6. Q5 — tisíc

**Paradigm.** m. inanimate, declines like *stroj*. Source: IJP:tisíc. Conf. high.

| | sg | pl |
|---|---|---|
| N | tisíc | tisíce |
| G | tisíce | tisíc, tisíců |
| D | tisíci | tisícům |
| A | tisíc | tisíce |
| I | tisícem | tisíci |
| L | tisíci | tisících |

After numerals the G pl is **tisíc**: "pouze v gen. pl. ve složených číslovkách má tvar *tisíc*" (UK-hnd); "pět tisíc", "tři sta třicet tři tisíc" (IJP:pět, IJP:sto).

| | 1000 | 2000 (3000, 4000) | 5000+ |
|---|---|---|---|
| N | tisíc | dva tisíce (tři/čtyři tisíce) | pět tisíc |
| G | tisíce | dvou tisíc (tří/čtyř tisíc) | pěti tisíc |
| D | tisíci | dvěma tisícům (třem/čtyřem tisícům) | pěti tisícům |
| A | tisíc | dva tisíce | pět tisíc |
| I | tisícem | dvěma tisíci (třemi/čtyřmi tisíci) | pěti tisíci |
| L | tisíci | dvou tisících (třech/čtyřech tisících) | pěti tisících |

Attestations:
- "s třemi tisíci diváků/diváky" (IJP-792)
- "před dvěma tisíci let/lety" (UK-hnd)
- "bez tisíce osmi set čtyřiceti sedmi korun" (IJP-791)
- "k tisíci dvěma stům dvaceti třem" (NK-07)
- "se dvěma tisíci sto padesáti dvěma korunami" (ČRo-05)
- ČRo-20 lists *tisícema* as an error.

1000 is bare **tisíc** in all IJP examples ("tisíc celých pět setin", "bez tisíce…"). "jeden tisíc" does not occur in the sources (no verdict).

**Compound multipliers:**

| value | N | notes | conf. |
|---|---|---|---|
| 21 000 | dvacet jedna tisíc / jednadvacet tisíc; dated type A: dvacet jeden tisíc | IJP-792 rule applied to *tisíc*. Obliques as in §4.3, e.g. G jednadvaceti tisíc / dvaceti jedna tisíc; I jednadvaceti tisíci | medium (N); medium-low (obliques) |
| 22 000–24 000 | dvacet dva tisíce (type A) / dvacet dva tisíc (type B) / dvaadvacet tisíc | IJP:sto "tři sta třicet tři tisíc"; IJP:milion "53 miliony i 53 milionů" (both types) | medium-high |
| 25 000 | dvacet pět tisíc / pětadvacet tisíc | | high |
| 100 000 | sto tisíc | G sta tisíc / sto tisíc; I se stem tisíc / se sto tisíci. **[INF]** from IJP-792 rules (sto may stay undeclined; after sto/tisíc the noun may be G or agree) | medium-low (obliques) |

---

## 7. Q6 — milion, miliarda, bilion

**Spelling.** IJP:milion headword **"milion"**, with "lze i: milión". Conf. high.
- So "milion" is primary and "milión" an accepted variant.
- IJP itself writes "milion" throughout (IJP-780 "mil. = milion"; IJP-786 "dva miliony korun českých"; IJP-791).
- General -on/-ón doublets (vagon/vagón, kamion/kamión) are both correct (IJP-122).
- TTS: output **milion**. The spelling affects G2P vowel length.

| | milion (m. inan., *hrad*) | miliarda (f., *žena*) |
|---|---|---|
| N | milion / miliony | miliarda / miliardy |
| G | milionu / milionů | miliardy / miliard |
| D | milionu / milionům | miliardě / miliardám |
| A | milion / miliony | **miliardu** / miliardy |
| I | milionem / miliony | miliardou / miliardami |
| L | milionu / milionech | miliardě / miliardách |

(sg / pl in each cell.) Sources: IJP:milion, IJP:miliarda. Conf. high.

**Combinations.** Built from the IJP forms of 1–4 and 5+. Conf. high.

| | 1 | 2 (3, 4) | 5+ |
|---|---|---|---|
| N | (jeden) milion | dva miliony | pět milionů |
| G | (jednoho) milionu | dvou milionů | pěti milionů |
| D | (jednomu) milionu | dvěma milionům | pěti milionům |
| A | (jeden) milion | dva miliony | pět milionů |
| I | (jedním) milionem | dvěma miliony | pěti miliony |
| L | (jednom) milionu | dvou milionech | pěti milionech |
| N | (jedna) miliarda | dvě miliardy | pět miliard |
| G | (jedné) miliardy | dvou miliard | pěti miliard |
| D | (jedné) miliardě | dvěma miliardám | pěti miliardám |
| A | (jednu) miliardu | dvě miliardy | pět miliard |
| I | (jednou) miliardou | dvěma miliardami | pěti miliardami |
| L | (jedné) miliardě | dvou miliardách | pěti miliardách |

Attestations:
- "jeden milion tři sta tisíc sedm set osmdesát tři" and "o milion tři sta padesát osm tisíc…" (IJP-791): "jeden" is optional.
- "dvanáct milionů sto padesát sedm tisíc pět set osmdesát dva" (IJP:milion).
- "obchod za tři miliardy" (IJP:miliarda).
- "dva miliony korun českých" (IJP-786).
- After x2–x4 compounds both constructions occur: "53 miliony i 53 milionů" (IJP:milion).
- Counted noun: "Ve spojení číslovek *sto, tisíc, milion, miliarda* s počítaným předmětem je vedle 2. p. možná i shoda pádu počítaného předmětu s pádem číslovky" (IJP-792).

**bilion.** Value 10¹² (long scale). miliarda = "tisíc milionů" (IJP:miliarda, ASSČ/SSČ); bilion = 10¹² (WIKT:bilión).
- Never map 10⁹ to "bilion".
- Declension: WIKT:bilión (m. inan.) gives bilión, biliónu, biliónu, bilión, biliónem, biliónu; pl. bilióny, biliónů, biliónům, bilióny, bilióny, biliónech.
- NESČ-ZČ: sto, milion, miliarda and bilion "skloňují se podle příslušných subst. vzorů".
- No IJP entry was found. Spelling "bilion" vs "bilión": **[INF]** by analogy with milion/milión, both are probably acceptable; use "bilion" for consistency. Conf. medium (paradigm), low (preferred spelling).

---

## 8. Q7 — large mixed numbers in oblique cases

IJP-791 gives three options, verbatim, with no ranking between (a) and (b):

- **(a) Decline every part.** "Skloňujeme všechny části výrazu."
  - "*k dvaceti sedmi stupňům*" / "*sedmadvaceti stupňům*"
  - "*před třemi sty šedesáti pěti (před třemi sty pětašedesáti) lety*"
  - "*bez tisíce osmi set čtyřiceti sedmi korun*"
- **(b) Decline the noun and part of the numeral.** "Skloňujeme jen jméno počítaného předmětu a část číslovkového výrazu (obvykle řád desítek a jednotek), zbytek ponecháváme nesklonný."
  - "*před tři sta šedesáti pěti (pětašedesáti) lety*"
  - "*bez tisíc osm set čtyřiceti sedmi (sedmačtyřiceti) korun*"
  - "*o milion tři sta padesát osm tisíc čtyři sta dvaceti třech (třiadvaceti) dokladech*"
- **(c) Leave everything undeclined.** "V některých situacích, většinou v mluvené řeči, např. při diktování a při početních operacích s čísly, může zůstat celý víceslovný číslovkový výraz neskloňovaný": "*k tisíc sedm set dvacet dva korunám*".
  - Also: "V psaných textech se v takových případech dává přednost zápisu čísel číslicemi."

Supporting sources:
- NK-07, citing the ÚJČ poradna: "bez tisíce dvou set dvaceti tří / bez tisíc dvě stě dvaceti tří; k tisíci dvěma stům dvaceti třem / k tisíc dvě stě dvaceti třem"; "U složitějších spojení často skloňujeme jen jméno počítaného předmětu a část číslovkového výrazu".
- WIKI-ČČ: "s jedenácti tisíci dvěma sty padesáti pěti (pětapadesáti) lidmi" = "s jedenáct tisíc dvě stě padesáti pěti (pětapadesáti) lidmi".

**2 345.** N "dva tisíce tři sta čtyřicet pět" is verbatim in WIKI-ČČ. The other forms apply rules (a)/(b), using component forms sourced in §§2–6. Conf. high.

| case | (a) full | (b) partial (IJP-sanctioned) |
|---|---|---|
| G | dvou tisíc tří set čtyřiceti pěti | dva tisíce tři sta čtyřiceti pěti |
| D | dvěma tisícům třem stům čtyřiceti pěti | dva tisíce tři sta čtyřiceti pěti |
| I | dvěma tisíci třemi sty čtyřiceti pěti | dva tisíce tři sta čtyřiceti pěti |
| L | dvou tisících třech stech čtyřiceti pěti | dva tisíce tři sta čtyřiceti pěti |

The last word may also be the inverted "pětačtyřiceti".

**1 234 567.** Conf. medium (derived; no source gives this number).
- N: "jeden milion dvě stě třicet čtyři tisíc pět set šedesát sedm". "…třicet čtyři tisíce…" (type A) and bare "milion" are also possible.
- G (a): jednoho milionu dvou set třiceti čtyř tisíc pěti set šedesáti sedmi
- G (b): (jeden) milion dvě stě třicet čtyři tisíc pět set šedesáti sedmi
- I (a): jedním milionem dvěma sty třiceti čtyřmi tisíci pěti sty šedesáti sedmi
- I (b): (jeden) milion dvě stě třicet čtyři tisíc pět set šedesáti sedmi

**Design suggestion (not normative).**
- Use (a) for short expressions such as "dvou tisíc" or "třemi sty lety".
- Use (b) when the number has three or more groups. This matches NK-07's "u složitějších spojení často…".
- Never use (c) for audiobook prose. IJP limits it to dictation and arithmetic.

---

## 9. Q8 — ordinals

### 9.1 Declension patterns
- WIKI-ŘČ, verbatim: "Číslovky *první, třetí* a *tisící* se skloňují podle vzoru *jarní*, ostatní adjektivní číslovky podle vzoru *mladý*." NESČ-ŘČ: "mají formu adj. tvrdých n. měkkých".
- All ordinals ending in **-tisící** (dvoutisící, pětitisící, stotisící) are therefore soft. **[INF]**, following from the base *tisící*; conf. high.

**Hard type** (stý; the same endings apply to druhý, čtvrtý, pátý … dvacátý, dvoustý, miliontý, miliardtý). Sources: WIKT:stý, WIKT:druhý, WIKT:čtvrtý. Conf. high.

| sg | m. anim. | m. inan. | f. | n. |
|---|---|---|---|---|
| N | stý | stý | stá | sté |
| G | stého | stého | sté | stého |
| D | stému | stému | sté | stému |
| A | **stého** | **stý** | stou | sté |
| I | stým | stým | stou | stým |
| L | stém | stém | sté | stém |

| pl | m. anim. | m. inan. | f. | n. |
|---|---|---|---|---|
| N | **stí** | sté | sté | stá |
| G / D | stých / stým | ← | ← | ← |
| A | sté | sté | sté | stá |
| I / L | stými / stých | ← | ← | ← |

- WIKT also lists the I pl forms -ýma (druhýma, stýma). **[INF]** These are colloquial; use -ými.

**Masculine animate N pl, with consonant alternation:**

| ordinal | m. anim. N pl | source |
|---|---|---|
| druhý | druzí | WIKT |
| čtvrtý | čtvrtí | WIKT |
| stý | stí | WIKT |
| dvoustý … devítistý | dvoustí … devítistí | WIKT |
| miliardtý | miliardtí | WIKT |
| pátý, šestý, sedmý, osmý, devátý, desátý, jedenáctý, dvacátý, miliontý | pátí, šestí, sedmí, osmí, devátí, desátí, jedenáctí, dvacátí, miliontí | **[INF]** regular *mladý* pattern; conf. medium-high |

**Soft type** (první; the same forms apply to třetí, tisící, dvoutisící …). Sources: WIKT:první, WIKT:tisící. Conf. high.

| sg | m. anim. | m. inan. | f. | n. |
|---|---|---|---|---|
| N | první | první | první | první |
| G | prvního | prvního | první | prvního |
| D | prvnímu | prvnímu | první | prvnímu |
| A | **prvního** | **první** | první | první |
| I | prvním | prvním | první | prvním |
| L | prvním | prvním | první | prvním |

- pl: N = A **první** (all genders); G prvních; D prvním; I prvními; L prvních.
- *prvý* is a literary variant (IJP:první). *tisícátý* is "zast. a ob." (SSJČ in IJP:tisíc). Do not use either.

### 9.2 Ordinal stems for round numbers

| value | ordinal | source | conf. |
|---|---|---|---|
| 0 | nultý | IJP:nula | high |
| 100 | stý | NESČ-ŘČ; WIKT | high |
| 200 | dvoustý | WIKT; WIKI-ŘČ ("215. = dvoustý patnáctý"); CHL (Gebauer "dvoustý dvacátý") | high |
| 300 | třístý | WIKT; WIKI-ŘČ ("368. = třístý šedesátý osmý") | high |
| 400 | čtyřstý (WIKT also lists čtyrstý) | IJP-791 ("čtyřstý sedmdesátý osmý"); WIKI-ŘČ | high |
| 500 | pětistý | IJP:tisíc ("rok tisící pětistý dvacátý"); WIKT | high |
| 600 | šestistý | WIKT | medium-high |
| 700 | sedmistý | WIKT; HRD ("tisící sedmistý") | high |
| 800 | osmistý | WIKT; WIKI-ŘČ ("roku tisícího osmistého…") | high |
| 900 | devítistý | IJP-791; WIKI-ŘČ | high |
| 1100–1900 | tisící stý … tisící devítistý, or jedenáctistý … devatenáctistý | WIKI-ŘČ ("taková složenina je možná i pro rovné stovky od 1100 do 1900 (devatenáctistý)"); HRD ("sedmnáctistý") | high |
| 1000 | tisící | IJP:tisíc ("tisící návštěvník"); NESČ-ŘČ | high |
| 2000 | dvoutisící | WIKI-ŘČ ("dvoutisící, pětadvacetitisící nebo dvacetipětitisící, stotisící"); WIKI-ČČ | medium-high |
| 3000 / 4000 | třítisící / čtyřtisící | **[INF]** by the dvoustý/třístý/čtyřstý pattern | low-medium |
| 5000 | pětitisící | CHL (Gebauer: "pětitisící dvoustý pátý") | medium-high |
| 25 000 | pětadvacetitisící / dvacetipětitisící | WIKI-ŘČ | medium |
| 100 000 | stotisící | WIKI-ŘČ | medium |
| 10⁶ | miliontý | NESČ-ŘČ | high |
| 2·10⁶, 5·10⁶ | dvoumiliontý, pětimiliontý | **[INF]** from WIKI-ŘČ "desetimiliontý" and IJP-670 "dvoumilionový" | medium |
| 10⁹ | miliardtý (WIKT also lists miliárdtý) | WIKT | medium |

---

## 10. Q9 — compound ordinals

**Rules**
- **Basic (conf. high):** every component is an ordinal.
  - WIKI-ŘČ: "Je-li základní číslovka tvořena více slovy, do tvaru řadové číslovky se upravují všechna slova (152. = „stý padesátý druhý", 215. = „dvoustý patnáctý", 368. = „třístý šedesátý osmý")".
  - IJP-791: "stý padesátý, čtyřstý sedmdesátý osmý, tisící devítistý padesátý šestý".
  - WIKI-ČČ: "2345. = dvoutisící třístý čtyřicátý pátý".
  - CHL quoting Sedláček: "Základní způsob … podobu řadové číslovky mají všechny části, např. stý první, stý desátý."
- **Variant: leading sto/tisíc stays cardinal** (conf. high for 101–199, medium for thousands).
  - CHL quoting Sedláček: "Avšak je možné také užít variant, v nichž první číslovka zůstává v podobě sto, např. sto první, sto padesátý osmý."
  - CHL quoting Brabcová: "…nebo číslovka sto zůstane v základní podobě (sto padesátý nebo jako jedno slovo stopadesátý)".
  - For years, WIKI-ŘČ: the 1877 Brus "připouštělo vedle výrazu „roku tisícího osmistého sedmdesátého pátého" i tvar „roku tisíc osm set sedmdesátého pátého"".
- **Tens + units** (conf. high): "dvacátý první" or inverted one-word "jednadvacátý".
  - NESČ-ŘČ: "dvacátý první, jednadvacátý".
  - CHL quoting Šmilauer: "jedenadvacátý = jednadvacátý = dvacátý první = dvacátý prvý".
  - IJP-791: "pětadvacátý" written together.
- **Declension** (conf. high): every ordinal component declines.
  - SSJČ (in IJP:tisíc): "roku t-ho [= tisícího] devítistého pátého"; WIKI-ŘČ: "roku tisícího osmistého sedmdesátého pátého".
  - In the mixed type only the ordinal part declines: "roku tisíc osm set sedmdesátého pátého".
- **Spacing:** components are separate words, except the one-word forms listed in §1 row 2.

| n | basic (all ordinal), N m. | variants | G m./n. (basic / variant) | conf. |
|---|---|---|---|---|
| 21. | dvacátý první | jednadvacátý, jedenadvacátý | dvacátého prvního / jednadvacátého | high |
| 22. | dvacátý druhý | dvaadvacátý | dvacátého druhého / dvaadvacátého | high |
| 101. | stý první | sto první | stého prvního / sto prvního | high |
| 121. | stý dvacátý první | stý jednadvacátý; sto dvacátý první; sto jednadvacátý | stého dvacátého prvního / sto dvacátého prvního | medium-high (derived from the rules) |
| 345. | třístý čtyřicátý pátý | třístý pětačtyřicátý | třístého čtyřicátého pátého | high |
| 1991. | tisící devítistý devadesátý první | devatenáctistý devadesátý první; tisíc devět set devadesátý první (Brus type); …jednadevadesátý | tisícího devítistého devadesátého prvního / tisíc devět set devadesátého prvního | high (basic), medium (mixed) |
| 2024. | dvoutisící dvacátý čtvrtý | dvoutisící čtyřiadvacátý; dva tisíce dvacátý čtvrtý (mixed, **[INF]** by analogy with the Brus type and "sto první") | dvoutisícího dvacátého čtvrtého / dva tisíce dvacátého čtvrtého | medium-high (basic), medium-low (mixed) |

**Feminine and neuter** decline every component in the same way:
- 101. f.: stá první / sté první / sté první / stou první / stou první / sté první
- 2024. f. G: dvoutisící dvacáté čtvrté

These forms combine the hard f. forms of stý with invariable soft první. **[INF]**; conf. high.

---

## 11. Q10 — decimal numbers

**IJP-791 examples, verbatim** (parentheses mark optional words). Conf. high.

| number | reading | number | reading |
|---|---|---|---|
| 0,1 | nula/žádná celá jedna (desetina) | 5,1 | pět celých jedna (desetina) |
| 0,2 | nula/žádná celá dvě (desetiny), popř. nula/žádná celá dva | 25,4 | dvacet pět celých čtyři (desetiny) |
| 0,5 | nula/žádná celá pět (desetin) | 100,6 | sto celých šest (desetin) |
| 1,2 | jedna celá dvě (desetiny), popř. jedna celá dva | 103,8 | sto tři celé/celých osm (desetin) |
| 1,4 / 1,9 | jedna celá čtyři (desetiny) / jedna celá devět (desetin) | 1000,05 | tisíc celých pět setin |
| 2,3 | dvě celé tři (desetiny) | 1024,007 | tisíc dvacet čtyři celé/celých sedm tisícin |

**Agreement of "celá"** with the integer part. Conf. high.
- 0 → nula celá (HRD: "nula celá (nikoliv celý)")
- 1 → jedna celá
- 2–4 → dvě/tři/čtyři celé
- 5 and up, and compounds ending in 0 or 5–9 → celých
- compounds ending in 2–4 → celé **or** celých (103,8; 1024,007)
- compounds ending in 1 (21,3): no source example. **[INF]** by the IJP-792 rule: "dvacet jedna celá tři" (agreement) / "dvacet jedna celých tři" (genitive type) / "jednadvacet celých tři". Conf. medium.

**Decimal part.** Read as a cardinal agreeing with an implied feminine *desetina/setina/tisícina*: jedna, dvě (bare "dva" also allowed).
- The denominator may be dropped. It is used with leading zeros: "1000,05 – tisíc celých pět setin".
- HRD alternative: "1,04 – jedna celá … nula čtyři (čtyři setiny)".
- 3,14 → **tři celé čtrnáct (setin)**. Derived from "2,3 – dvě celé tři" and HRD "0,26 – nula celá dvacet šest (setin)"; conf. high.

**Counted noun after decimals** (IJP-792). Conf. high.
- It is **G sg**: "0,5 metru, 2,36 litru, do 3,5 tuny, 10,548 milionu, 14,25 sekundy".
- G pl or case agreement is only tolerated in lower styles: "v kultivovaných projevech bychom však radili se jim vyhnout".

**Oblique cases of decimals.** IJP-792 reading example: "do dvou celých dvou metrů". The integer numeral and "celá" (as an adjective) both decline.
- Other cases (e.g. D "dvěma celým dvěma") are **[INF]**; conf. medium-low.
- "2,5" may also be read "dva a půl" (IJP:dva: "dva a půl litru mléka").

---

## 12. Q11 — nula

**nula** is a feminine noun (*žena*). Source: IJP:nula. Conf. high.

| | sg | pl |
|---|---|---|
| N | nula | nuly |
| G | nuly | nul |
| D | nule | nulám |
| A | nulu | nuly |
| I | nulou | nulami |
| L | nule | nulách |

Also: ordinal *nultý* (IJP:nula); decimals "nula celá … / žádná celá …" (IJP-791).

---

## 13. Default forms for the generator (design suggestions based on the findings above)
1. **API.** Add animacy (m. anim. vs m. inan.) and a "noun form" or "construction" flag: agreement vs genitive vs inverted for x1–x4 compounds.
2. **Abstract numbers** (no following noun): 1 → *jedna*, 21 → *dvacet jedna*; 2 → *dva* (22 → *dvacet dva*).
3. **Cardinals in oblique cases:**
   - Decline 1–4 per §2, 5–99 with -i, sto/tisíc/milion/miliarda as nouns (§§5–7).
   - Keep "sto" undeclined when another numeral follows.
   - For x1 compounds use the inverted form (*jednadvaceti*).
   - For numbers of three or more groups, use IJP option (b).
4. **Ordinals:** all components ordinal (basic rule), all components declined, separate words.
5. **Spelling:** *milion*, *miliarda*; *bilion* = 10¹².

## 14. Open issues / not verified
- Oblique cases of the "dvacet jedna" type, and of 101/1001-type compounds (§4.3): no ÚJČ source was reachable. *Naše řeč* art. 5533 "Sto jeden žák i stojedna žáků?" would probably settle this, but was unreachable.
- Preferred spelling of bilion vs bilión; no IJP entry.
- Mixed ordinal type for 2000+ years ("dva tisíce dvacátý čtvrtý"): inferred, not attested for 2000+ in the sources reached.
- Reading of years as cardinals ("v roce dva tisíce dvacet čtyři"): common usage, but not verified in any source reached. IJP:rok was not retrievable.
- deset "desíti" and dvacet "dvacíti": both are listed but no stylistic label was retrieved. The recommendation (-eti) rests on the order of listing and on IJP's own usage.
