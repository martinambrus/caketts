# Slovak numerals: normative spec for `num2words_sk.py` (TTS text normalisation)

> **Note (added when this file moved into the repository).** These research notes were compiled on 25 September 2026, before the native-speaker review. Where they differ from the modules, `NUM2WORDS_CHANGES.md` §4 and the tests take precedence. Examples: *dvomi* with feminine nouns, *stodve knihy*, *dvetisíci*, and milión groups that decline before a smaller part.

Compiled 2026-09-25 for review by a native speaker. Scope: standard Slovak (spisovná slovenčina) cardinal and ordinal numerals, as used when reading digits aloud.

**Legend**
- **Confidence:** **H** = stated explicitly in a JÚĽŠ/SAV normative source I read and not contradicted, or a trivially regular form. **M** = one source only, a secondary source, or a direct inference from an explicit rule. **L** = my own inference with no direct support, or the sources conflict.
- **[INF]** = my inference. It was not read in any source.
- `Src` codes point to the source table in §0. "PSP 91" and "PSP 98" are quoted second-hand from JAR (see §0).

---

## 0. Sources and access log

### 0.1 Sources I read (through the WebFetch tool; quotes are exact short excerpts)

| ID | Source | URL | Authority |
|---|---|---|---|
| MSJ | Ružička, J. (ed.): *Morfológia slovenského jazyka*. SAV 1966. Numerals chapter pp. 314–356. Readable part: pp. 300–327 only | https://www.juls.savba.sk/ediela/msj/pdf100/msj300-399.pdf (TOC: https://www.juls.savba.sk/ediela/msj/pdf100/MSJ_000_099_LQ.pdf) | academic grammar, JÚĽŠ (old: 1966) |
| NAV | Navrátil, L.: Neohybnosť v ohybných slovných druhoch. *Kultúra slova* 37, 2003, č. 6, pp. 336–341 | https://www.juls.savba.sk/ediela/ks/2003/6/ks2003-6.html | JÚĽŠ journal |
| BEL | *Encyclopaedia Beliana* (SAV), entry "číslovky" (2003) | https://beliana.sav.sk/heslo/cislovky | SAV encyclopedia |
| JAR | Jarošová, A.: Zložené číslovky so zreteľom na písanie spolu a oddelene. *Slovenská reč* 86, 2021, č. 2, pp. 208–229. Quotes PSP 1991 (p. 44) and PSP 1998 (p. 24) | https://www.sav.sk/journals/uploads/12101101zlozene-cislovky-so-zretelom-na-pisanie-spolu-a-oddelene.pdf | JÚĽŠ journal |
| SRA | Šrámeková, I. (JÚĽŠ): K potenciálnej variantnosti číslovkových tvarov dvojstý, dvestý. *Kultúra slova* 57, 2023, č. 5, pp. 268–284 | https://www.sav.sk/journals/uploads/11281051268-284-k-potencialnej-variantnosti-cislovkovych-tvarov-dvojsty-dvesty.pdf | JÚĽŠ journal |
| PAL | Páleníková, J. (JÚĽŠ SAV): Číslovky dvadsaťjeden, tridsaťjeden, štyridsaťjeden. Quark, "Jazykové okienko" column (page dated 5 Jul 2026; cites SSSJ 2011) | https://www.quark.sk/cislovky-dvadsatjeden-tridsatjeden-styridsatjeden/ | JÚĽŠ linguist, popular column |
| POV18 | Považaj, M. (JÚĽŠ): Pravopis rozličných tvarov čísloviek sedem, osem, sedemdesiat, osemdesiat. Quark, 7 Jan 2018 | https://www.quark.sk/pravopis-rozlicnych-tvarov-cisloviek-sedem-osem-sedemdesiat-osemdesiat/ | JÚĽŠ linguist |
| DUCH | Duchková, S. (JÚĽŠ SAV): Ako čítame a skloňujeme desatinné čísla. Quark, 5 Jun 2021 | https://www.quark.sk/ako-citame-a-sklonujeme-desatinne-cisla/ | JÚĽŠ linguist |
| POV94 | Považaj, M.: Sonda do jazyka rozhlasového spravodajstva. *Kultúra slova* 28, 1994, č. 1 | https://www.juls.savba.sk/ediela/ks/1994/1/ks1994-1.html | JÚĽŠ journal |
| MAS96 | Masár, I.: Čo nekrášli Krásy Slovenska. *Kultúra slova* 30, 1996, č. 3, p. 157 | https://www.juls.savba.sk/ediela/ks/1996/3/ks1996-3.html | JÚĽŠ journal |
| SNK11 | Slovenský národný korpus (JÚĽŠ): Odpovede na kvíz – Noc výskumníkov 2011 | https://korpus.juls.savba.sk/promo(2f)NocVyskumnikov2011(2f)Kviz.html | JÚĽŠ (corpus dept.) |
| TASR18 | TASR: Slovenčina: Ako písať jednoslovné a viacslovné číslovky (21 Aug 2018). No linguist or source named | https://www.teraz.sk/import/slovencina-jednoslovne-viacslovne-cislov/343961-clanok.html | secondary |
| HS | hovormespisovne.sk, "Desatinné čísla" (credits Jazyková poradňa JÚĽ; operator unknown) | https://hovormespisovne.sk/portfolio/desatinne-cisla/ | secondary |
| SKA | skvelaasistentka.sk, "Bolo to pred dvoma alebo dvomi rokmi?" (cites PSP + KSSJ) | https://www.skvelaasistentka.sk/sk/pravopis-gramatika-profesionalka-vie-svet-asistentky/bolo-pred-dvoma-alebo-dvomi-rokmi | secondary |
| STVR | Rádio Slovensko language spots with S. Mislovičová (JÚĽŠ): "S dvoma aj dvomi moderátormi" (2016); "S troma, či s tromi?" (2017). **Audio only: I read only the titles.** | https://slovensko.stvr.sk/rubriky/aktualne-rubriky/99152/s-dvoma-aj-dvomi-moderatormi ; https://slovensko.stvr.sk/rubriky/aktualne-rubriky/144763/s-troma-ci-s-tromi | JÚĽŠ linguist (unread) |
| EDU | eduself.sk morphology pages. Built from sk-spell plus corpus rules and "validated" against the JÚĽŠ morphological database. **Not normative; weak corroboration only** | https://www.eduself.sk/slovensky-jazyk/otazky-databaza-slov ; https://www.eduself.sk/slovensky-jazyk/tvary-slova/tis%C3%ADci ; https://www.eduself.sk/slovensky-jazyk/tvary-slova/nula?word_id=66794 | non-normative |
| WPen | en.wikipedia, "Slovak declension", numerals section (no citations) | https://en.wikipedia.org/wiki/Slovak_declension | tertiary |
| USE | usage only: SME "…dvanásť a štvrť miliónty návštevník"; trencinak.sk "26-miliónteho návštevníka"; Pravda "desiatky miliárd dolárov" | https://domov.sme.sk/c/3311369/bojnicky-zamok-navstivi-jeho-dvanast-a-stvrt-milionty-navstevnik.html ; https://www.trencinak.sk/clanky/13199/bojnicka-zoo-privitala-26-milionteho-navstevnika-stastie-sa-usmialo-na-mladu-rodinu-otec-matej-sem-chodi-uz-od-detstva ; https://spravy.pravda.sk/svet/clanok/646570-ktori-najvacsi-bohaci-prisli-o-desiatky-miliard-dolarov/ | usage |

### 0.2 Could NOT be read. The native reviewer should check these directly.

| Source | Reason |
|---|---|
| slovnik.juls.savba.sk (KSSJ 2003, PSP 2013 dictionary part, SSSJ, morphological database / tvaroslovné tabuľky) | robots.txt disallows automated fetching. slovniky.korpus.sk redirects there. I did **not** work around this. |
| jazykovaporadna.sme.sk answers | HTTP 402. Only question titles were visible, so the answers are unread: q/713 "Je správne dvoma alebo dvomi?", q/3702 "…troma alebo tromi?", q/637 "Ako sa skloňujú zložené číslovky, ktoré…", q/5924 "Ako sa píšu vyššie zložené číslovky…", q/3099 "…skloňuje slovné spojenie tisíc chlapov?", q/8170 "…dvesté výročie alebo dvojsté…", q/2031 "…dve celé alebo dva celé?", q/1680 "Ako sa čítajú desatinné čísla? Je nula c…", q/704 and q/6876 (forms of "celá"), q/5911 (formation of ordinals) |
| sk.wiktionary.org, en.wiktionary.org, sk.wikipedia.org ("Číslovka") | the fetch tool is cache-only for these. Only en.wiktionary "dva" was cached, and it had no Slovak table. **No Wiktionary tables were consulted.** |
| PSP 2000/2013 rules text https://www.juls.savba.sk/ediela/psp2000/psp.pdf | readable only to p. 37. The numerals and morphology chapters were not reached, so PSP is cited only second-hand via JAR. JÚĽŠ states that the PSP 2013 rules text is this PDF: https://www.juls.savba.sk/psp_2013.html |
| MSJ pp. 327–356 (sto, tisíc, milión, ordinals, číslovkové spojenia) | the text window ends at p. 327 and the PNG scans cannot be processed. |
| UK Bratislava textbook (Stanková 2021) https://fphil.uniba.sk/fileadmin/fif/katedry_pracoviska/kzur/Publikacie_pdf/SSJ2_skripta.pdf | the numerals chapter (p. 30+) is beyond the readable window. |
| slovnik.aktuality.sk (KSSJ/PSP mirror) | the fetch tool reports the site as blocked. |
| bash/curl | every external host is blocked by the egress policy (HTTP 403). |

**Caveat.** WebFetch passes documents through a summariser. I cross-checked each quote below with a second, narrower query. One table (MSJ "jeden") came back garbled, so I do not quote it cell by cell.

---

## 1. Key corrections to the current module (summary)

| # | Current output (examples) | Standard Slovak | Src | Conf. |
|---|---|---|---|---|
| 1 | 21 m. G `dvadsiatehojedného`, f. `dvadsaťjedna`, n. `dvadsaťjedno`, A `dvadsaťjednu/-jedného` | **`dvadsaťjeden` in every case and gender.** Numerals ending in *-jeden* never decline and always use the form *jeden*. The noun is G pl in N/A, otherwise in its own case. | PAL, MSJ p. 326, NAV | **H** |
| 2 | 22/25/99 oblique `dvadsiatehodruhého`, `dvadsiatympiatym`, `deväťdesiatehodeviateho` | These are **ordinal** forms. Correct is either **undeclined** `dvadsaťdva/dvadsaťpäť` or **both parts declined as cardinals** `dvadsiatich dvoch`, `dvadsiatim piatim`, `dvadsiatimi piatimi`, written as two words | MSJ p. 326, NAV, PAL, PSP 91 via JAR | **H** |
| 3 | f./n. 22 `dvadsaťdve` | `dvadsaťdva` for all genders. In compounds of tens with 1 or 2, the unit takes the masculine form. | MSJ p. 326, JAR p. 223 | **H** |
| 4 | 100 `stého/stému/stým/stom`, f. `stá/stej/stú/stou`, n. `sté`; 200 `dvestá/dvesté`, etc. | **`sto`, `dvesto`, …, `deväťsto` are invariable** with counted nouns, in every case and gender. The listed forms belong to the ordinal *stý*. | BEL, NAV, MSJ p. 317 | **H** |
| 5 | 1000 `tisíceho/tisícemu/tisícim/tisícom`, f. `tisícej/tisícou` | **`tisíc` is normally invariable** with nouns. A rarer variant declines like *päť*: `tisícich, tisícim, tisícimi`. `tisíceho`/`tisícemu` are ordinal forms. | BEL, NAV | **H** |
| 6 | 2000, 5000, 21000, 100000 oblique `dvetisíceho`, … | **Invariable**: `dvetisíc`, `päťtisíc`, `dvadsaťjedentisíc`, `stotisíc`. Compounds ending in *-tisíc* never decline. | NAV | **H** |
| 7 | millions `miliónteho`, `dvamiliónteho`, `päťmilióntym`; `miliardtého` | *milión* is a **noun (vzor dub)** and *miliarda* a **noun (vzor žena)**. Correct: `dvoch miliónov`, `dvoma miliónmi`, `piatim miliónom`, `dvoch miliárd`, … The `-ónt-`/`-rdt-` forms are ordinals. | NAV, BEL | **H** (rule); M (each derived form) |
| 8 | `dvamilióny`, `päťmiliónov`, `dvemiliardy` written as one word | **Separate words**: `dva milióny`, `päť miliónov`, `dve miliardy` | JAR p. 221 fn. 18 | **H** |
| 9 | 101 f. `stájedna`, 121 f. `stádvadsaťjedna`, n. `stéjedno` | `stojeden` and `stodvadsaťjeden` for all genders and cases | NAV (rule on *-jeden*), BEL | M–H |
| 10 | 2 f. I `dvomi`; 3 f. I `tromi`, but m./n. `dvoma`/`troma` | The instrumental does **not depend on gender**. Use `dvoma`, `troma`, `štyrmi` everywhere. `dvomi`/`tromi` are accepted variants today but were formerly judged colloquial or wrong. | MSJ pp. 320–323, POV94, MAS96, SNK11, SKA | H (dvoma/troma are correct); M (variant status) |
| 11 | m. A of 3, 4 = `troch/štyroch` while 2 gives `dva`; no `dvaja/traja/štyria` | These forms depend on **masculine personal** nouns (persons, not animals). The API needs a flag: N `dvaja/traja/štyria`, A = G (`dvoch/troch/štyroch`). Otherwise use `dva/tri/štyri`. For *jeden*, A = G applies to all animates (including animals). | MSJ pp. 318–323 | **H** |
| 12 | 1 m. A `jedného` always | `jedného` only for animates. Inanimates take `jeden`. | MSJ p. 318 | **H** |
| 13 | 0 `nula` in all cases | *nula* is a noun of vzor žena: `nuly, nule, nulu, nule, nulou` (for example *od nuly*, *pod nulou*) | EDU; [INF] regular vzor žena | M |
| 14 | ordinal 101 `stýprvý`, 121 `stýdvadsiatyprvý`, 1001 `tisíciprvý`, 2024 `dvojtisícidvadsiatyštvrtý` | In compound ordinals the **thousands and hundreds stay cardinal prefixes**; only the tens and units are ordinal: `stoprvý`, `stodvadsiaty prvý`, `tisícprvý`, `dvetisícdvadsiaty štvrtý` | PSP 91 p. 44, PSP 98 p. 24 (via JAR) | **H** |
| 15 | ordinals 21–99 written as one word (`dvadsiatyprvý`) | Two words: `dvadsiaty prvý`, genitive `dvadsiateho prvého` | BEL, JAR | **H** |
| 16 | ordinal 1000th f. `tisícia/tisíciu`, n. `tisície` | Rhythmic shortening applies: `tisíca`, `tisícu`, `tisíce`. The SSSJ entry reads "dvojtisíci -ca -ce čísl. radová … dvojtisíce jubileum, výročie". | SRA p. 276 (quoting SSSJ 2006), EDU | **H** (-tisíci f./n.); M (other cases) |
| 17 | ordinal 1,000,000th m. `milióny` (this is the N pl of the noun!), `milióntého` | `miliónty`, G `miliónteho`, D `milióntemu`, I `milióntym`, L `milióntom` (short *-e-* after long *ó*) | USE; SRA p. 281 (*dvojmiliónty*); [INF] rhythmic law | M |
| 18 | decimals `jeden celých päť`, `dva celých päť`, `tri celých štrnásť` | `jedna celá päť desatín`, `dve celé päť desatín`, `tri celé štrnásť stotín`. For 0 and 5+: `nula celých`, `päť celých`. | DUCH, HS | **H** |

**Pronunciation note for TTS [INF, general orthoepy]:** the cardinal *-tich* is read [-ťix] and the ordinal *-tych* is read [-tix]. So *piatich* and *piatych*, or *dvadsiatich* and *dvadsiatych*, sound different, and mixing up the cardinal and ordinal series is audible.

---

## Q1. Numerals 1–4

**Rule.** *jeden* agrees in gender, number and case (BEL: "zhoduje s pomenúvanou vecou v rode, čísle i v páde"). *dva, tri, štyri* have special forms for **masculine personal** nouns (MSJ labels them "muž. os."), with N *dvaja/traja/štyria* and A = G. All other nouns take *dva/dve, tri, štyri*, and this includes animals ("muž. neos."). The oblique cases are the same for all genders.

### jeden (MSJ p. 318; BEL) — **H**
| | m. animate | m. inanimate | f. | n. |
|---|---|---|---|---|
| N | jeden | jeden | jedna | jedno |
| G | jedného | jedného | jednej | jedného |
| D | jednému | jednému | jednej | jednému |
| A | **jedného** | **jeden** | jednu | jedno |
| L | (o) jednom | jednom | jednej | jednom |
| I | jedným | jedným | jednou | jedným |

MSJ p. 318 gives A as "jedného (živ.) / jeden (neživ.)". The tool's extraction of the rest of the MSJ table was garbled. The forms above are the standard ones, and the module already produces them.

### dva (MSJ p. 320) — **H**
| | m. personal | m. non-personal | f. | n. |
|---|---|---|---|---|
| N | **dvaja** | dva | dve | dve |
| G | dvoch | dvoch | dvoch | dvoch |
| D | dvom | dvom | dvom | dvom |
| A | **dvoch** | dva | dve | dve |
| L | (o) dvoch | dvoch | dvoch | dvoch |
| I | **dvoma** (var. dvomi) | dvoma | dvoma | dvoma |

### tri (MSJ p. 322) — **H**
| | m. personal | other |
|---|---|---|
| N | **traja** | tri |
| G / L | troch | troch |
| D | trom | trom |
| A | **troch** | tri |
| I | **troma** (var. tromi) | troma |

### štyri (MSJ p. 323) — **H**
| | m. personal | other |
|---|---|---|
| N | **štyria** | štyri |
| G / L | štyroch | štyroch |
| D | štyrom | štyrom |
| A | **štyroch** | štyri |
| I | **štyrmi** (only form) | štyrmi |

**Instrumental variants: sources disagree over time.**
- MSJ p. 320: "V siedmom páde má číslovka dva tvar dvoma…"; "Tvar dvomi … má hovorový ráz." MSJ p. 322: "popri tvare troma je zriedkavejší variant tromi." MSJ p. 323: "spisovný tvar štyrmi …; podoba štyrma je nárečová."
- POV94 calls dvomi a "chybný tvar … dvomi namiesto náležitého tvaru dvoma". MAS96 corrects "s dvomi" to "dvoma".
- SNK11 (JÚĽŠ, 2011) lists "dvoma – dvomi, troma – tromi" and "obidvoma – obidvomi" as pairs where "obe [prípony sú] správne". SKA, which cites PSP and KSSJ, says "akceptované obe podoby – dvoma aj dvomi". STVR aired a 2016 spot titled "S dvoma aj dvomi moderátormi" (audio not read).
- **Recommendation:** `dvoma`, `troma`, `štyrmi` for all genders. These were never judged wrong. dvomi and tromi appear to be accepted variants in the current codification, but I could not verify that in KSSJ/PSP directly. The instrumental is **never gender-dependent**, so the module's feminine `dvomi`/`tromi` has no basis. — H for the recommendation; M for the variant status.

---

## Q2. Numerals 5–99 (simple: 5–20 and round tens)

**Rule.** MSJ p. 324: "Podľa vzoru päť sa skloňujú číslovky od päť po deväťdesiatdeväť." For masculine personal nouns, N and A have optional congruent forms: "N. päť, muž. os. rod aj piati … A. päť, muž. os. rod aj piatich". MSJ gives "piati chlapci / päť chlapcov" as alternatives. So **A animate = G is optional** (päť or piatich chlapcov). — **H**

### Paradigm päť (MSJ p. 324) — **H**
| N | G | D | A | L | I |
|---|---|---|---|---|---|
| päť (m. pers. also **piati**) | piatich | piatim | päť (m. pers. also **piatich**) | (o) piatich | piatimi |

### Stems
| n | N | m. pers. N | G = L | D | I | Src / Conf. |
|---|---|---|---|---|---|---|
| 5 | päť | piati | piatich | piatim | piatimi | MSJ — H |
| 6 | šesť | šiesti | šiestich | šiestim | šiestimi | MSJ — H |
| 7 | sedem | **siedmi** | **siedmich** | siedmim | siedmimi | MSJ, POV18 — H |
| 8 | osem | **ôsmi** | **ôsmich** | ôsmim | ôsmimi | MSJ, POV18 — H |
| 9 | deväť | deviati | deviatich | deviatim | deviatimi | MSJ — H |
| 10 | desať | desiati | desiatich | desiatim | desiatimi | MSJ — H |
| 11–19 | jedenásť … devätnásť | jedenásti … [INF] | jedenástich … sedemnástich, osemnástich | -ástim | -ástimi | MSJ rule; forms [INF] — M–H |
| 20 | dvadsať | dvadsiati | dvadsiatich | dvadsiatim | dvadsiatimi | MSJ rule; NAV "dvadsiati piati žiaci" — H |
| 30, 40 | tridsať, štyridsať | tridsiati, štyridsiati [INF] | tridsiatich, štyridsiatich | -iatim | -iatimi | MSJ rule — H (forms), M (m. pers.) |
| 50, 60, 90 | päťdesiat, šesťdesiat, deväťdesiat | päťdesiati … [INF] | päťdesiatich … | -iatim | -iatimi | MSJ rule — H |
| 70, 80 | sedemdesiat, osemdesiat | [INF] sedemdesiati | **sedemdesiatich, osemdesiatich** | sedemdesiatim, osemdesiatim | -iatimi | POV18 — H |

- The *siedm-* and *ôsm-* stems occur **only** in simple 7 and 8. Teens and tens keep *sedem-* and *osem-*: sedemnástich, osemdesiatich (POV18). I found no source listing *sedmich* or *osmich* as standard.
- POV18 marks *siedmych*, *ôsmych* and *osemdesiatych* as errors **when used as cardinals**. Those *-ych* forms belong to ordinals.
- Every source I read gives only declined forms for simple 5–99 in oblique cases. None endorses undeclined *pred päť rokmi*. — M

---

## Q3. Compounds 21–99

**Rules (all H):**
1. **Units 1 (21, 31, … 91):** "číslovky dvadsaťjeden, tridsaťjeden, štyridsaťjeden… sa nikdy neskloňujú" (PAL). "Nikdy sa neskloňujú … zložené základné číslovky, ktoré obsahujú na konci komponent -jeden" (NAV). MSJ p. 326: "Číslovky typu dvadsaťjeden … sú nesklonné."
2. **Gender:** "Pri čítaní jednotiek sa používa len číslovka jeden, a to aj v spojení s pomenovaním ženského alebo stredného rodu, napr. 21 žien = dvadsaťjeden žien" (PAL). MSJ p. 326 extends this to 2: "Číslovky, ktoré sa skladajú z desiatok a jednej alebo dvoch jednotiek, majú bez ohľadu na rod … podobu dvadsaťjeden, … dvadsaťdva, tridsaťdva atď. (číslovka jeden a dva má pri všetkých podstatných menách mužský tvar)." JAR p. 223 agrees: "dvadsaťdva – pri všetkých substantívach základný tvar (porov. dvadsaťdva žien, detí a dve ženy, deti)."
3. **Noun agreement:** "V nominatíve a akuzatíve má určovaný predmet tvar genitívu plurálu, napr. 21 chlapov = dvadsaťjeden chlapov" (PAL). In other cases the noun takes that case: "o 41 ženách = o štyridsaťjeden ženách, so 61 hosťami = so šesťdesiatjeden hosťami" (PAL). MSJ p. 326: "Meno číselne určenej veci sa pri nich skloňuje ako pri číslovke päť, napr. štyridsaťjeden družstiev, zo štyridsaťjeden družstiev, ku štyridsaťjeden družstvám…"
4. **Units 2–9:** the compound "sa alebo neskloňujú (dvadsaťdva žiakov, dvadsaťdva žiakom, dvadsaťdva žiakoch, s dvadsaťdva žiakmi), alebo sa skloňujú v oboch častiach" (MSJ p. 326). Examples: "dvadsiatich dvoch žiakov, dvadsiatim dvom žiakom, s dvadsiatimi dvoma žiakmi" (MSJ); "o dvadsaťdva stromoch alebo o dvadsiatich dvoch stromoch"; "pre štyridsaťpäť žiakov alebo pre štyridsiatich piatich žiakov" (PAL); "dvadsaťpäť žiakov" or "dvadsiati piati žiaci" (NAV).
5. **Masculine personal:** 21 has "bez kongruenčnej kategórie mužskej osoby" (JAR p. 223), so only *dvadsaťjeden žiakov*. For 22–99 the declined congruent N is *dvadsiati dvaja žiaci* (NAV pattern; PSP 91 example "stodvadsiati štyria muži").
6. **Spelling:** the nominative is one word (*dvadsaťpäť*). Declined forms write the units **apart**. PSP 1991, p. 44 (via JAR): "V číslovkách od 21 vyššie píšeme oddelene od ostatnej časti jednotky: a) pri skloňovaní základných čísloviek, napr. stodvadsiati štyria muži, stodvadsiatich štyroch mužov." So *dvadsiatich piatich* is written as two words.
7. **jedenadvadsať type:** the SSSJ (2011) labels it "hovorová a zastarávajúca" (PAL). It declines as a single word: "jedenadvadsiatich rokoch" (TASR18). **Do not generate it.**

### Forms
Where two forms appear, the undeclined and the declined alternatives are both standard.

| n | N (all genders) | G | D | A | L | I |
|---|---|---|---|---|---|---|
| 21 | dvadsaťjeden | dvadsaťjeden | dvadsaťjeden | dvadsaťjeden | dvadsaťjeden | dvadsaťjeden |
| 22 | dvadsaťdva (m. pers. alt. dvadsiati dvaja) | dvadsiatich dvoch / dvadsaťdva | dvadsiatim dvom / dvadsaťdva | dvadsaťdva (m. pers. alt. dvadsiatich dvoch) | dvadsiatich dvoch / dvadsaťdva | dvadsiatimi dvoma / dvadsaťdva |
| 25 | dvadsaťpäť (m. pers. alt. dvadsiati piati) | dvadsiatich piatich / dvadsaťpäť | dvadsiatim piatim / dvadsaťpäť | dvadsaťpäť (m. pers. alt. dvadsiatich piatich) | dvadsiatich piatich / dvadsaťpäť | dvadsiatimi piatimi / dvadsaťpäť |
| 99 | deväťdesiatdeväť (m. pers. alt. deväťdesiati deviati) | deväťdesiatich deviatich / … | deväťdesiatim deviatim / … | deväťdesiatdeväť (alt. deväťdesiatich deviatich) | deväťdesiatich deviatich / … | deväťdesiatimi deviatimi / … |

- Confidence: rules H. The 99 row and the teens-plus-units pattern are M, [INF] by applying the rule.
- **TTS default [INF]:** when the case is known, decline both parts, since the sources give it as the more explicit form. Keep undeclined as a configurable alternative. Never produce ordinal-looking forms (*dvadsiateho*, *druhého*, *piateho*) for cardinals.

---

## Q4. 100 and 200–900; compounds above 100

**Rule (H):**
- "Číslovky sto a tisíc v spojení s počítaným predmetom sa neskloňujú (pre sto občanov, o sto občanoch)" (BEL).
- "Číslovka sto sa alebo neskloňuje, alebo sa správa ako podstatné meno a skloňuje sa podľa vzoru mesto." With nouns it is undeclined: "sto, dvesto, päťsto, viac sto, niekoľko sto robotníkov". It declines only in arithmetic: "Číslovka sto sa skloňuje iba v počtových úkonoch, t. j. v absolútnom postavení pri násobení a delení: násobiť stom, deliť piatimi stami." (NAV)
- MSJ p. 317: sto, tisíc, milión and miliarda "majú gramatickú povahu podstatného mena"; with masculine personal nouns they "kongruentné tvary obyčajne nemajú, napr. sto členov, tisíc učiteľov, milión obyvateľov". The congruent form is "len veľmi zriedka … stí (tisíci) ľudia".

**Are *stého/stému/stým/stom/stá/stej/stú/stou* ever cardinal?** No. They are the ordinal *stý* in all its genders. The exceptions: *stom* also occurs as the instrumental singular of the noun *sto* in arithmetic (*násobiť stom*, NAV), and *stá* is also the N/A plural of the noun *sto* [INF]. **The module's feminine "stá" and neuter "sté" for 100 are wrong.** — H

| n | all cases, all genders (counting use) | Src / Conf. |
|---|---|---|
| 100 | sto | BEL, NAV — H |
| 200–900 | dvesto, tristo, štyristo, päťsto, šesťsto, sedemsto, osemsto, deväťsto | NAV ("dvesto, päťsto … robotníkov") — H |
| 101 | stojeden (the noun is G pl in N/A: *stojeden žien*) | NAV rule on *-jeden* — M–H |
| 121 | stodvadsaťjeden | NAV rule — M–H |
| 102 | N stodva; G *stodvoch* (JAR p. 217 reading) or undeclined. No source covers the gender of a bare unit 2 after hundreds with f./n. nouns (*stodva kníh* vs *stodve knihy*); the MSJ masculine-form rule is stated only for tens + unit. | JAR — M (gender: L) |
| 345 | N tristoštyridsaťpäť; G/L tristo štyridsiatich piatich; D tristo štyridsiatim piatim; I tristo štyridsiatimi piatimi; or undeclined throughout | pattern below — M–H |

Evidence for 3-digit compounds: only the tens and units decline, and *sto/dvesto* stay as they are. PSP 91 (via JAR): "stodvadsiatich štyroch mužov". TASR18: "s dvestoštyridsiatimi piatimi lôžkami", "tristopäťdesiatim dvom prieskumníkom". Codified spelling joins the hundreds and tens and separates the units: *tristoštyridsiatich piatich*.

The noun *sto* (vzor mesto; arithmetic only): the forms *stom* (I sg) and *stami* (I pl) are sourced (NAV). By pattern [INF] the rest are *sta, stu, ste; pl. stá, stám, stách*. — M

---

## Q5. 1000 and 2000–999 000

**Rule (H):**
- With counted nouns, *tisíc* is normally undeclined: "tisíc ľudí (ľuďom, ľuďoch, ľuďmi), alebo sa – čo je zriedkavejšie – skloňuje ako číslovka päť" (NAV), with the examples "tisíci ľudia, k tisícim ľuďom, od tisícich ľudí, s tisícimi ľuďmi". BEL gives only the undeclined use.
- Used on its own (as a noun), it follows vzor stroj: "Číslovka tisíc sa skloňuje ako podstatné meno, a to podľa vzoru stroj, keď stojí samostatne", for example "násobiť tisícom, deliť tisícmi/tisícami" (NAV).
- "Zložené číslovky s komponentom -tisíc sa neskloňujú, napr. tritisíc robotníkov, od tritisíc robotníkov, k tritisíc robotníkom atď." (NAV)
- *tisíceho/tisícemu/tisícej/tisícou* are **never** cardinal. They belong to the ordinal *tisíci* (Q8).

| | N | G | D | A | L | I | Conf. |
|---|---|---|---|---|---|---|---|
| 1000 + noun (default) | tisíc | tisíc | tisíc | tisíc | tisíc | tisíc | H |
| 1000 + noun (rarer) | tisíc (m. pers. tisíci) | tisícich | tisícim | tisíc | tisícich | tisícimi | H (NAV) |
| 2000, 5000, 21 000, 100 000 | dvetisíc, päťtisíc, dvadsaťjedentisíc, stotisíc (**invariable**) | = N | = N | = N | = N | = N | H |
| *tisíc* noun, sg [INF except I] | tisíc | tisíca | tisícu | tisíc | tisíci | tisícom (NAV) | M |
| *tisíc* noun, pl [INF except I] | tisíce | tisícov | tisícom | tisíce | tisícoch | tisícmi / tisícami (NAV) | M |

- So *s tisíc korunami* is the default, and *s tisícimi korunami* is allowed but rarer (NAV).
- **Spelling:** one word (*dvetisíc*). PSP 98 (via JAR) writes the example "dvetisícdruhý", and PSP 91 p. 44 says "V zložitých prípadoch možno na zvýšenie prehľadnosti oddeľovať tisícky, stovky a desiatky s jednotkami". Thousands may therefore be separated for readability; TASR18 writes "sedemtisíc dvesto osemdesiattri". — H
- **1001 and 1100:** *tisícjeden* (invariable, *-jeden* rule) and *tisícsto* (invariable). — M–H

---

## Q6. milión, miliarda, bilión

**Rule (H):** "Číslovky milión, bilión, trilión majú substantívnu povahu a skloňujú sa podľa vzoru dub; tendencia k neohybnosti je pri nich menšia ako pri číslovkách sto a tisíc. Základná číslovka miliarda má takisto substantívnu povahu a skloňuje sa podľa vzoru žena" (NAV). BEL: "milión, miliarda … sa skloňujú ako podstatné mená". **Spelling:** "Vo vyšších zložených číslovkách sa milióny a miliardy píšu vždy oddelene" (JAR p. 221, fn. 18), for example "päť miliónov dvestotridsaťštyritisícpäťstošesťdesiatsedem".

### Noun paradigms (forms follow the stated vzor [INF]; M–H)
| | milión sg | milión pl | miliarda sg | miliarda pl |
|---|---|---|---|---|
| N | milión | milióny | miliarda | miliardy |
| G | milióna | miliónov | miliardy | **miliárd** (usage: "desiatky miliárd", USE) |
| D | miliónu | miliónom | miliarde | miliardám |
| A | milión | milióny | miliardu | miliardy |
| L | (o) milióne | miliónoch | miliarde | miliardách |
| I | miliónom | miliónmi | miliardou | miliardami |

bilión and trilión follow milión (vzor dub); biliarda follows miliarda [INF].

### With counting numerals
Both parts decline. 2–4 take the N/A plural of the noun and 5+ take its G plural. The gender follows the noun: *dva milióny*, *dve miliardy*. The combinations are [INF] from the rules in Q1–Q2 and the vzory above. — M–H

| | 2 000 000 | 5 000 000 | 2 000 000 000 | 5 000 000 000 |
|---|---|---|---|---|
| N | dva milióny | päť miliónov | dve miliardy | päť miliárd |
| G | dvoch miliónov | piatich miliónov | dvoch miliárd | piatich miliárd |
| D | dvom miliónom | piatim miliónom | dvom miliardám | piatim miliardám |
| A | dva milióny | päť miliónov | dve miliardy | päť miliárd |
| L | (o) dvoch miliónoch | piatich miliónoch | dvoch miliardách | piatich miliardách |
| I | dvoma miliónmi | piatimi miliónmi | dvoma miliardami | piatimi miliardami |

- 1 000 000 = *milión*, with G *milióna*, D *miliónu*, L *milióne*, I *miliónom*. 1 000 000 000 = *miliarda*, *miliardy*, …
- A counted noun follows in the G pl: *dva milióny obyvateľov* (MSJ p. 317 pattern: "milión obyvateľov").

---

## Q7. Large mixed numbers in oblique cases

**What the sources support:**
- *sto* and *tisíc* components do not decline (BEL, NAV).
- Numbers ending in *-jeden* never decline (NAV, PAL).
- Otherwise, either leave the whole number undeclined, or decline **only the tens and units**. PSP 91: "stodvadsiatich štyroch mužov". JAR p. 214: "tisícdvestodvadsiatich piatich" / "tisíc dvesto dvadsiatich piatich", noting "Jazyková prax je rozkolísaná". TASR18: "dvestoštyridsiatimi piatimi". — H for numbers below one million.

| 2 345 | recommended declined | undeclined alternative |
|---|---|---|
| G | dvetisíc tristo štyridsiatich piatich (codified spelling: dvetisíctristoštyridsiatich piatich) | dvetisíc tristo štyridsaťpäť |
| I | dvetisíc tristo štyridsiatimi piatimi | dvetisíc tristo štyridsaťpäť |

Confidence M–H: the pattern is attested for 124, 245, 352 and 1225, and extended here by analogy.

- **With milión or miliarda inside** (for example 1 234 567 in the G): **no source found.** [INF, L] Keep the million part in its counting form (*milión* / *dva milióny* / *päť miliónov*) and decline only the final tens and units: *milión dvestotridsaťštyritisíc päťstošesťdesiatich siedmich*. Alternatively leave everything undeclined. **The native reviewer should check this.**
- **TTS spacing [INF]:** emit each component as a separate token (*dvetisíc tristo štyridsiatich piatich*). PSP explicitly allows separating the orders for readability, and separate tokens give better prosody. Spelling does not change pronunciation.

---

## Q8. Ordinals

**Declension types:**
- Hard ordinals follow vzor **pekný**: prvý, druhý, štvrtý, piaty, šiesty, siedmy, ôsmy, deviaty, desiaty, jedenásty…, dvadsiaty…, stý, dvojstý, miliónty. POV18: siedmy and ôsmy "skloňujú sa ako prídavné mená vzoru pekný".
- *tretí* and *tisíci* follow vzor **cudzí**.
- The rhythmic law shortens the ending after a long syllable or diphthong: piaty → *piateho*, and tisíci → *tisíca*.

The paradigm tables below are the standard adjectival ones. Where no source is cited, they are **[INF] from the declension type**. The module's singular forms already match them, except where marked.

### Singular
| | prvý (m./f./n.) | piaty (m./f./n.) | tretí (m./f./n.) | **tisíci** (m./f./n.) | **miliónty** (m./f./n.) |
|---|---|---|---|---|---|
| N | prvý / prvá / prvé | piaty / piata / piate | tretí / tretia / tretie | tisíci / **tisíca** / **tisíce** | miliónty / miliónta / miliónte |
| G | prvého / prvej | piateho / piatej | tretieho / tretej | tisíceho / tisícej | **miliónteho** / milióntej |
| D | prvému / prvej | piatemu / piatej | tretiemu / tretej | tisícemu / tisícej | milióntemu / milióntej |
| A | prvého (anim.), prvý (inan.) / prvú / prvé | piateho, piaty / piatu / piate | tretieho, tretí / tretiu / tretie | tisíceho, tisíci / **tisícu** / tisíce | miliónteho, miliónty / milióntu / miliónte |
| L | prvom / prvej | piatom / piatej | treťom / tretej | tisícom / tisícej | milióntom / milióntej |
| I | prvým / prvou | piatym / piatou | tretím / treťou | tisícim / tisícou | milióntym / milióntou |

### Plural
| | prvý | piaty | tretí | tisíci | siedmy / ôsmy (POV18) |
|---|---|---|---|---|---|
| N m. pers. | **prví** | **piati** | **tretí** | **tisíci** | **siedmi / ôsmi** |
| N other | prvé | piate | tretie | tisíce | siedme / ôsme |
| G = L (A m. pers.) | prvých | piatych | tretích | tisícich | siedmych / ôsmych |
| D | prvým | piatym | tretím | tisícim | siedmym / ôsmym |
| I | prvými | piatymi | tretími | tisícimi | siedmymi / ôsmymi |

Other m. pers. plurals follow the same pattern [INF]: druhí, štvrtí, šiesti, deviati, desiati, jedenásti, dvadsiati, stí.

**tisíci:** SRA p. 276 quotes the SSSJ (A–G, 2006) entry verbatim: "dvojtisíci -ca -ce čísl. radová ▶ majúci v poradí číslo dvetisíc: d. návštevník; d. výrobok; dvojtisíce jubileum, výročie". This gives f. *-tisíca* and n. *-tisíce*, i.e. rhythmically shortened. The corpus collocations "dvojtisíce výročie" (38) and "dvetisíce výročie" (36) (SRA p. 275) agree. EDU gives f. *tisíca, tisícej, tisícu, tisícou*. So the module's *tisícia*, *tisíciu* and *tisície* are wrong. — H for N f./n.; M for the other cases, which are [INF] from the same shortened soft pattern.

### Special ordinals
| Value | Form | Variant | Src | Conf. |
|---|---|---|---|---|
| 100th | stý | — | standard | H |
| 200th | **dvojstý** (codified in KSSJ 2003, PSP 2013, SSSJ 2006: "za jedinú správnu podobu"; the SSSJ entry *dvestý* cross-refers to *dvojstý*). The SSSJ entry reads "dvojstý -tá -té čísl. radová", i.e. the hard pattern, like *stý* | *dvestý*: uncodified but frequent (760 vs 743 corpus hits). SRA argues it should be accepted as a variant | SRA pp. 268–271 | H |
| 300th, 400th | trojstý, štvorstý (given by Ondrus 1969) | SRA marks *tristý and *štyristý with an asterisk as hypothetical | SRA pp. 270, 282 | M (status in KSSJ/PSP not verified) |
| 500th–900th | päťstý, šesťstý, sedemstý, osemstý, deväťstý | — | [INF]; EDU lists *päťstý* | M |
| 1000th | tisíci | — | EDU; SRA | M–H |
| 2000th | **dvojtisíci** (codified; SSSJ) | *dvetisíci*: uncodified, but more frequent (213 vs 101 corpus hits) | SRA pp. 268–273 | H / M |
| 3000th, 4000th | trojtisíci, štvortisíci (Ondrus 1969) | *tritisíci, *štyritisíci are hypothetical; *štyritisíci* would be "priveľmi kakofonické" | SRA pp. 270, 282 | M |
| 5000th+ | päťtisíci, desaťtisíci, stotisíci | — | [INF] | M |
| 1 000 000th | miliónty (hard, shortened: miliónteho, milióntemu…) | — | USE ("miliónty návštevník", "26-miliónteho"); SRA p. 281 names *dvojmiliónty* as an ordinal | M |
| 2 000 000th | dvojmiliónty | — | SRA p. 281 | M |
| 10⁹th | miliardtý (miliardtého: no shortening after a short syllable) | — | only slovensky.eu (unknown source) and [INF] | L–M |

**Module bug:** the ordinal 10⁶ is output as `milióny`, which is the nominative plural of the noun. The feminine and neuter *milióntého/-ému* also carry a wrong long *é*.

---

## Q9. Compound ordinals

**Rules (H):**
- Ordinals 21–99 are two words: "Radové číslovky od 21. do 99. sa píšu ako dve slová (päťdesiaty ôsmy)" (BEL). JAR adds that in the codification they "sa v základnom tvare aj v ohýbaných tvaroch píšu vždy ako dve slová".
- Above 100, PSP 1991, p. 44 (via JAR): the units are written apart "b) v radových číslovkách, napr. stodvadsiaty štvrtý účastník, stodvadsiateho štvrtého účastníka, v tisícdeväťstoosemdesiatom piatom roku".
- PSP 1998, p. 24 (via JAR) added an exception, which does not replace the rule: "Spolu sa píšu radové číslovky zložené z jednotiek, stovák a tisícok, napr. stoprvý, dvetisícdruhý".
- **Hence the thousands and hundreds are cardinal prefixes** (*sto-*, *tristo-*, *tisíc-*, *dvetisíc-*). Only the tens and units are ordinal, and only they decline. JAR p. 217 gives G "stodruhého, tisícdruhého".
- Usage varies (JAR). TASR18 even accepts "päťstodvadsiatyôsmy, ale aj päťsto dvadsiaty ôsmy". Spelling does not affect pronunciation.

| n | N m. / f. / n. | G m.=n. / f. | L m. | Conf. |
|---|---|---|---|---|
| 21. | dvadsiaty prvý / dvadsiata prvá / dvadsiate prvé | dvadsiateho prvého / dvadsiatej prvej | dvadsiatom prvom | H |
| 22. | dvadsiaty druhý / dvadsiata druhá / dvadsiate druhé | dvadsiateho druhého / dvadsiatej druhej | dvadsiatom druhom | H |
| 101. | stoprvý / stoprvá / stoprvé | stoprvého / stoprvej | stoprvom | H (PSP 98) |
| 121. | stodvadsiaty prvý / stodvadsiata prvá / … | stodvadsiateho prvého / stodvadsiatej prvej | stodvadsiatom prvom | H (PSP 91 pattern) |
| 345. | tristoštyridsiaty piaty / tristoštyridsiata piata / … | tristoštyridsiateho piateho / tristoštyridsiatej piatej | tristoštyridsiatom piatom | H (pattern) |
| 1991. | tisícdeväťstodeväťdesiaty prvý / … | tisícdeväťstodeväťdesiateho prvého | (v) tisícdeväťstodeväťdesiatom prvom (roku) | H (PSP 91: "v tisícdeväťstoosemdesiatom piatom roku") |
| 2024. | dvetisícdvadsiaty štvrtý / dvetisícdvadsiata štvrtá / … | dvetisícdvadsiateho štvrtého / dvetisícdvadsiatej štvrtej | dvetisícdvadsiatom štvrtom | H (pattern; "dvetisícdruhý") |
| 1001. / 1100. | tisícprvý / tisícstý | tisícprvého / tisícstého | — | M ([INF] from PSP 98 rule) |

Note on 2024: the bare 2000th is *dvojtisíci*, but inside a compound the prefix is the **cardinal** *dvetisíc-*, as in PSP's own "dvetisícdruhý".

---

## Q10. Decimals

**Rule (H):** "nula celých…, jedna celá…, dve (tri, štyri) celé…, [päť/]osem celých" (DUCH). HS, which credits the JÚĽŠ poradňa, gives the same: "nula celých (jednotiek), jedna celá (jednotka), dve celé (jednotky), tri celé …, päť celých (jednotiek)". The integer part agrees with the feminine *celá*, hence *jedna* and *dve*. **Zero takes *nula celých*.**

The fractional part is read as a number of *desatín / stotín / tisícin*. The optional *a* is shown as "/a/" in DUCH:
- "nula celých (a) dvadsaťpäť stotín" (0,25)
- "jedna celá (a) deväť desatín" (1,9)
- "dve celé (a) osemdesiatštyri stotín" (2,84)
- "41,2 s – štyridsaťjeden celých /a/ dve desatiny sekundy"
- "396,4 g – tristodeväťdesiatšesť celých /a/ štyri desatiny gramu"
- "2,531 % – dve celé /a/ päťstotridsaťjeden tisícin percenta"
- "68,50 € – šesťdesiatosem celých /a/ päť desatín eura" (the trailing zero is dropped)

With a measure noun, the noun takes the **G sg**: "je dané podstatné meno v tvare genitívu jednotného čísla" (DUCH).

| Input | Current | Correct | Conf. |
|---|---|---|---|
| 0,5 | nula celých päť | nula celých (a) päť desatín | H |
| 1,5 | jeden celých päť | **jedna celá** (a) päť desatín | H |
| 2,5 | dva celých päť | **dve celé** (a) päť desatín | H |
| 3,14 | tri celých štrnásť | **tri celé** (a) štrnásť stotín | H (pattern) |
| 5,25 | päť celých dvadsaťpäť | päť celých (a) dvadsaťpäť stotín | H |
| 21,3 | dvadsaťjeden celých tri | dvadsaťjeden celých (a) tri desatiny | H (DUCH "štyridsaťjeden celých … dve desatiny") |
| 22,5 | — | dvadsaťdva celých (a) päť desatín | M ([INF] from the MSJ p. 326 rule on 22) |

**Fraction word:**
- 1 → *jedna desatina / stotina / tisícina*
- 2–4 → *dve (tri, štyri) desatiny* ("dve desatiny", DUCH)
- 5+ and all compounds from 21 up → G pl: *päť desatín*, *osemdesiatštyri stotín*, *päťstotridsaťjeden tisícin* (DUCH) — H

**Not covered by any source I read:**
- Decimals in oblique cases. DUCH gives no example. [INF, L] Keep the nominative reading.
- Whether *stotín* and the like may be omitted ("tri celé štrnásť"). DUCH always includes the fraction word, so include it.
- Reading more than three decimal places.

---

## Q11. nula

*nula* is a feminine noun of vzor žena. **M–H.** Source: EDU (weak) plus the regular vzor žena [INF]. SSSJ (vol. M–N, 2015) was not reachable.

| | sg | pl |
|---|---|---|
| N | nula | nuly |
| G | nuly | **núl** |
| D | nule | nulám |
| A | nulu | nuly |
| L | (o) nule | nulách |
| I | nulou | nulami |

- Use the noun forms in oblique contexts: *od nuly*, *k nule*, *pod nulou* [INF: ordinary noun use].
- In counting use with a noun in the N/A, the form is *nula* + G pl, as in DUCH's *nula celých*. By analogy: *nula stupňov* [INF].
- I found no source for zero plus a counted noun in an oblique case.

---

## Appendix: implementation consequences (for the module author) [INF]

1. Delete `TENS_COMPOUND`, `ONES_COMPOUND` (the adjectival/ordinal forms), the gendered `HUNDREDS` rows, `TISIC_COMPOUND` and `SCALE_T_COMPOUND`.
2. Cardinal generation:
   - A unit of 1 in any compound (21, 101, 1001, …) → the whole number is invariant, and the unit is always *jeden*.
   - Compounds ending in 2–9 → invariant, or [tens-oblique] + [units-oblique], with sto/tisíc prefixes left as they are.
   - Compound units 1 and 2 are the masculine forms (*dva*) regardless of gender.
   - milión and miliarda → a separate count word plus the declined noun.
3. Add a `masc_personal` or `animacy` parameter. It selects dvaja/traja/štyria, A = G, the optional piati form, and jedného vs jeden in the A.
4. Ordinal generation: cardinal prefix for the thousands and hundreds + ordinal tens + a space + the ordinal unit. Build dvojstý/trojstý/štvorstý and dvojtisíci/trojtisíci/štvortisíci only for bare values. Offer dvestý/dvetisíci as an option. Correct the tisíci and miliónty paradigms.
5. Decimals: the integer part takes feminine agreement with *celá/celé/celých*, then the fraction value plus *desatina/stotina/tisícina* in the right number. Strip trailing zeros.
