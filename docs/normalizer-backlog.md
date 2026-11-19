# Normalizer backlog (Prompt 2.2)

Found-not-fixed items of the text normalizer, triaged 2026-11-17. **Kept** items are real work: a crash, a regression,
or a reading an ordinary listener would hear wrong. **Parked** items are nits or diminishing returns, and are built only
when revived. New findings are added to one of the two lists as they are verified.

## Kept

Next batch, queued by the user on 2026-11-18: weekday abbreviations. `Otvorené Po–Pia 8–17 h.` reads "Po— Pia od ôsmej
do sedemnástej" (sk Po, Ut, St, Št, Pi/Pia, So, Ne; cs Po, Út, St, Čt, Pá, So, Ne; "Po" is also the preposition "po"; the
em dash spacing).

Pinned, its own task when taken up: Stanza 1.15.0 (Dependabot #12) changes 563 of 9,818 corpus readings with its new
CAC and SNK models and fails 19 tests, so `pyproject.toml` holds `stanza<1.15`. Upgrading means going through those
readings and retuning the normalizer to the new tags. Its security alert (unrestricted pickle in the lemmatizer
checkpoint loader) does not reach caketts, which loads only the default `tokenize,pos` models (dismissed 2026-11-18).
The FEATURE_FIXES stopgaps work around cs/sk tagger errors; a retune to Stanza 1.15 or a lexicon would replace them.

## Fixed

Numbers chain, 090f7f3..b83ec25 (one commit per step; N41 and N76 together).

| id | input | now |
|---|---|---|
| N42 | `Mám 5 milionů €.` / `5 tisíc m²` / `5 tisíc Kč/kg` | `pět milionů eur` / `metrů čtverečních` / `korun za kilogram` |
| N43 | `Přijdu v 0 h.` / `S 0 body skončil.` | `v nula hodin` / `S nula body` |
| N41 + N76 | `Dosáhl 5 bodů, 10 bodů nebo 15 bodů.` | all genitive; `2 muži a 5 žen dosáhli cíle.` stays nominative |
| N5 | `Dosáhl 5, 10 nebo 15 bodů.` | `pěti, deseti nebo patnácti` |
| N6 | `Dosáhl 1 nebo 2 bodů.` | `jednoho nebo dvou` |
| N39 | `Stál mezi 1 a 2 stromy.` | `mezi jedním a dvěma` |
| N44 | `Trvalo to 1 nebo 2 hodiny.` | `jednu nebo dvě` |
| — | a list of n items cost O(n³) after the list commits (200 noun phrases: 26 s) | 0.36 s (b83ec25) |

Slovak chain, c0dbec5..4910515.

| id | input | now |
|---|---|---|
| N22 | sk `Týka sa to 5 ľudí.` / `Bál sa 5 psov.` | `piatich` |
| N29 | sk `Získal 2 body.` | `dva body` |
| N32 | `Zúčastňoval se 5 závodů.` / `Neboj se 5 psů.` | `pěti` |
| — | sk `Zúčastnili sa 60 % voličov.` / `Vzdal sa 5 krát.` / `Zúčastnili sa 5 žien.` (regressions of the chain) | read or logged as the Czech twins |

Preps and misc chain, 4c7d398..7450008.

| id | input | now |
|---|---|---|
| N53 | `Záleží na 5 %.` / `Po 5 km zastavil.` | `na pěti procentech` / `Po pěti kilometrech` |
| N74 | `Trvá 1 nebo 2 hodiny.` | `jednu nebo dvě` |
| N58 | `Patřilo to Václavu II. Přemyslovci.` | `Václavu druhému Přemyslovci` (no break) |
| N71 | sk `Mám 5 miliónov €.` | `päť miliónov eur` |
| N68 | sk `Prídem o 1:00.` | `o jednej` |
| N65 | sk `Otvorené od 8 do 17 h.` | `od ôsmej do sedemnástej`, logged (user's rule: an afternoon hour, or the `od` of a span ending at one) |
| — | `Zůstala mu faktura na 5 Kč.` / `Zůstalo mu jen na 5 Kč.` (regressions of N53a) | accusative again |
| N91 | `book_config["num2words"]` with a bad value (`"agreemnt"`, `"yes"` for a flag) | ValueError (89f90db) |

Found-not-fixed follow-ups, 95ca4dc..91c63c3.

| id | input | now |
|---|---|---|
| — | `Jednání trvalo 1 nebo 2 hodiny.` / `Film trvá 1 nebo 2 hodiny.` | `jednu nebo dvě` |
| — | sk `Otvorené 8–17 h.` / `Otvorené od 8–17 h.` | `od ôsmej do sedemnástej`, logged (user 2026-11-17) |

Batch 5 hours chain, de10052..5e5f572.

| id | input | now |
|---|---|---|
| N61 | `v 8.30, asi 9.30 a 10.30 hod.` | `v osm třicet, asi devět třicet a deset třicet` (a dotted time in a list or before an approximator is a time) |
| — | sk `Otvorené 8.30–17 h.` / cs `8–17:30` | `od ôsmej tridsať do sedemnástej` / `osm až sedmnáct třicet` |
| N38 + N69 + sk `o 17 h` | sk `Prídem o 5 h.` / `o 5 alebo 6 h` | `o piatej` / `o piatej alebo šiestej`, logged (also in lists and dash spans) |
| N38 | sk `o 2 h neskôr` / `Posunuli to o 1 h` / `Prosil o 2 h` / `Meškanie vlaku je o 2 h` / `o 2 h času` / `Po 20 h jazdy` | stay a span (comparative, word of change or request, copular subject, genitive noun after the unit, minutes or `a pol` after the hours) |
| N65 | sk `Otvorené 8:30–17:00.` / `od 9 do 12 h a od 13 do 17 h` | `od ôsmej tridsať do sedemnástej`, logged / hours of the day for both spans |
| N65 | sk `o 17 h dňa 5. júna` / `denne`, `týždenne`, `mesačne`, `ročne`, dňa, roka, týždňa, mesiaca, dopoludnia, popoludnia, leta, and the Czech weekday genitives | `o sedemnástej` (the hour is kept) |
| — | sk `Toho dňa sa bojí 5 psov.` (parked 2026-11-17: no Slovak time genitives) | `piatich psov`, logged as its Czech twin |
| — | `kolem 13. h` | `kolem třinácté hodiny` (an ordinal before `h` or `hod.` is the Nth hour) |
| N81 | `Ve 24 h provozu` | `Ve dvaceti čtyřech hodinách provozu` (hours of operation) |

Batch 5 prepositions chain, 5e5f572..2b2a92a.

| id | input | now |
|---|---|---|
| N88 | `Po 30 min` / `Voda sahala po 50 cm` | `Po třiceti minutách` / `po padesát centimetrů` (accusative in `od ... po`, after `až`, and after a word of reaching, depth or height) |
| — | `Měří od 1 m po 5 m` / `Dostal od něj po 5 Kč` | `od jednoho metru po pět metrů` / `po pěti korunách` |
| N83 | `Dohodli se na 5 %` / `Shodli se na 50 Kč` / `Trval na 5 %` (also bydlet, ležet, žít, zůstat; sk zostať, ostať, zotrvať) | `na pěti procentech` / `na padesáti korunách`; a time unit stays the accusative |
| N84 | `Jednali o 5 %` / `Mluvili o 5 %` | `o pěti procentech` (the locative of a topic after a Czech verb or noun of speaking and writing) |
| N55 | `Zpoždění o 2 h` | `o dvě hodiny` (the accusative of a difference after a word of change or before a comparative) |
| N86 | sk `Zostala na 5 %.` / `Trvalo to 1 h.` | `na piatich percentách` / `jednu hodinu` |
| — | `Mluvil 1 h a čekal 2 h.` / `Uplynul 1 h.` | `jednu hodinu` ... `dvě hodiny` / `jedna hodina` (the logged nominative after a verb of time passing) |
| — | `Po 10 s začalo pršet.` / `s 5 lidmi` | `Po deseti sekundách` / unchanged (a bare `s` is seconds unless an instrumental follows; logged when a noun, name or adjective follows; `sek` is a unit) |

Batch 5 Slovak forms chain, 2b2a92a..de7a8e5.

| id | input | now |
|---|---|---|
| N30 | sk `Prišli 5 muži.` / `Prišli 22 študenti.` | `piati muži` / `dvadsiati dvaja študenti`; with the num2words agreement construction the subject of a neuter verb reads `päť mužov` (`Prišlo 5 mužov.`), not `piatich` |
| N63 (stopgap, user 2026-11-18) | sk `Má 2 hrady.`; `pravnuk`, `prapravnuk`, `pradědeček`, `prastrýc` | `dva hrady` (FEATURE_FIXES: 87 Slovak plurals the tagger gives the other gender, and the kinship forms) |
| N64 | sk `Prišlo 0 ľudí.` / `Videl 0 mužov.` / `s 0 bodmi` / `bez 0 bodů` | `nula ľudí` / `nula mužov` / `s nula bodmi` / `bez nula bodů` (zero before a counted noun or unit reads `nula` after any preposition; a bare zero is declined: `klesla na nulu`) |
| — | sk `Dočkal sa 5 vnúčat a 3 pravnúčat.` / `Má 1 pravnuka` | `piatich vnúčat a troch pravnúčat` / `jednoho pravnuka` |
| — | `Viděl ho 2 krát.` / `Za 0 tis. Kč` | `dva krát` / `Za nula korun` |
| — | `Dotkl se 2. prstem stolu.` | `druhým prstem` (an ordinal agrees with its own instrumental noun after a genitive verb) |

Batch 5 Roman numerals and capitals chain, de7a8e5..7237fd5.

| id | input | now |
|---|---|---|
| N58 | `Patřilo to Václavu II. Přemyslovci i jeho synovi.` / `Vzpomínal na Karla IV. Karel IV. byl.` / `Karlu IV. Dům.` | `Václavu druhému Přemyslovci` (no break) / a new sentence after the name / a new sentence |
| N82 | `Viz I. Jan 3,16.` with `verse_references` | `první Jan tři, šestnáct` (a Roman numeral from I to the book's top number with its period before a numbered book) |
| N59 | `V KNIZE II. DÍL` | `druhý DÍL` (after kniha, díl, kapitola, část, svazek, oddíl, ročník, sešit) |
| N89 | `PŘIŠLO 5 TIS. LIDÍ.` / `DNE 5. KVĚTNA` / `KAREL V. BYL KORUNOVÁN.` / `HOD.` | `pět tisíc LIDÍ` (no sentence end before the counted noun, never after `ATD.`/`APOD.`) / `pátého` / `pátý`, logged / works in capitals |
| — | `v 5. stol.` / `V 5. STOL. PŘ. N. L.` | `v pátém století` (instrumental `stoletím`) / also with the era |

Closed without a change: `Uplynulo 1 nebo 2 hodiny.` (ungrammatical; `Uplynuly 2 hodiny` reads right), `Prišli 5 ľudia`
(ungrammatical; read as written), Slovak hundreds in oblique cases stay undeclined (`o päťsto ľuďoch`, user 2026-11-18).

## Parked

- Genitive-verb subject or object, rare or log-only: N7 (`Dosáhl 2 vítězství` dvě; with N77, its regression), N12
  (`Dosáhl 5 tisíc bodů` pět), N21 (past `Zúčastnilo se 5 závodů`), N27 (ordinal + non-genitive noun), N49 (`Zúčastní se 5
  soutěží` pět: tagger), N51 (`Dítě se bálo 5 psů`), N57 (sentence-initial imperatives), N75 (ordinal after a mistagged
  noun), N78 (`sdružení` takes part), N79 (`Dosáhlo 2 %`), N80 (noise logs), N90 (sk `Bál sa 5 miliónov ľudí`), N94–N99
  (twins and trade-offs of N7b/N77/N79).
- Slovak edges: N31 (`100. narodenín`), N66 (`Oslávil 2. Vianoce`), N67 (collective numerals `dvoje dveře`), N70
  (plural-ordinal log noise).
- Prepositions and lists: N40 (year shares the next noun across a comma), N54 (walk stops at an abbreviation), N56 (amount
  with its own genitive noun opening a clause), N60 + N85 (approximator mid-chain with a counted noun), N72 (approximator
  inside a list), N73 (abbreviation ends a list).
- Config nits: N62 (unclear error when the num2words config is not a dict), N92 (agreement option `dvacet jedno jablek`).
- Found by the numbers chain (2026-11-17): `Dosáhl 5 tisíc Kč` pět (needs N12A); `Dosáhl 1,2,3 nebo 5 bodů` jedna, dva,
  tři nebo pěti; `Bál se 5, 10 psů` pěti, deset (a lone comma pair is no list, by design); `Dosáhl 2,5 nebo 5 bodů`
  decimal nominative; `Pod 5, 10 nebo 15 stromy` (two-case preposition, no verb); `Dosáhl 3 gólů, 2 asistencí` dvě (N49
  family); lists log once per number (log volume only, N5A trade-off).
- Found by the Slovak chain (2026-11-17): sk `Zúčastní sa 5 pretekov` päť + log (no Slovak event list; N21 twin);
  `Bál sa 2 stretnutí` / `Bál se 2 setkání` dve/dvě (N7 family); sk `Zúčastnilo sa 2 alebo 50 ľudí` dvoch (personal
  masculine A = G); sk `Vzdal sa 5 percent mzdy` päť (SNK tags `percent` singular).
- Batch 5 (2026-11-19), meaning decides, logged: sk `Pracoval po 14 h denne`, `Cesta trvala do 20 h`, `Do 14, 15 a 16 h`,
  `Hovorili o 5 h.`, `Ludvík XIV. Slunce vládl dlouho.`.
- Batch 5, needs a lexicon: the N63 remainder (sk `kone`, `elementy`, `obyčaje`, `medze`; `snímky` is left to the tagger:
  snímok and snímka), topic nouns before "o" (`Psal knihu o 5 %`), caps `PATŘILO TO VÁCLAVU II. PŘEMYSLOVCI.`,
  agreement-construction present tense (`Prichádza 5 mužov`, `To bolo 5 mužov`).
- Batch 5, needs a restructure: a bare "s" before a pronoun or adverb (`Po 10 s mu odpověděl.`, `Po 10 s všichni
  odešli.`; the unit is decided in a regex before tagging).
- Batch 5, rare, contrived, or wrong before and after: `Mají dům na 200 m²`, `Zkrátili zprávu o 2 h`, sk `Dostal od
  neho po 5 eur.`, `Zúčastnil se 2. kolem.`, `Dotkli se 2. prsty stolu.`, `Se 13. stol. to bylo jinak.`, `V 5 stol.`,
  `Dal to Karlu IV. Muži i jeho synovi.`, `Dal to Karlu IV. Jan.`, verse config `Karel I. Jan 3,16 byl.`, sk `v 5. stor.`,
  caps `KAREL V.` at a sentence end, `ŽIL VE 3. STOL. PŘ. N. L. V ŘÍMĚ.`, `V 5. STOL. N. L. VLÁDL ŘÍM.`, caps `8.30 H.`,
  `PŘIŠEL V 5 HOD. RÁNO.`, `OD 8.30 HOD. AŽ 9.30 HOD.`, `PRACOVAL S 2 H. A 3 H. ZŮSTALY MU.`, `Po 18 h 30 min odišiel.`,
  `Po 20 h chôdze`, `Otvorené od 13 do 17 h a od 9 do 12 h.`, `Otvorené od 9 do 12 a od 13 do 17:00.`, `Otvorené 13 do 17
  h.`, `Prídem o 9,10,11 h.`, `Medzi 13. h a 15. h.`, `Skóre 2:30–3:15.` (logged), other prastrýc forms, sk `Prišli 21
  učitelia.`, `Ráno prišli na stretnutie 5 muži z mesta.`, cs `Při 0 V napětí.`, `Skončili blízko 0.`, sk `Pri 0:00
  hodinách.`, cs `Dosáhl 2. cíle.`.
