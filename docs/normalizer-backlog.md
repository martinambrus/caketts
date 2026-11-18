# Normalizer backlog (Prompt 2.2)

Found-not-fixed items of the text normalizer, triaged 2026-11-17. **Kept** items are real work: a crash, a regression,
or a reading an ordinary listener would hear wrong. **Parked** items are nits or diminishing returns, and are built only
when revived. New findings are added to one of the two lists as they are verified.

## Kept

Batch 5, revived from Parked on 2026-11-18 (one commit per item on `normalizer-limits-6`). An item that already reads
right is closed with its evidence, and one whose fix needs a lexicon or risks wide regressions goes back to Parked with
its reason.

- Hours and times: sk `Otvorené 8.30–17 h` (a dotted time in a dash span), N61 (`v 8.30, asi 9.30 a 10.30 hod.`), N38 +
  N69 + sk `o 17 h` (`o 5 h`, `o 5 alebo 6 h`), the N65 trade-offs (`Pracoval po 14 h denne`, `Cesta trvala do 20 h`, mixed
  spans, `Do 14, 15 a 16 h`), N81 (`Ve 24 h provozu`).
- Case after prepositions: N55 (`o` + time after verbs of change), N84 (`Jednali o 5 %`), `Dohodli se na 5 %` / `Shodli se
  na 50 Kč` / `Trval na 5 %`, N83 (`na` + area), `Měří od 1 m po 5 m` / `Dostal od něj po 5 Kč`, N86 (sk `Zostala na 5 %` /
  `Trvalo to 1 h`), `Mluvil 1 h a čekal 2 h`, N88 (`po` + time).
- Slovak forms and word-form slips: N30 (`Prišli 5 muži`), N63 (masculine plurals mistagged), N64 (`Prišlo nulu ľudí`),
  `Dočkal sa 5 vnúčat a 3 pravnúčat`, `Viděl ho 2 krát`, `Za 0 tis. Kč`, `Má 1 pravnuka`, `Dotkl se 2. prstem stolu`,
  `Uplynulo 1 nebo 2 hodiny`.
- Roman numerals and capitals: the N58 remainders (`Václavu II. Přemyslovci i jeho synovi`, `Vzpomínal na Karla IV. Karel
  IV. byl`, `Ludvík XIV. Slunce vládl`, capitals `VÁCLAVU II.`), N82 (`I. Jan`), N59 (`V KNIZE II. DÍL`), N89 (abbreviation
  period in capitals).

Pinned, its own task when taken up: Stanza 1.15.0 (Dependabot #12) changes 563 of 9,818 corpus readings with its new
CAC and SNK models and fails 19 tests, so `pyproject.toml` holds `stanza<1.15`. Upgrading means going through those
readings and retuning the normalizer to the new tags. Its security alert (unrestricted pickle in the lemmatizer
checkpoint loader) does not reach caketts, which loads only the default `tokenize,pos` models (dismissed 2026-11-18).

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
- Found by the Slovak chain (2026-11-17): sk `Zúčastní sa 5 pretekov` päť + log (no Slovak event list; N21 twin); sk
  `Toho dňa sa bojí 5 psov` päť + log (no Slovak time genitives); `Bál sa 2 stretnutí` / `Bál se 2 setkání` dve/dvě (N7
  family); sk `Zúčastnilo sa 2 alebo 50 ľudí` dvoch (personal masculine A = G); sk `Vzdal sa 5 percent mzdy` päť (SNK
  tags `percent` singular).
