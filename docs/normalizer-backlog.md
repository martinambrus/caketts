# Normalizer backlog (Prompt 2.2)

Found-not-fixed items of the text normalizer, triaged 2026-11-17. **Kept** items are real work: a crash, a regression,
or a reading an ordinary listener would hear wrong. **Parked** items are nits or diminishing returns, and are built only
when revived. New findings are added to one of the two lists as they are verified.

## Kept

Readings checked on 825ae8d; N74 and N71 re-checked on b83ec25.

| id | input | today | expected |
|---|---|---|---|
| N74 | `Trvá 1 nebo 2 hodiny.` (present verb, no subject; N44 fixed the past) | `jedna nebo dvě` | `jednu nebo dvě` |
| N71 | sk `Mám 5 miliónov €.` | ValueError | `päť miliónov eur` |
| N53 | `Záleží na 5 %.` / `Po 5 km zastavil.` | `na pět procent` / `Po pět kilometrů` | `na pěti procentech` / `Po pěti kilometrech` |
| N58 | `Patřilo to Václavu II. Přemyslovci.` | `Václavu druhému. Přemyslovci.` | no sentence break |
| N32 | `Zúčastňoval se 5 závodů.` / `Neboj se 5 psů.` | `pět` | `pěti` |
| N22 | sk `Týka sa to 5 ľudí.` / `Bál sa 5 psov.` | `päť` | `piatich` |
| N29 | sk `Získal 2 body.` | `dve body` | `dva body` |
| N65 | sk `Otvorené od 8 do 17 h.` | `od ôsmich do sedemnástich hodín` | `od ôsmej do sedemnástej` (user 2026-11-17: clock time for 13–23 or the `od` of such a span; logged) |
| N68 | sk `Prídem o 1:00.` | `o prvej` | `o jednej` (user 2026-11-17) |
| N91 | num2words variant values in book_config | not validated (not re-checked) | a bad value raises |

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

## Parked

- Genitive-verb subject or object, rare or log-only: N7 (`Dosáhl 2 vítězství` dvě; with N77, its regression), N12
  (`Dosáhl 5 tisíc bodů` pět), N21 (past `Zúčastnilo se 5 závodů`), N27 (ordinal + non-genitive noun), N49 (`Zúčastní se 5
  soutěží` pět: tagger), N51 (`Dítě se bálo 5 psů`), N57 (sentence-initial imperatives), N75 (ordinal after a mistagged
  noun), N78 (`sdružení` takes part), N79 (`Dosáhlo 2 %`), N80 (noise logs), N90 (sk `Bál sa 5 miliónov ľudí`), N94–N99
  (twins and trade-offs of N7b/N77/N79).
- Slovak edges: N30 (`Prišli 5 muži`), N31 (`100. narodenín`), N38 (`o 5 h`), N63 (masculine plurals mistagged; no safe fix
  without a lexicon), N64 (`Prišlo nulu ľudí`), N66 (`Oslávil 2. Vianoce`), N67 (collective numerals `dvoje dveře`), N69
  (`o 5 alebo 6 h`), N70 (plural-ordinal log noise), N86 (Slovak twins of N53/N74).
- Prepositions and lists: N40 (year shares the next noun across a comma), N54 (walk stops at an abbreviation), N55 (`o` +
  time after verbs of change), N56 (amount with its own genitive noun opening a clause), N60 + N85 (approximator mid-chain
  with a counted noun), N61 (`v 8.30, asi 9.30 a 10.30 hod.`), N72 (approximator inside a list), N73 (abbreviation ends a
  list), N81 (`Ve 24 h provozu`), N83 (`na` + area), N84 (`Jednali o 5 %`), N88 (`po` + time, narrow: `Po 5 minutách` reads
  right).
- Capitals and Roman numerals: N59 (`V KNIZE II. DÍL`), N82 (`I. Jan`), N89 (abbreviation period in capitals).
- Config nits: N62 (unclear error when the num2words config is not a dict), N92 (agreement option `dvacet jedno jablek`).
- Found by the numbers chain (2026-11-17): `Dosáhl 5 tisíc Kč` pět (needs N12A); `Dosáhl 1,2,3 nebo 5 bodů` jedna, dva,
  tři nebo pěti; `Bál se 5, 10 psů` pěti, deset (a lone comma pair is no list, by design); `Dosáhl 2,5 nebo 5 bodů`
  decimal nominative; `Za 0 tis. Kč` za nulu; `Pod 5, 10 nebo 15 stromy` (two-case preposition, no verb); `Uplynulo 1 nebo
  2 hodiny` jednu (ungrammatical input); `Dosáhl 3 gólů, 2 asistencí` dvě (N49 family); `Má 1 pravnuka` jedna (tagger);
  lists log once per number (log volume only, N5A trade-off).
