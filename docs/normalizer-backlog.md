# Normalizer backlog (Prompt 2.2)

Found-not-fixed items of the text normalizer, triaged 2026-11-17. **Kept** items are real work: a crash, a regression,
or a reading an ordinary listener would hear wrong. **Parked** items are nits or diminishing returns, and are built only
when revived. New findings are added to one of the two lists as they are verified.

## Kept

Readings checked on 825ae8d unless noted.

| id | input | today | expected |
|---|---|---|---|
| N42 | `Mám 5 milionů €.` / `Mám 5 tisíc m².` | ValueError | `pět milionů eur`, `pět tisíc metrů čtverečních` |
| N71 | sk `Mám 5 miliónov €.` | ValueError | `päť miliónov eur` |
| N43 | `Přijdu v 0 h.` / `S 0 body skončil.` | `v nulu hodin` / `S nulou body` (regression) | `v nula hodin` / `S nula body` |
| N41 + N76 | `Dosáhl 5 bodů, 10 bodů nebo 15 bodů.` | `pěti bodů, deset bodů nebo patnáct bodů` | all genitive |
| N5 | `Dosáhl 5, 10 nebo 15 bodů.` | `pěti, deset nebo patnáct` | `pěti, deseti nebo patnácti` |
| N6 | `Dosáhl 1 nebo 2 bodů.` | `jeden nebo dvou` | `jednoho nebo dvou` |
| N39 | `Stál mezi 1 a 2 stromy.` | `mezi jedním a dva stromy` | `mezi jedním a dvěma stromy` |
| N44 + N74 | `Trvalo to 1 nebo 2 hodiny.` / `Trvá to 1 nebo 2 h.` | `jedna nebo dvě` | `jednu nebo dvě` |
| N53 | `Záleží na 5 %.` / `Po 5 km zastavil.` | `na pět procent` / `Po pět kilometrů` | `na pěti procentech` / `Po pěti kilometrech` |
| N58 | `Patřilo to Václavu II. Přemyslovci.` | `Václavu druhému. Přemyslovci.` | no sentence break |
| N32 | `Zúčastňoval se 5 závodů.` / `Neboj se 5 psů.` | `pět` | `pěti` |
| N22 | sk `Týka sa to 5 ľudí.` / `Bál sa 5 psov.` | `päť` | `piatich` |
| N29 | sk `Získal 2 body.` | `dve body` | `dva body` |
| N65 | sk `Otvorené od 8 do 17 h.` | `od ôsmich do sedemnástich hodín` | clock time (rule to confirm) |
| N68 | sk `Prídem o 1:00.` | `o prvej` | `o jednej` (to confirm) |
| N91 | num2words variant values in book_config | not validated (not re-checked) | a bad value raises |

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
