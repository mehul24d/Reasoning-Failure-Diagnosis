# E4 (dev): content attribution and typology

**DEVELOPMENT CHECK of the E4 code on dev (prereg/E4.md is a DRAFT).** Labels: automatic (parser v4); S5 kappa pending (no agreement record).

Criterion (prereg/E4.md, DRAFT): template-bootstrap CI of the dev/test AUROC lies above 0.5.

## Share of failures content-attributable at s0

| set | train | dev |
|---|---|---|
| overall | 0.288 [0.131, 0.463] (n 1352) | 0.236 [0.081, 0.478] (n 496) |
| population=distractor, clean also failed | 0.295 [0.103, 0.500] (n 535) | 0.172 [0.050, 0.419] (n 198) |
| population=natural | 0.252 [0.085, 0.433] (n 656) | 0.176 [0.035, 0.427] (n 239) |
| population=primary | 0.410 [0.239, 0.565] (n 161) | 0.695 [0.312, 0.878] (n 59) |
| s0_error_type=arithmetic | 0.000 [0.000, 0.000] (n 153) | 0.000 [0.000, 0.000] (n 218) |
| s0_error_type=binding | 0.022 [0.000, 0.071] (n 725) | 0.007 [0.000, 0.047] (n 144) |
| s0_error_type=planning | 0.787 [0.527, 0.939] (n 474) | 0.866 [0.650, 0.973] (n 134) |

## Classifier

| features | n train / eval (templates) | AUROC | shuffled mean | AUROC − shuffled | p_perm | CI > 0.5 |
|---|---|---|---|---|---|---|
| typology (primary) | 1352 / 496 (12) | 0.654 [0.409, 0.881] | 0.563 | 0.091 [-0.007, 0.243] | 0.010 | no |
| typology + s0 error type (partly definitional) | 1352 / 496 (12) | 0.993 [0.974, 1.000] | 0.986 | 0.007 [-0.002, 0.032] | 0.010 | yes |
