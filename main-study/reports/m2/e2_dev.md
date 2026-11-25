# E2 development run (dev templates; automatic labels, S5 kappa pending)

## Best layer on the dev map (optimistic: selected and reported on dev)

| variable | step_first: layer, dev | step_last: layer, dev |
|---|---|---|
| content/binary | L19, 0.882 | L16, 0.933 |
| content/4-class | L16, 0.667 | L11, 0.885 |
| evaluation/binary | L0, 0.686 | L14, 0.819 |
| evaluation/4-class | L19, 0.380 | L11, 0.454 |

## Probe against baselines (E1 layer, step_first, dev)

| variable | probe | text before (Δ [95% CI]) | question only (Δ [95% CI]) | logit margin (Δ [95% CI]) |
|---|---|---|---|---|
| content/binary | 0.810 | 0.712 (+0.099 [-0.136, +0.303]) | 0.411 (+0.400 [-0.021, +0.632]) | 0.611 (+0.200 [-0.116, +0.467]) |
| content/4-class | 0.624 | 0.312 (+0.312 [+0.184, +0.441]) | 0.250 (+0.374 [+0.241, +0.509]) | 0.251 (+0.373 [+0.240, +0.508]) |
| evaluation/binary | 0.664 | 0.626 (+0.038 [-0.124, +0.171]) | 0.505 (+0.159 [-0.018, +0.320]) | 0.628 (+0.036 [-0.057, +0.143]) |
| evaluation/4-class | 0.338 | 0.250 (+0.088 [+0.014, +0.166]) | 0.250 (+0.088 [+0.014, +0.166]) | 0.251 (+0.087 [+0.014, +0.165]) |

## RQ2 timing (dev failures with s0)

- Threshold τ = 0.817 (FPR 0.1 on out-of-fold correct train steps); alarm rate on correct dev traces 0.055.
- Excess alarm before s0: +0.125 [+0.024, +0.234] (1084 steps).
- Excess alarm at and after s0: +0.179 [+0.128, +0.230] (1006 steps).
- Earliest alarm vs s0: {'natural': {'before': 94, 'at': 21, 'after': 39, 'none': 85}, 'primary': {'before': 22, 'at': 5, 'after': 7, 'none': 25}}.
- **Development verdict: unsupported: evaluation alarms precede the first content deviation.**
