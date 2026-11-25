# E1 development run (dev templates; automatic labels, S5 kappa pending)

Fit and selection on 33 train templates (template-grouped CV), scores on 16 dev templates; equal train size 3957 rows per variable. Not a test-set result.

| variable | layer, λ (train CV) | dev metric [95% CI] | shuffled mean (p) | probe − shuffled [95% CI] | diff-in-means dev | seeds 0–4 mean ± sd | readable on dev |
|---|---|---|---|---|---|---|---|
| content/binary | L13, 0.0001 | 0.810 [0.523, 0.967] | 0.556 (0.010) | +0.254 [-0.017, +0.417] | 0.875 (L25) | 0.814 ± 0.012 | no |
| content/4-class | L18, 10 | 0.624 [0.491, 0.759] | 0.330 (0.010) | +0.294 [+0.175, +0.419] | 0.404 (L19) | 0.641 ± 0.016 | no |
| evaluation/binary | L15, 100 | 0.664 [0.542, 0.769] | 0.613 (0.069) | +0.051 [-0.036, +0.129] | 0.685 (L13) | 0.656 ± 0.026 | no |
| evaluation/4-class | L23, 1 | 0.338 [0.264, 0.416] | 0.271 (0.030) | +0.066 [-0.009, +0.143] | 0.333 (L11) | 0.321 ± 0.031 | no |

n and clustering of the dev rows (A16):

- content/binary: n = 2223 steps, 16 templates, classes {'False': 2165, 'True': 58}; ICC 0.13507666802873705, deff 22.130292916624946, n_eff 100.45054570109261
- content/4-class: n = 4668 steps, 16 templates, classes {'+': 1114, '-': 521, '*': 2505, '/': 528}; ICC 0.2996900800836645, deff 98.86627380836676, n_eff 47.21529213337219
- evaluation/binary: n = 4865 steps, 16 templates, classes {'False': 1083, 'True': 3782}; ICC 0.2283191880523917, deff 80.89116005176629, n_eff 60.14254211321291
- evaluation/4-class: n = 4865 steps, 16 templates, classes {'none': 3782, 'planning': 278, 'binding': 498, 'arithmetic': 307}; ICC 0.16196168410826747, deff 57.67200789264399, n_eff 84.3563485609199
