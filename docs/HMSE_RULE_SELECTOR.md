# Shared Shape Radar + HMSE rule selector

## Objective

Test whether the solver can select between two previously demonstrated rule
families using training evidence, without receiving a task-family label.

Registered families:

1. \`hole-fill\`: propose a single \`FILL_HOLES(color)\` program when Shape Radar
   detects enclosed shapes, observed changes preserve existing foreground, and
   one added color exactly reproduces every visible training output.
2. \`anchor-quadrant-relocation\`: propose the spatial marker-to-2x2-anchor
   program from the second demonstration.

## Selection policy

- Evaluate both rule families on the visible training pairs.
- A candidate is accepted only if it exactly reproduces every visible pair.
- Select only if exactly one candidate passes.
- If neither passes, return unresolved; do not fall back to a guess.
- If multiple candidates pass, return ambiguous; do not break the tie using
  held-out outputs.
- Apply the selected program to the held-out input only after selection.

## Reproduction

\`\`\`bash
python -m arc_r2.hmse_rule_selector \\
  --task-json tests/fixtures/arc_agi2_00d62c1b_train.json \\
  --task-json tests/fixtures/arc_agi2_d89b689b_train.json \\
  --candidate-cap 1500 \\
  --max-depth 2 \\
  --output arc_hmse_selector_report.json
\`\`\`

The fixture files contain training pairs only. The held-out training output is
used by the outer evaluator after prediction and is not passed into the
selector.

## Limits

This experiment registers only two authored rule families. It tests evidence-
based selection between those families, not open-ended rule invention. Folds
from the same task are correlated and must not be counted as independent tasks.
Baseline candidate counts and selector family checks are different search
spaces, not a fair efficiency comparison.
