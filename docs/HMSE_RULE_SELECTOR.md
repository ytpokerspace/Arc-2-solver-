# Shared Shape Radar + HMSE rule selector

## Objective

Test whether the selector can choose among registered rule families using
visible training evidence, without receiving a task-family label.

Registered families:

1. `hole-fill`: propose a single `FILL_HOLES(color)` program when Shape Radar
   detects enclosed shapes, observed changes preserve existing foreground, and
   one added color exactly reproduces every visible training output.
2. `anchor-quadrant-relocation`: propose the spatial marker-to-2x2-anchor
   program from the second demonstration.
3. `reflection`: test left-right and top-bottom reflection against all visible
   pairs, accepting only one axis that exactly fits every pair.

## Selection policy

- Evaluate all registered families on the visible training pairs.
- A candidate is accepted only if it exactly reproduces every visible pair.
- Select only if exactly one family passes.
- If none passes, return unresolved; if multiple pass, return ambiguous.
- Never use the held-out output to choose a rule.
- Apply the selected program to the held-out input only after selection.

## Reproduction

```bash
python -m arc_r2.hmse_rule_selector \
  --task-json tests/fixtures/arc_agi2_00d62c1b_train.json \
  --task-json tests/fixtures/arc_agi2_d89b689b_train.json \
  --task-json tests/fixtures/synthetic_reflection_train.json \
  --candidate-cap 1500 \
  --max-depth 2 \
  --output arc_hmse_selector_report.json
```

The first two fixtures contain only pinned ARC-AGI-2 training pairs. The third
is a locally authored controlled synthetic fixture used to test whether the
selector can accommodate a distinct family. It is not an ARC dataset task.
The outer evaluator uses each held-out training output only after prediction;
the selector receives no held-out output.

## Limits

Only three authored rule families are registered. This is evidence-based
selection among known candidates, not open-ended rule invention. Folds from
the same task are correlated and must not be counted as independent tasks.
Baseline candidate counts and selector family checks are different search
spaces, not a fair efficiency comparison.
