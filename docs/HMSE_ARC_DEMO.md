# Shape Radar + HMSE: first ARC-AGI-2 demonstration

## Goal

Run one complete, reproducible ARC-AGI-2 training task through Shape Radar and
the Hierarchical Method Synthesis Engine (HMSE), then compare it with the
existing primitive-only fitting baseline.

## Pinned task

- Task: \`00d62c1b\`
- Source: \`arcprize/ARC-AGI-2\`
- Revision: \`f3283f727488ad98fe575ea6a5ac981e4a188e49\`
- Fixture: \`tests/fixtures/arc_agi2_00d62c1b_train.json\`
- Only the five public training pairs are included. The test section and any
  test outputs are intentionally excluded.

## Run

\`\`\`bash
python -m arc_r2.hmse_demo \\
  --task-json tests/fixtures/arc_agi2_00d62c1b_train.json \\
  --candidate-cap 1500 \\
  --max-depth 2 \\
  --output arc_hmse_demo.json
\`\`\`

Or run the regression test:

\`\`\`bash
python -m unittest tests.test_hmse_arc_demo -v
\`\`\`

## Protocol

The benchmark uses leave-one-training-pair-out validation. On each of five
folds, four input/output pairs are visible to the solver and the fifth input is
predicted. The held-out output is used only by the outer evaluator after the
prediction has been produced.

The HMSE procedure has three explicit stages:

1. Shape Radar scans the visible training inputs and reports object geometry
   and enclosed-hole counts.
2. A bounded rule proposer infers a fill color from the visible pair
   differences, proposes \`FILL_HOLES(color)\`, and accepts it only if it exactly
   reproduces every visible training output. If that proposal fails, the
   existing bounded primitive enumerator is used as a fallback.
3. The selected program is applied to the held-out input. The evaluator then
   compares the prediction with the held-out output.

The primitive baseline uses the same visible training pairs, candidate cap,
and maximum program depth, but calls the existing \`fit_program\` directly.

## Interpretation limits

- This is a single-task engineering demonstration, not an ARC benchmark score.
- Five held-out pairs from one task are not five independent tasks.
- The HMSE schedule is explicitly authored for this demonstration; automatic
  schedule invention is not implemented.
- The hole-fill proposal is a narrow rule family, not general ARC rule
  induction.
- Matching the baseline on this task demonstrates integration and traceability,
  not a general accuracy improvement.
- Runtime values are machine- and environment-dependent.
