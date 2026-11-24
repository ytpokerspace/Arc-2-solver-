# Shape Radar + HMSE: second ARC-AGI-2 demonstration

## Goal

Test transfer of the Shape Radar + HMSE workflow on a transformation family
different from hole-filling: moving four isolated color markers into a central
2x2 anchor according to their original diagonal quadrants.

## Pinned training task

- Task: \`d89b689b\`
- Source: \`arcprize/ARC-AGI-2\`
- Revision: \`f3283f727488ad98fe575ea6a5ac981e4a188e49\`
- Fixture: \`tests/fixtures/arc_agi2_d89b689b_train.json\`
- The fixture contains only the three training pairs. The task test input and
  test output are excluded.

## Run

\`\`\`bash
python -m arc_r2.hmse_spatial_demo \\
  --task-json tests/fixtures/arc_agi2_d89b689b_train.json \\
  --candidate-cap 1500 \\
  --max-depth 2 \\
  --output arc_hmse_spatial_demo.json
\`\`\`

## Protocol

Leave-one-training-pair-out validation: on each of three folds, two pairs are
visible to both engines and the third output is held back until after
prediction. The baseline is the existing bounded primitive \`fit_program\`
search. The HMSE path uses Shape Radar, a fixed three-stage procedure, and a
task-specific rule proposer. Every proposed rule must exactly reproduce all
visible training pairs before it is applied to the held-out input.

## Interpretation limits

- This is a second single-task engineering demonstration, not a benchmark score.
- Three folds from one task are not three independent tasks.
- The spatial rule family and three-stage schedule are authored for this task.
  This tests whether the orchestration can support a different rule family; it
  does not establish automatic rule discovery or general ARC generalization.
- Baseline and HMSE search counts are not equivalent: one enumerates general
  primitives, while the other tests one structured task-specific hypothesis.
- Runtime is specific to the CI environment.
