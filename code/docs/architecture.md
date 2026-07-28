# Architecture

## Object model

- `ExperimentConfig` is the immutable source of truth loaded from YAML.
- `Sample` owns the input image, attack prompt, edit prompts, and named masks.
- `AttackMethod` converts one clean sample into one protected image.
- `Inpainter` applies one backend to a protected image, mask, prompt, and seed.
- `Evaluator` consumes the complete inpainting artifact table.
- `ExperimentPipeline` resolves dependencies and never contains method-specific math.

The registry maps YAML `type` names to classes. Heavy packages are imported
inside `execute`, so `validate`, `plan`, and `--dry-run` stay cheap.

## Artifact contract

```text
runs/<experiment>/
├── attacks/<method>/<sample>/protected.png
├── inpainting/<backend>/<method>/<sample>/<mask>/prompt_NN.png
├── evaluation/<evaluator>/
├── logs/
└── run.json
```

Each image has `<image>.json`. Its SHA-256 protocol fingerprint includes the
component type and parameters, sample metadata, input identity, mask identity,
resolution, seed, and upstream fingerprint. Resume occurs only when both the
artifact and a matching sidecar exist.

## Why algorithms remain adapters

The released algorithms encode different native budgets, masks, prompt rules,
and optimization loops. Converting them into one shared optimizer would change
the scientific methods. GuardBench instead normalizes lifecycle and I/O while
letting each `AttackMethod` preserve its algorithm-specific semantics.
