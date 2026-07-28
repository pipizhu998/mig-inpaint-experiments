# Experiment configuration

An experiment YAML contains six independent decisions:

1. `dataset`: manifest, image/mask roots, mask names, and optional sample slice.
2. `protocol`: resolution and attack/inpainting seeds.
3. `methods`: clean reference, AdvPaint variants, and baseline adapters.
4. `inpainters`: one or more target models.
5. `evaluators`: one or more metric suites with an explicit clean baseline.
6. `experiment`: output namespace and resume policy.

All paths are resolved from `project_root`. Component `name` controls the
artifact namespace; component `type` selects the registered Python class.
Changing parameters changes the fingerprint and prevents stale reuse.

Use `variants` when several ablations share one implementation:

```yaml
- name: family
  type: advpaint
  params:
    iterations: 250
    # shared fields ...
  variants:
    - name: single_l2
      params: {attack_component: all, timestep_indices: "0"}
    - name: multi_l2
      params: {attack_component: all_multistep, timestep_indices: "0,5,10,15,19"}
```

`transfer_7_18.yaml` is the formal comparison configuration.
`advpaint_ablation.yaml` is the complete 2×2×2 AdvPaint matrix.
`smoke.yaml` exercises orchestration without torch, diffusers, or a GPU.
