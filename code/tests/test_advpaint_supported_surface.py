from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADVPAINT = ROOT / "AdvPaint-main_revised"


def _literal_assignment(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment: {name}")


def _parser_argument(source: str, flag: str) -> ast.Call:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if isinstance(node.args[0], ast.Constant) and node.args[0].value == flag:
            return node
    raise AssertionError(f"Missing parser argument: {flag}")


def _keyword(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name:
            return ast.literal_eval(keyword.value)
    raise AssertionError(f"Missing keyword {name!r}")


def _function_dict_keys(source: str, function_name: str, variable_name: str) -> set[str]:
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            raise AssertionError(f"{variable_name!r} is not a dict literal")
        return {
            ast.literal_eval(key)
            for key in node.value.keys
            if key is not None
        }
    raise AssertionError(
        f"Missing dict assignment {variable_name!r} in {function_name!r}"
    )


def test_only_g1_g8_components_are_builtin() -> None:
    source = (ADVPAINT / "AdvPaint.py").read_text(encoding="utf-8")
    assert _literal_assignment(source, "G1_G8_ATTACK_COMPONENTS") == (
        "all",
        "all_multistep",
        "cross_concentration_self_l2",
        "cross_concentration_self_l2_multistep",
    )


def test_retired_objective_modules_stay_removed() -> None:
    for relative in (
        "global_attention_entropy_objective.py",
        "outside_attention.py",
        "run_mask_square_all_layers.sh",
        "run_outside_self_all_layers.sh",
    ):
        assert not (ADVPAINT / relative).exists()


def test_gradient_balanced_is_an_explicit_first_step_block_cli_mode() -> None:
    source = (ADVPAINT / "AdvPaint.py").read_text(encoding="utf-8")
    argument = _parser_argument(source, "--adaptive_block_score_mode")

    assert _keyword(argument, "choices") == [
        "legacy",
        "objective_aligned",
        "counterfactual_gap",
        "gradient_balanced",
        "mask_correlation",
        "mask_jacobian",
    ]
    help_text = _keyword(argument, "help")
    assert "fixed blocks" in help_text
    assert "mask correlation/Jacobian" in help_text


def test_gradient_block_weight_modes_are_explicit_and_backward_compatible() -> None:
    source = (ADVPAINT / "AdvPaint.py").read_text(encoding="utf-8")
    mode = _parser_argument(source, "--adaptive_block_weight_mode")
    shrink = _parser_argument(source, "--adaptive_block_causal_shrink")
    minimum = _parser_argument(source, "--adaptive_block_causal_min_weight")
    maximum = _parser_argument(source, "--adaptive_block_causal_max_weight")

    assert _keyword(mode, "default") == "inverse_gradient"
    assert _keyword(mode, "choices") == [
        "inverse_gradient",
        "causal_proportional",
        "uniform",
    ]
    assert _keyword(shrink, "default") == 0.25
    assert _keyword(minimum, "default") == 0.9
    assert _keyword(maximum, "default") == 1.1


def test_gradient_balanced_runtime_selection_is_delayed_to_first_pgd_forward() -> None:
    source = (ADVPAINT / "AdvPaint.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    attack = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "pgd_SelfQKV_And_Cross_Xadv"
    )
    attack_source = ast.get_source_segment(source, attack)
    assert attack_source is not None
    assert "and not mask_sensitivity_gating_mode" in attack_source

    pgd_loop = next(
        node
        for node in ast.walk(attack)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "iter"
    )
    dynamic_branch = next(
        node
        for node in pgd_loop.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "gradient_balanced_mode and iter == 0"
    )
    branch_source = ast.get_source_segment(source, dynamic_branch)
    assert branch_source is not None
    assert "probe_and_select_gradient_balanced_blocks(" in branch_source
    assert "weight_mode=args.adaptive_block_weight_mode" in branch_source
    assert "causal_shrink=args.adaptive_block_causal_shrink" in branch_source
    assert "retain_attention_cache_stems_(" in branch_source
    assert "register_cross_attention_hook(" in branch_source
    assert "weight_mode {args.adaptive_block_weight_mode}" in branch_source

    first_forward_line = min(
        node.lineno
        for node in ast.walk(pgd_loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "model"
        and node.func.attr == "unet"
    )
    assert dynamic_branch.lineno > first_forward_line


def test_adaptive_score_mode_is_in_advpaint_output_fingerprint() -> None:
    source = (ADVPAINT / "AdvPaint.py").read_text(encoding="utf-8")

    assert "adaptive_block_score_mode" in _function_dict_keys(
        source,
        "build_output_filename",
        "config",
    )
    assert 'if args.adaptive_block_weight_mode != "inverse_gradient"' in source
    assert 'config["adaptive_block_weight_mode"]' in source
