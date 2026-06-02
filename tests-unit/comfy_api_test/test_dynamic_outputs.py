"""Unit tests for ``DynamicOutputs.ByKey`` and the finalized-outputs path."""

import pytest

from comfy_api.latest import _io as io


# ---------------------------------------------------------------------------
# Schema-level construction and validation
# ---------------------------------------------------------------------------

def _byke():
    return io.DynamicOutputs.ByKey(
        id="result",
        selector="mode",
        options=[
            io.DynamicOutputs.Option(key="image",
                                     outputs=[io.Image.Output("image"), io.Mask.Output("mask")]),
            io.DynamicOutputs.Option(key="latent",
                                     outputs=[io.Latent.Output("latent")]),
        ],
    )


def test_option_rejects_empty_key():
    with pytest.raises(ValueError, match="non-empty string"):
        io.DynamicOutputs.Option(key="", outputs=[])


def test_option_rejects_non_output_entry():
    with pytest.raises(ValueError, match="Output instances"):
        io.DynamicOutputs.Option(key="x", outputs=["not an output"])


def test_option_requires_explicit_output_ids():
    with pytest.raises(ValueError, match="declare an id"):
        io.DynamicOutputs.Option(key="x", outputs=[io.Image.Output()])  # no id


def test_bykey_rejects_empty_options():
    with pytest.raises(ValueError, match="at least one Option"):
        io.DynamicOutputs.ByKey(id="r", selector="m", options=[])


def test_bykey_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate option key"):
        io.DynamicOutputs.ByKey(
            id="r", selector="m",
            options=[
                io.DynamicOutputs.Option(key="x", outputs=[io.Image.Output("a")]),
                io.DynamicOutputs.Option(key="x", outputs=[io.Latent.Output("b")]),
            ],
        )


def test_bykey_rejects_duplicate_output_ids_across_options():
    with pytest.raises(ValueError, match="appears in more than one option"):
        io.DynamicOutputs.ByKey(
            id="r", selector="m",
            options=[
                io.DynamicOutputs.Option(key="x", outputs=[io.Image.Output("dup")]),
                io.DynamicOutputs.Option(key="y", outputs=[io.Latent.Output("dup")]),
            ],
        )


# ---------------------------------------------------------------------------
# Schema integration
# ---------------------------------------------------------------------------

def _make_node(extra_outputs=None):
    """Build a V3 node class with a selector input + DynamicOutputs group."""
    extras = extra_outputs or []

    class DynNode(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="DynNode",
                inputs=[io.Combo.Input("mode", options=["image", "latent"], default="image")],
                outputs=[*extras, _byke()],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({"image": None, "mask": None})

    return DynNode


def test_schema_validate_rejects_unknown_selector():
    class BadSelector(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="BadSelector",
                inputs=[io.Combo.Input("not_mode", options=["a"])],
                outputs=[
                    io.DynamicOutputs.ByKey(
                        id="r", selector="mode",
                        options=[io.DynamicOutputs.Option(key="a", outputs=[io.Image.Output("a")])],
                    ),
                ],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({"a": None})

    with pytest.raises(ValueError, match="selector input 'mode' does not exist"):
        BadSelector.GET_SCHEMA()


def test_schema_validate_rejects_id_collision_with_static_output():
    class Collision(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="Collision",
                inputs=[io.Combo.Input("mode", options=["a"])],
                outputs=[
                    io.Image.Output("shared"),
                    io.DynamicOutputs.ByKey(
                        id="r", selector="mode",
                        options=[io.DynamicOutputs.Option(key="a", outputs=[io.Latent.Output("shared")])],
                    ),
                ],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({"shared": None})

    with pytest.raises(ValueError, match="Output ids must be unique"):
        Collision.GET_SCHEMA()


def test_schema_get_v1_info_emits_dynamic_outputs_field():
    DynNode = _make_node()
    DynNode.GET_SCHEMA()
    info = DynNode.SCHEMA.get_v1_info(DynNode)
    assert info.dynamic_outputs is not None and len(info.dynamic_outputs) == 1
    group = info.dynamic_outputs[0]
    assert group["kind"] == "by_key"
    assert group["selector"] == "mode"
    assert {opt["key"] for opt in group["options"]} == {"image", "latent"}
    # Static output arrays are empty — only the dynamic group is declared.
    assert info.output == []
    assert info.output_is_list == []


def test_schema_static_outputs_stable_prefix_in_v1_arrays():
    """A static output before a dynamic group still surfaces in RETURN_TYPES etc."""
    DynNode = _make_node(extra_outputs=[io.String.Output("status")])
    DynNode.GET_SCHEMA()
    # Class-level static arrays are the always-present prefix.
    assert list(DynNode.RETURN_TYPES) == ["STRING"]
    assert list(DynNode.RETURN_NAMES) == ["status"]
    assert list(DynNode.OUTPUT_IS_LIST) == [False]


# ---------------------------------------------------------------------------
# get_finalized_class_outputs
# ---------------------------------------------------------------------------

def test_finalize_picks_active_branch():
    schema_outputs = [_byke()]
    finalized = io.get_finalized_class_outputs(schema_outputs, {"mode": "latent"})
    assert finalized.output_ids == ["latent"]
    assert finalized.return_types == ["LATENT"]
    assert finalized.output_is_list == [False]


def test_finalize_unknown_selector_yields_empty():
    schema_outputs = [_byke()]
    finalized = io.get_finalized_class_outputs(schema_outputs, {"mode": "nonexistent"})
    assert len(finalized) == 0


def test_finalize_link_selector_yields_empty():
    """Link as selector value is treated as 'not finalizable' — no branch."""
    schema_outputs = [_byke()]
    finalized = io.get_finalized_class_outputs(schema_outputs, {"mode": ["src", 0]})
    assert len(finalized) == 0


def test_finalize_static_prefix_preserved():
    schema_outputs = [io.String.Output("status"), _byke()]
    finalized = io.get_finalized_class_outputs(schema_outputs, {"mode": "image"})
    assert finalized.output_ids == ["status", "image", "mask"]
    assert finalized.return_types == ["STRING", "IMAGE", "MASK"]


# ---------------------------------------------------------------------------
# NodeOutput.from_named
# ---------------------------------------------------------------------------

def test_nodeoutput_from_named_stores_dict():
    out = io.NodeOutput.from_named({"a": 1, "b": 2})
    assert out.named == {"a": 1, "b": 2}
    assert out.args == ()
    assert out.result is None  # `.result` is the positional tuple


def test_nodeoutput_rejects_mixed_positional_and_named():
    with pytest.raises(ValueError, match="cannot mix positional"):
        io.NodeOutput(1, 2, named={"a": 1})


# ---------------------------------------------------------------------------
# Group-id uniqueness
# ---------------------------------------------------------------------------

def test_schema_rejects_duplicate_dynamic_group_ids():
    class Dup(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="Dup",
                inputs=[io.Combo.Input("mode", options=["a"])],
                outputs=[
                    io.DynamicOutputs.ByKey(id="r", selector="mode",
                        options=[io.DynamicOutputs.Option(key="a", outputs=[io.Image.Output("x")])]),
                    io.DynamicOutputs.ByKey(id="r", selector="mode",
                        options=[io.DynamicOutputs.Option(key="a", outputs=[io.Latent.Output("y")])]),
                ],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({"x": None, "y": None})

    with pytest.raises(ValueError, match="DynamicOutputs group ids must be unique"):
        Dup.GET_SCHEMA()


# ---------------------------------------------------------------------------
# DynamicOutputs.FromInput — DynamicCombo / DynamicSlot integration
# ---------------------------------------------------------------------------

def _combo_options_with_outputs():
    return [
        io.DynamicCombo.Option(
            key="image",
            inputs=[io.Image.Input("img")],
            outputs=[io.Image.Output("processed"), io.Mask.Output("alpha")],
        ),
        io.DynamicCombo.Option(
            key="latent",
            inputs=[io.Latent.Input("lat")],
            outputs=[io.Latent.Output("denoised")],
        ),
    ]


def _slot_options_with_outputs():
    return [
        io.DynamicSlot.Option(
            when=io.Image,
            outputs=[io.Image.Output("processed"), io.Mask.Output("alpha")],
        ),
        io.DynamicSlot.Option(
            when=io.Latent,
            outputs=[io.Latent.Output("denoised")],
        ),
        io.DynamicSlot.Option(
            when=None,
            inputs=[io.Int.Input("seed")],
            outputs=[],
        ),
    ]


def test_fromInput_finalizes_combo_branch():
    schema_inputs = [io.DynamicCombo.Input("mode", options=_combo_options_with_outputs())]
    schema_outputs = [io.String.Output("status"), io.DynamicOutputs.FromInput("mode")]
    finalized = io.get_finalized_class_outputs(
        schema_outputs, {"mode": "image"}, schema_inputs=schema_inputs,
    )
    assert finalized.output_ids == ["status", "processed", "alpha"]
    assert finalized.return_types == ["STRING", "IMAGE", "MASK"]


def test_fromInput_unknown_combo_key_yields_only_static():
    schema_inputs = [io.DynamicCombo.Input("mode", options=_combo_options_with_outputs())]
    schema_outputs = [io.String.Output("status"), io.DynamicOutputs.FromInput("mode")]
    finalized = io.get_finalized_class_outputs(
        schema_outputs, {"mode": "missing"}, schema_inputs=schema_inputs,
    )
    assert finalized.output_ids == ["status"]


def test_fromInput_finalizes_slot_by_resolved_type():
    schema_inputs = [io.DynamicSlot.Input("slot", options=_slot_options_with_outputs())]
    schema_outputs = [io.DynamicOutputs.FromInput("slot")]
    # Connected with resolved type IMAGE → first option matches
    finalized = io.get_finalized_class_outputs(
        schema_outputs,
        {"slot": ["upstream", 0]},
        schema_inputs=schema_inputs,
        live_input_types={"slot": "IMAGE"},
    )
    assert finalized.output_ids == ["processed", "alpha"]
    # Connected, LATENT branch
    finalized = io.get_finalized_class_outputs(
        schema_outputs,
        {"slot": ["upstream", 0]},
        schema_inputs=schema_inputs,
        live_input_types={"slot": "LATENT"},
    )
    assert finalized.output_ids == ["denoised"]


def test_fromInput_slot_unconnected_uses_when_none_option():
    schema_inputs = [io.DynamicSlot.Input("slot", options=_slot_options_with_outputs())]
    schema_outputs = [io.DynamicOutputs.FromInput("slot")]
    finalized = io.get_finalized_class_outputs(
        schema_outputs, {}, schema_inputs=schema_inputs,
    )
    # when=None option declares outputs=[] → no active outputs
    assert finalized.output_ids == []


def test_fromInput_slot_unmatched_type_yields_empty():
    """Resolved upstream type with no matching option contributes no slots."""
    schema_inputs = [io.DynamicSlot.Input("slot", options=_slot_options_with_outputs())]
    schema_outputs = [io.DynamicOutputs.FromInput("slot")]
    finalized = io.get_finalized_class_outputs(
        schema_outputs,
        {"slot": ["upstream", 0]},
        schema_inputs=schema_inputs,
        live_input_types={"slot": "AUDIO"},
    )
    assert finalized.output_ids == []


def test_schema_rejects_fromInput_pointing_at_missing_input():
    class BadRef(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="BadRef",
                inputs=[io.Combo.Input("mode", options=["a"])],
                outputs=[io.DynamicOutputs.FromInput("does_not_exist")],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({})

    with pytest.raises(ValueError, match="must reference a DynamicCombo or DynamicSlot"):
        BadRef.GET_SCHEMA()


def test_schema_rejects_fromInput_referenced_more_than_once():
    class DupRef(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="DupRef",
                inputs=[io.DynamicCombo.Input("mode", options=_combo_options_with_outputs())],
                outputs=[io.DynamicOutputs.FromInput("mode"), io.DynamicOutputs.FromInput("mode")],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({})

    with pytest.raises(ValueError, match="referenced more than once"):
        DupRef.GET_SCHEMA()


def test_schema_rejects_fromInput_output_collision_with_static():
    class Collision(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="Collision",
                inputs=[
                    io.DynamicCombo.Input("mode", options=[
                        io.DynamicCombo.Option(
                            key="image", inputs=[io.Image.Input("img")],
                            outputs=[io.Image.Output("processed")],
                        ),
                    ]),
                ],
                outputs=[io.Image.Output("processed"), io.DynamicOutputs.FromInput("mode")],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({"processed": None})

    with pytest.raises(ValueError, match="Output ids must be unique"):
        Collision.GET_SCHEMA()


def test_v1_info_emits_by_key_for_combo_fromInput():
    class N(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="ComboFI",
                inputs=[io.DynamicCombo.Input("mode", options=_combo_options_with_outputs())],
                outputs=[io.DynamicOutputs.FromInput("mode")],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({})

    N.GET_SCHEMA()
    info = N.SCHEMA.get_v1_info(N)
    assert info.dynamic_outputs is not None and len(info.dynamic_outputs) == 1
    entry = info.dynamic_outputs[0]
    assert entry["kind"] == "by_key"
    assert entry["selector"] == "mode"
    keys = {opt["key"] for opt in entry["options"]}
    assert keys == {"image", "latent"}


def test_v1_info_emits_by_slot_for_slot_fromInput():
    class N(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="SlotFI",
                inputs=[io.DynamicSlot.Input("slot", options=_slot_options_with_outputs())],
                outputs=[io.DynamicOutputs.FromInput("slot")],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({})

    N.GET_SCHEMA()
    info = N.SCHEMA.get_v1_info(N)
    assert info.dynamic_outputs is not None and len(info.dynamic_outputs) == 1
    entry = info.dynamic_outputs[0]
    assert entry["kind"] == "by_slot"
    assert entry["selector"] == "slot"
    whens = [opt["when"] for opt in entry["options"]]
    assert whens == [["IMAGE"], ["LATENT"], None]
