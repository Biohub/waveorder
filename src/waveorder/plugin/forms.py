"""A magicgui form of the version-4 configuration schema, and the configuration mapping read back from it.

A block of the schema is a container. An optional block (``solver.calibration``) starts with an ``enable`` checkbox,
off when the configuration leaves the block out. A field of a few names is a combo box, a flag is a checkbox, and every
other field is a line of YAML (``0.9``, ``[BF]``, ``{nominal: 0.45, bounds: [0.3, 0.7]}``), so the form can write every
value the schema accepts. :func:`read_form` returns the mapping that :func:`waveorder.config.load` validates.
"""

import types
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAliasType, Union, get_args, get_origin

import yaml
from magicgui import widgets
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined, to_jsonable_python

from waveorder.config import V4Config
from waveorder.config.loader import parse

OPTIONAL_BLOCK_TOGGLE = "enable"
"""Name of the checkbox that heads the container of an optional block; unchecked, the block is submitted as null."""

HIDDEN = frozenset({"version"})
"""Fields the form does not show: :func:`read_form` writes ``version: 4`` itself."""

_HIGHLIGHT = "border: 1px solid rgb(255, 255, 0);"


class FormError(ValueError):
    """A line of the form that is not YAML.

    Parameters
    ----------
    path : str
        Dotted path of the field, e.g. ``"imaging_system.detection.numerical_aperture"``.
    message : str
        What the YAML parser reported.
    """

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path


def _resolved(annotation: object) -> object:
    """Return ``annotation`` without its ``type`` aliases and ``Annotated`` metadata."""
    while True:
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
        elif get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        else:
            return annotation


def _members(annotation: object) -> list[object]:
    """Return the members of a union, flattened and resolved; ``[annotation]`` for anything else."""
    resolved = _resolved(annotation)
    if get_origin(resolved) is Union or isinstance(resolved, types.UnionType):
        return [member for argument in get_args(resolved) for member in _members(argument)]
    return [resolved]


def _is_block(annotation: object) -> bool:
    """Whether ``annotation`` is a schema block with fixed fields (not a mapping of free keys)."""
    return (
        isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
        and annotation.model_config.get("extra") != "allow"
    )


def field_kind(annotation: object) -> tuple[str, Any]:
    """Return how a field is shown: as a block (with its model), or a choice, a flag or a line of text.

    Parameters
    ----------
    annotation : object
        The field's annotation in the schema.

    Returns
    -------
    tuple of str and object
        The kind, and the block's model or the choices.
    """
    members = _members(annotation)
    present = [member for member in members if member is not type(None)]
    if len(present) == 1:
        (only,) = present
        if _is_block(only):
            return ("optional" if len(members) > 1 else "block"), only
        if len(members) == 1 and only is bool:
            return "flag", None
        if len(members) == 1 and get_origin(only) is Literal:
            return "choice", get_args(only)
    return "text", None


def to_text(value: object) -> str:
    """Return ``value`` as one line of YAML; ``None`` is the empty line."""
    if value is None:
        return ""
    text = yaml.safe_dump(to_jsonable_python(value), default_flow_style=True, width=1_000_000, sort_keys=False)
    return text.removesuffix("\n...\n").strip()


def from_text(text: str, path: str) -> object:
    """Return the value of one line of YAML; the empty line is ``None``.

    Raises
    ------
    FormError
        If ``text`` is not YAML.
    """
    if not text.strip():
        return None
    try:
        return parse(text)
    except yaml.YAMLError as error:
        raise FormError(path, " ".join(str(error).split())) from None


def _default(field: FieldInfo) -> object:
    """Return the JSON form of a field's default, or ``None`` for a required field."""
    if field.default is PydanticUndefined and field.default_factory is None:
        return None
    return to_jsonable_python(field.get_default(call_default_factory=True, validated_data={}))


def _tooltip(name: str, field: FieldInfo, kind: str) -> str:
    parts = [field.description or name.replace("_", " ")]
    if kind == "text":
        parts.append("YAML; empty is null")
    return " | ".join(parts)


def _field_widget(name: str, field: FieldInfo, value: object) -> widgets.Widget:
    kind, detail = field_kind(field.annotation)
    tooltip = _tooltip(name, field, kind)
    if kind in ("block", "optional"):
        block = widgets.Container(name=name, label=name)
        if kind == "optional":
            block.append(widgets.CheckBox(name=OPTIONAL_BLOCK_TOGGLE, value=value is not None, text=f"enable {name}"))
        add_fields(detail, block, value if isinstance(value, Mapping) else None)
        block.tooltip = tooltip
        return block
    if kind == "choice":
        choices = list(detail)
        return widgets.ComboBox(
            name=name, choices=choices, value=value if value in choices else choices[0], tooltip=tooltip
        )
    if kind == "flag":
        return widgets.CheckBox(name=name, value=bool(value), tooltip=tooltip)
    return widgets.LineEdit(name=name, value=to_text(value), tooltip=tooltip)


def add_fields(model: type[BaseModel], container: widgets.Container, values: Mapping[str, Any] | None) -> None:
    """Append one widget per field of ``model`` to ``container``, showing ``values`` (else the defaults).

    Parameters
    ----------
    model : type of BaseModel
        A block of the schema, e.g. :class:`waveorder.config.V4Config`.
    container : magicgui.widgets.Container
        Where the widgets go; a nested block gets its own container, named as its field.
    values : Mapping or None
        The block's values in their JSON form (:func:`waveorder.config.dump`); a missing key shows the default.
    """
    for name, field in model.model_fields.items():
        if name in HIDDEN:
            continue
        value = values[name] if values is not None and name in values else _default(field)
        container.append(_field_widget(name, field, value))


def read_fields(model: type[BaseModel], container: widgets.Container, prefix: str = "") -> dict[str, Any]:
    """Return the values of a block built by :func:`add_fields`, as a mapping for the schema.

    Raises
    ------
    FormError
        If a YAML line does not parse.
    """
    values: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        if name in HIDDEN:
            continue
        path = f"{prefix}{name}"
        widget = container[name]
        kind, detail = field_kind(field.annotation)
        if kind in ("block", "optional"):
            assert isinstance(widget, widgets.Container)
            values[name] = (
                None
                if kind == "optional" and not widget[OPTIONAL_BLOCK_TOGGLE].value
                else read_fields(detail, widget, f"{path}.")
            )
        elif kind == "text":
            values[name] = from_text(widget.value, path)
        else:
            values[name] = widget.value
    return values


def build_form(name: str, values: Mapping[str, Any]) -> widgets.Container:
    """Return the form of a whole configuration, filled in with ``values`` (a :func:`waveorder.config.dump`)."""
    form = widgets.Container(name=name, scrollable=False)
    add_fields(V4Config, form, values)
    return form


def read_form(form: widgets.Container) -> dict[str, Any]:
    """Return the configuration mapping of a form built by :func:`build_form`, with ``version: 4``.

    Raises
    ------
    FormError
        If a YAML line does not parse.
    """
    return {"version": 4, **read_fields(V4Config, form)}


def locate(form: widgets.Container, location: Sequence[int | str]) -> widgets.Widget | None:
    """Return the deepest widget of ``form`` on the path ``location`` of a validation error, or ``None``."""
    found: widgets.Widget | None = None
    node: widgets.Widget = form
    for part in location:
        if not isinstance(node, widgets.Container) or not isinstance(part, str):
            break
        child = next((widget for widget in node if widget.name == part), None)
        if child is None:
            break
        found = node = child
    return found


def highlight(widget: widgets.Widget, message: str) -> None:
    """Frame ``widget`` and add ``message`` to its tooltip."""
    widget.native.setStyleSheet(_HIGHLIGHT)
    widget.tooltip = f"{widget.tooltip or ''}\n{message}".strip()
