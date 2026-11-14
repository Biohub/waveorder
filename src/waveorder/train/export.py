"""Export a bundle's student as a self-checked ONNX file: :func:`export_onnx`.

The graph folds in the input normalizer, the student and the inverse target normalizer. Inputs: ``data``
``(B, C, Z, Y, X)`` float32 and one ``(B, K)`` float32 per condition (named as the condition); output: ``estimate``
``(B, F, Zout, Y, X)`` in the model's units. The caller prepares the data, computes the conditions, and gives ``Y``
and ``X`` that are multiples of the student's stride. Needs ``pip install "waveorder[export]"``.
"""

import warnings
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from waveorder.errors import BundleError, ParameterError
from waveorder.nn.networks import Bundle, Normalizer, RestorationNetwork

FILE_NAME = "model.onnx"
"""The name of the ONNX file inside a bundle directory."""


class Folded(nn.Module):
    """A student with its normalizers folded in; ``names`` are the conditions in the order ``forward`` takes them."""

    def __init__(
        self,
        student: RestorationNetwork,
        input_normalizer: Normalizer,
        target_normalizer: Normalizer,
        *,
        names: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.student = student
        self.input_normalizer = input_normalizer
        self.target_normalizer = target_normalizer
        self.names = names

    def forward(self, data: Tensor, *conditions: Tensor) -> Tensor:
        """Return the estimate ``(B, F, Zout, Y, X)`` of ``data`` ``(B, C, Z, Y, X)`` and one ``(B, K)`` per name."""
        condition = dict(zip(self.names, conditions, strict=True))
        return self.target_normalizer.inverse(self.student(self.input_normalizer(data), condition or None))


def export_onnx(  # noqa: PLR0913 (the bundle and the file, and one keyword per setting of the export)
    bundle: Bundle,
    path: str | Path | None = None,
    *,
    opset: int = 18,
    planes: int | None = None,
    size: tuple[int, int] = (256, 256),
    dynamic: bool = True,
    check: bool = True,
) -> Path:
    """Write the student of a bundle as an ONNX file and check that it computes what the torch student does.

    Parameters
    ----------
    bundle : Bundle
        The bundle (:func:`waveorder.io.bundle.load_bundle` reads one).
    path : str or Path, optional
        The file to write; default ``model.onnx`` in the bundle's directory (record it with
        :func:`waveorder.io.bundle.record_file`).
    opset : int
        The ONNX opset version.
    planes : int, optional
        The fixed plane count ``Z`` of the graph; default that of the bundle's test input.
    size : tuple[int, int]
        ``(Y, X)`` of the export example, multiples of the student's stride.
    dynamic : bool
        Leave batch, ``Y`` and ``X`` free; otherwise the graph takes exactly ``(1, C, planes, *size)``.
    check : bool
        Run the file in onnxruntime on the bundle's test input and compare with the torch student and the recorded
        output. A static graph is checked only when ``planes`` and ``size`` match the test input.

    Returns
    -------
    Path
        The ONNX file.

    Raises
    ------
    ImportError
        If ``onnx``, ``onnxscript`` or (for the check) ``onnxruntime`` is missing.
    ParameterError
        If ``size`` is not a positive multiple of the stride, a static graph cannot be checked on the test input,
        the bundle has no test input, or there is neither a bundle directory nor ``path``.
    ShapeError
        If the student does not take ``planes`` planes.
    BundleError
        If the ONNX output differs from a reference by more than the bundle's tolerance; the file is removed.
    """
    _require("onnx", "onnxscript")
    test = bundle.test
    folder = bundle.path if path is None else Path(path).parent
    if test is None or folder is None:
        raise ParameterError(
            "export_onnx needs a bundle with a test input, and a path for a bundle not read from a directory; "
            "write it with waveorder.io.bundle.save_bundle and export the bundle it reads back",
            path="bundle",
        )
    target = folder / FILE_NAME if path is None else Path(path)
    student = bundle.student
    names = tuple(student.condition_sizes)
    planes = int(test.data.shape[2]) if planes is None else planes
    height, width = size
    if height % student.stride or width % student.stride or height < 1 or width < 1:
        raise ParameterError(
            f"export_onnx size {tuple(size)} is not a positive multiple of the student's stride {student.stride} "
            "along both axes",
            path="size",
        )
    static_test = (planes, height, width) == tuple(test.data.shape[2:])
    if check and not dynamic and not static_test:
        raise ParameterError(
            f"export_onnx cannot check a static graph for (planes, Y, X) = {(planes, height, width)} on the "
            f"bundle's test input of {tuple(test.data.shape[2:])}; pass dynamic=True, or planes and size of the "
            "test input, or check=False",
            path="dynamic",
        )
    student.planes_out(planes)  # refuses a plane count the stem does not take, before anything is traced
    module = Folded(student, bundle.input_normalizer, bundle.target_normalizer, names=names).to("cpu").eval()
    example = (
        torch.zeros(1, student.channels, planes, height, width),
        *(torch.zeros(1, student.condition_sizes[name]) for name in names),
    )
    batch = torch.export.Dim("batch")  # one dimension shared by the data and every condition
    free = {0: batch, 3: torch.export.Dim.AUTO, 4: torch.export.Dim.AUTO}
    # One entry per parameter of Folded.forward: the data, then the tuple of conditions when there are any.
    conditions = (tuple({0: batch} for _ in names),) if names else ()
    shapes = (free, *conditions) if dynamic else None
    target.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        # The data and the conditions share the batch axis; torch says it names that axis once, which is the intent.
        warnings.filterwarnings("ignore", message="# The axis name: batch will not be used")
        torch.onnx.export(
            module,
            example,
            str(target),
            input_names=["data", *names],
            output_names=["estimate"],
            opset_version=opset,
            dynamo=True,
            dynamic_shapes=shapes,
            external_data=False,
            verbose=False,
        )
    if check:
        with torch.no_grad():
            computed = module(test.data.float(), *(test.condition[name].float() for name in names))
        references = {"the torch student's output": computed}
        if test.expected is not None:
            references["the bundle's test output"] = test.expected
        try:
            _check(target, test.data, test.condition, references=references, tolerance=test.tolerance, whole=dynamic)
        except BundleError:
            target.unlink(missing_ok=True)
            raise
    return target


def _require(*packages: str) -> None:
    """Raise an ``ImportError`` naming the extra unless every package imports."""
    import importlib

    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError as error:
            raise ImportError(f'export_onnx needs the {package} package: pip install "waveorder[export]"') from error


def _check(
    path: Path,
    data: Tensor,
    condition: Mapping[str, Tensor],
    *,
    references: Mapping[str, Tensor],
    tolerance: float,
    whole: bool,
) -> None:
    """Run the ONNX file on the test input in onnxruntime; raise if a reference differs by more than ``tolerance``."""
    _require("onnxruntime")
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    batches = [slice(None)] if whole else [slice(index, index + 1) for index in range(data.shape[0])]
    outputs = []
    for rows in batches:
        feed = {"data": data[rows].float().numpy(force=True)}
        feed.update({name: value[rows].float().numpy(force=True) for name, value in condition.items()})
        outputs.append(torch.from_numpy(session.run(["estimate"], feed)[0]))
    estimate = torch.cat(outputs)
    for what, expected in references.items():
        if estimate.shape != expected.shape:
            raise BundleError(
                f"the ONNX file {path} returns {tuple(estimate.shape)} for the bundle's test input, but {what} is "
                f"{tuple(expected.shape)}; the export does not reproduce the student"
            )
        error = float((estimate - expected).abs().max())
        if not error <= tolerance:
            raise BundleError(
                f"the ONNX file {path} does not reproduce {what} in onnxruntime on the bundle's test input: the "
                f"largest absolute error is {error:.3g}, above the bundle's tolerance {tolerance:.3g}. The file was "
                "removed; an operator of the student may not export exactly (try another opset)"
            )
