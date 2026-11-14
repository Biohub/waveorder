"""The distilled solver: a student run as the solver of a model, with no transfer function computed."""

import pickle

import pytest
import torch
from torch import Tensor

from tests.models.plates import fluorescence_model, thin_phase_model
from tests.nn.networks import tiny
from waveorder.errors import BackendUnavailableError, IncompatibleModelError, ParameterError
from waveorder.io.bundle import load_bundle, save_bundle
from waveorder.models import ReconstructionModel
from waveorder.models._pool import pickled_model
from waveorder.nn import Distilled, Sampling, SolveContext, ThinPhase2D, WidefieldFluorescence3D, solve_context
from waveorder.nn.networks import (
    Affine,
    Attributes,
    Bias,
    Bundle,
    Contract,
    Example,
    FieldPosition,
    FieldStandardize,
    RestorationNetwork,
    predict,
)
from waveorder.tiling import halo_for

PIXEL_SIZE = (0.4, 0.15, 0.15)


def _phase_bundle(path, *, student: RestorationNetwork | None = None, **changes: object):
    """A one-field phase student (``Phase2D`` plates hold phase only) for stacks of 5 planes."""
    model = thin_phase_model()
    contract = Contract.of(
        model.imaging_model,
        pixel_size=PIXEL_SIZE,
        model="Phase2D",
        prepare_scope=model.prepare_scope,
        planes=(5,),
        fields=("phase",),
    )
    settings: dict = {
        "student": student or tiny.strided(5, halo=16),
        "contract": contract,
        "target_normalizer": Affine(offset=0.1, scale=0.5),
        "model": {"class": "Phase2D"},
        "record": {},
        "test": Example(data=torch.randn(1, 1, 5, 32, 32, generator=torch.Generator().manual_seed(0))),
        **changes,
    }
    return save_bundle(Bundle(**settings), path)


def _intensities(planes: int = 5, height: int = 40, width: int = 44, *, seed: int = 1) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return 1000 * (1 + 0.02 * torch.randn(planes, height, width, generator=generator))


def _by_hand(model: ReconstructionModel, data: Tensor, **condition: Tensor) -> Tensor:
    """The model's distilled student called directly on the model's prepared data, without the solver."""
    solver = model.solver
    assert isinstance(solver, Distilled)
    prepared = model.prepare(data[None] if data.dim() == 3 else data).float()
    with torch.no_grad():
        return predict(
            solver.student,
            prepared[None],
            condition,
            input_normalizer=solver.input_normalizer,
            target_normalizer=solver.target_normalizer,
        )[0]


@pytest.fixture(autouse=True)
def _seeded():
    """Every test builds its random networks from the same seed, so a tolerance holds on every run."""
    torch.manual_seed(0)


@pytest.fixture
def phase_bundle(tmp_path):
    return _phase_bundle(tmp_path / "phase")


def test_a_phase_student_is_the_solver_of_phase_2d_and_returns_zero_absorption(phase_bundle):
    model = thin_phase_model()
    model.solver = Distilled(load_bundle(phase_bundle))
    data = _intensities()
    result = model.reconstruct(data, pixel_size=PIXEL_SIZE)
    assert result.estimate.shape == (2, 1, 40, 44)  # every field of the system
    assert result.status == "distilled"
    assert result.iterations == 0
    assert torch.isnan(result.objective)
    assert bool(result.finite)  # the NaN objective was not computed and is not a failure
    phase, absorption = result.estimate
    torch.testing.assert_close(phase, _by_hand(model, data)[0])
    assert phase.abs().max() > 0
    assert torch.equal(absorption, torch.zeros_like(absorption))


def test_a_volume_student_is_the_solver_of_fluorescence_3d(tmp_path):
    model = fluorescence_model()
    contract = Contract.of(
        model.imaging_model, pixel_size=PIXEL_SIZE, model="Fluorescence3D", prepare_scope=model.prepare_scope
    )
    student = tiny.sliding(3)
    bundle = Bundle(
        student=student,
        contract=contract,
        input_normalizer=FieldStandardize(),
        test=Example(data=torch.randn(1, 1, 4, 32, 32)),
    )
    path = save_bundle(bundle, tmp_path / "volume")
    model.solver = Distilled(load_bundle(path))
    data = 100 + torch.rand(6, 40, 64, generator=torch.Generator().manual_seed(2))
    result = model.reconstruct(data, pixel_size=PIXEL_SIZE)
    assert result.estimate.shape == (1, 6, 40, 64)
    torch.testing.assert_close(result.estimate, _by_hand(model, data))
    assert model.solver.tile_halo() == (1, 32, 32)


def test_a_model_with_a_distilled_solver_pickles_and_moves(phase_bundle):
    model = thin_phase_model()
    model.solver = Distilled(load_bundle(phase_bundle))
    data = _intensities()
    expected = model.reconstruct(data, pixel_size=PIXEL_SIZE).estimate
    for copy in (pickle.loads(pickle.dumps(model)), pickle.loads(pickle.loads(pickled_model(model))[1])):
        assert isinstance(copy.solver, Distilled)
        torch.testing.assert_close(copy.reconstruct(data, pixel_size=PIXEL_SIZE).estimate, expected)
        assert copy.solver.bundle is not None
        assert copy.solver.bundle.sha256 == load_bundle(phase_bundle).sha256
    model.to("cpu")
    assert isinstance(model.solver, Distilled)
    assert {parameter.device.type for parameter in model.solver.parameters()} == {"cpu"}


def test_data_outside_the_contract_is_refused_with_the_reason(phase_bundle):
    model = thin_phase_model()
    model.solver = Distilled(load_bundle(phase_bundle))
    with pytest.raises(IncompatibleModelError, match="the data has 7 plane"):
        model.reconstruct(_intensities(planes=7), pixel_size=PIXEL_SIZE)
    with pytest.raises(IncompatibleModelError, match=r"the pixel size along y is 0\.18 um, but the student was"):
        model.reconstruct(_intensities(), pixel_size=(0.4, 0.18, 0.15))  # 20% off
    other = fluorescence_model()
    with pytest.raises(IncompatibleModelError, match="distilled from a ThinPhase2D system") as error:
        other.solver = Distilled(load_bundle(phase_bundle))
    assert "the student returns the field(s) phase, which WidefieldFluorescence3D does not have" in str(error.value)


def test_a_distilled_solve_computes_no_transfer_function(phase_bundle, monkeypatch):
    model = thin_phase_model()
    solver = Distilled(load_bundle(phase_bundle))
    model.solver = solver
    binds: list[Sampling] = []

    def bind(system, sampling, **options):
        binds.append(sampling)
        return original(system, sampling, **options)

    original = ThinPhase2D.bind
    monkeypatch.setattr(ThinPhase2D, "bind", bind)
    data = _intensities()
    model.check((5, 40, 44), pixel_size=PIXEL_SIZE)
    model.reconstruct(data, pixel_size=PIXEL_SIZE)
    prepared = model.prepare(data[None]).float()
    solver(prepared, model.imaging_model, Sampling(shape=(5, 40, 44), pixel_size=PIXEL_SIZE))  # the module level
    assert binds == []
    scored = Distilled(load_bundle(phase_bundle), score_objective=True)
    result = scored(prepared, model.imaging_model, Sampling(shape=(5, 40, 44), pixel_size=PIXEL_SIZE))
    assert len(binds) == 1  # the objective needs H
    assert torch.isfinite(result.objective)


def test_no_transfer_function_is_computed_for_tiles_either(tmp_path, monkeypatch):
    model = fluorescence_model()
    contract = Contract.of(
        model.imaging_model, pixel_size=PIXEL_SIZE, model="Fluorescence3D", prepare_scope=model.prepare_scope
    )
    bundle = Bundle(student=tiny.sliding(3), contract=contract, test=Example(data=torch.randn(1, 1, 4, 32, 32)))
    path = save_bundle(bundle, tmp_path / "volume")
    model.solver = Distilled(load_bundle(path))
    monkeypatch.setattr(WidefieldFluorescence3D, "bind", lambda *_, **__: pytest.fail("bound"))
    data = 100 + torch.rand(1, 4, 48, 80, generator=torch.Generator().manual_seed(3))
    plan = model.tile_plan((4, 48, 80), pixel_size=PIXEL_SIZE, tile=(4, 32, 32))
    assert plan.halo == (1, 24, 32)  # the student's halo, each axis at most half the read region
    assert halo_for(model.imaging_model, Sampling(shape=(4, 48, 80), pixel_size=PIXEL_SIZE), solver=model.solver) == (
        1,
        24,
        32,
    )
    sink = torch.zeros(1, 4, 48, 80)
    stats = model.reconstruct_tiles([(data, sink)], pixel_size=PIXEL_SIZE, plan=plan)
    assert stats.tiles == len(plan.regions) > 1
    assert stats.nonfinite == stats.unconverged == 0
    assert sink.abs().max() > 0


def test_bf16_runs_on_the_cpu_and_fp16_is_refused_there(phase_bundle):
    data = _intensities()
    model = thin_phase_model()
    model.solver = Distilled(load_bundle(phase_bundle))
    full = model.reconstruct(data, pixel_size=PIXEL_SIZE).estimate
    model.solver = Distilled(load_bundle(phase_bundle), precision="bf16")
    half = model.reconstruct(data, pixel_size=PIXEL_SIZE).estimate
    assert half.dtype == full.dtype == torch.float32
    assert (half - full).norm() <= 0.05 * full.norm()  # bfloat16 keeps about 3 significant digits per layer
    model.solver = Distilled(load_bundle(phase_bundle), precision="fp16")
    with pytest.raises(BackendUnavailableError, match="'fp16' runs on CUDA only"):
        model.reconstruct(data, pixel_size=PIXEL_SIZE)
    with pytest.raises(ParameterError, match="precision must be one of float32, bf16, fp16"):
        Distilled(load_bundle(phase_bundle), precision="int8")  # ty: ignore[invalid-argument-type]


def test_a_float64_model_keeps_a_float32_student_and_returns_float64(phase_bundle):
    data = _intensities()
    model = thin_phase_model()
    model.solver = Distilled(load_bundle(phase_bundle))
    single = model.reconstruct(data, pixel_size=PIXEL_SIZE).estimate
    model.to(dtype=torch.float64)
    assert {parameter.dtype for parameter in model.solver.parameters()} == {torch.float32}
    double = model.reconstruct(data, pixel_size=PIXEL_SIZE).estimate
    assert double.dtype == torch.float64
    torch.testing.assert_close(double.float(), single, rtol=1e-5, atol=1e-6)


def test_a_distilled_solve_is_differentiable_and_never_trains_the_student(phase_bundle):
    model = thin_phase_model()
    solver = Distilled(load_bundle(phase_bundle))
    solver.train()
    assert not solver.student.training
    data = model.prepare(_intensities()[None]).float().requires_grad_()
    solver(data, model.imaging_model, Sampling(shape=(5, 40, 44), pixel_size=PIXEL_SIZE)).estimate.sum().backward()
    assert all(parameter.grad is not None for parameter in solver.student.parameters())
    assert data.grad is not None
    assert data.grad.abs().max() > 0


def well(context: SolveContext) -> dict[str, float]:
    """Per-position values: the column of the well a position is in (``A/3/0`` is column 3)."""
    return {"column": float((context.position or "A/0/0").split("/")[1])}


def test_conditions_are_computed_from_the_solve_context_and_the_attributes_function(tmp_path):
    student = tiny.strided(5, conditioning=Bias(sizes={"field_position": 2, "well": 1}))
    path = _phase_bundle(
        tmp_path / "placed",
        student=student,
        conditions=(FieldPosition(), Attributes(["column"], name="well")),
        test=Example(
            data=torch.randn(1, 1, 5, 32, 32),
            condition={"field_position": torch.full((1, 2), 0.5), "well": torch.ones(1, 1)},
        ),
    )
    model = thin_phase_model()
    model.solver = Distilled(load_bundle(path), attributes=well)
    data = _intensities(height=32, width=32)
    whole = model.reconstruct(data, pixel_size=PIXEL_SIZE).estimate[0]
    centre = torch.tensor([[0.5, 0.5]])
    torch.testing.assert_close(whole, _by_hand(model, data, field_position=centre, well=torch.zeros(1, 1))[0])
    with solve_context(position="A/3/0", field_shape=(64, 64), region=(slice(32, 64), slice(0, 32))):
        placed = model.reconstruct(data, pixel_size=PIXEL_SIZE).estimate[0]
    corner = torch.tensor([[0.75, 0.25]])
    torch.testing.assert_close(placed, _by_hand(model, data, field_position=corner, well=torch.full((1, 1), 3.0))[0])
    assert not torch.allclose(placed, whole)
    pickle.dumps(model.solver)  # a module-level function pickles


def test_a_field_scope_input_normalizer_refuses_part_of_a_field(tmp_path):
    path = _phase_bundle(tmp_path / "field", input_normalizer=FieldStandardize())
    model = thin_phase_model()
    model.solver = Distilled(load_bundle(path))
    data = _intensities(height=32, width=32)
    model.reconstruct(data, pixel_size=PIXEL_SIZE)
    refused = "FieldStandardize normalizes by statistics of the whole field"
    with (
        solve_context(field_shape=(64, 64), region=(slice(0, 32), slice(0, 32))),
        pytest.raises(IncompatibleModelError, match=refused),
    ):
        model.reconstruct(data, pixel_size=PIXEL_SIZE)


def test_a_solver_whose_parts_disagree_is_refused_when_built(phase_bundle):
    bundle = load_bundle(phase_bundle)
    contract = bundle.contract
    with pytest.raises(ParameterError, match="the student returns 2 field"):
        Distilled(Bundle(student=tiny.strided(5, fields=2), contract=contract))
    with pytest.raises(ParameterError, match="the contract accepts 5 planes, but StridedVolume takes stacks of 3"):
        Distilled(Bundle(student=tiny.strided(3), contract=contract))
    with pytest.raises(ParameterError, match="reads the condition 'field_position'"):
        Distilled(Bundle(student=tiny.strided(5, conditioning=Bias(sizes={"field_position": 2})), contract=contract))
    with pytest.raises(ParameterError, match="does not read the condition"):
        Distilled(Bundle(student=bundle.student, contract=contract, conditions=(FieldPosition(),)))
    with pytest.raises(ParameterError, match="cannot be inverted voxel by voxel"):
        Distilled(Bundle(student=bundle.student, contract=contract, target_normalizer=FieldStandardize()))
