import iohub
import pytest
import torch

from tests.tiling.conftest import blur_model, create_sink
from waveorder.errors import ParameterError, ShapeError
from waveorder.io.positions import pixel_size
from waveorder.nn import Wiener
from waveorder.tiling import TileOptions, TilePlan, reconstruct_series
from waveorder.tiling.merge import volumes

SHAPE = (12, 48, 40)
PIXEL = (0.5, 0.25, 0.25)


@pytest.fixture
def plate(tmp_path):
    """An HCS plate with one position: 3 time points, channels BF, GFP, RFP, chunked (1, 1, 6, 16, 16)."""
    generator = torch.Generator().manual_seed(0)
    data = torch.randn(3, 3, *SHAPE, generator=generator)
    path = tmp_path / "raw.zarr"
    with iohub.open_ome_zarr(path, layout="hcs", mode="w-", channel_names=["BF", "GFP", "RFP"]) as created:
        position = created.create_position("A", "1", "0")
        position.create_image("0", data.numpy(), chunks=(1, 1, 6, 16, 16))
        for axis, size in zip("zyx", PIXEL, strict=True):
            position.set_scale("0", axis, size)
    return path, data


def _plan():
    return TilePlan(shape=SHAPE, tile=(12, 16, 16), overlap=(0, 4, 4), halo=(0, 4, 4), chunks=(6, 16, 16))


def test_reconstructs_selected_channels_and_times_of_an_hcs_position(plate, tmp_path):
    path, data = plate
    model, solver = blur_model(1.0), Wiener()
    with iohub.open_ome_zarr(path, mode="r") as opened:
        source = opened["A/1/0"]
        assert pixel_size(source) == PIXEL
        with create_sink(str(tmp_path / "recon.zarr"), like=source, fields=["f"]) as sink:
            series = volumes(source, sink, channels=["RFP"], times=[2, 0])
            stats = reconstruct_series(
                series,
                model=model,
                solver=solver,
                pixel_size=pixel_size(source),
                plan=_plan(),
                device="cpu",
                options=TileOptions(write_workers=2),
            )
    assert stats.tiles == 2 * len(_plan().regions)
    # Each chunk of the selected channel was decoded once per time point: nothing else was read.
    assert stats.bytes_decoded == 2 * data[0, 2].numel() * data.element_size()
    with iohub.open_ome_zarr(tmp_path / "recon.zarr", mode="r") as written:
        result = torch.as_tensor(written["0"][:])
        assert written.channel_names == ["f"]
        assert tuple(written.get_effective_scale("0")[-3:]) == PIXEL
    for t in (0, 2):
        expected = torch.full((1, *SHAPE), float("nan"))
        reconstruct_series(
            [(data[t, 2:3].clone(), expected)], model=model, solver=solver, pixel_size=PIXEL, plan=_plan(), device="cpu"
        )
        torch.testing.assert_close(result[t], expected, rtol=0, atol=0)
    assert torch.all(result[1] == 0)  # time point 1 was not selected


def test_several_channels_map_onto_data_channels_in_the_given_order(plate, tmp_path):
    path, data = plate
    with iohub.open_ome_zarr(path / "A" / "1" / "0", mode="r") as source:
        with create_sink(str(tmp_path / "out.zarr"), like=source, fields=["a", "b"]) as sink:
            (view, out), *_ = volumes(source, sink, channels=["RFP", "BF"], times=[1])
            torch.testing.assert_close(
                view[(slice(None), slice(0, 4), slice(0, 8), slice(8, 16))], data[1, [2, 0], :4, :8, 8:16]
            )
            out[(slice(0, 2), slice(0, 4), slice(0, 8), slice(8, 16))] = torch.ones(2, 4, 8, 8).numpy()
            assert view.chunks == (1, 6, 16, 16)
            assert out.shape == (2, *SHAPE)
        with iohub.open_ome_zarr(tmp_path / "out.zarr", mode="r") as written:
            stored = torch.as_tensor(written["0"][:])
    assert stored[1, :, :4, :8, 8:16].eq(1).all()
    assert stored.sum() == 2 * 4 * 8 * 8


@pytest.mark.parametrize(
    ("kwargs", "error", "path"),
    [
        ({"channels": ["DAPI"]}, ParameterError, "channels.0"),
        ({"channels": ["GFP", "GFP"]}, ParameterError, "channels"),
        ({"channels": ["GFP"], "times": [3]}, ParameterError, "times.0"),
        ({"channels": ["GFP"], "fields": ["g"]}, ParameterError, "fields.0"),
    ],
)
def test_volumes_name_the_bad_selection(plate, tmp_path, kwargs, error, path):
    raw, _ = plate
    with (
        iohub.open_ome_zarr(raw / "A" / "1" / "0", mode="r") as source,
        create_sink(str(tmp_path / "out.zarr"), like=source, fields=["f"]) as sink,
        pytest.raises(error) as caught,
    ):
        volumes(source, sink, **kwargs)
    assert caught.value.path == path


def test_volumes_reject_a_sink_of_another_geometry_or_with_time_chunks(plate, tmp_path):
    raw, _ = plate
    with iohub.open_ome_zarr(raw / "A" / "1" / "0", mode="r") as source:
        with (
            create_sink(str(tmp_path / "short.zarr"), like=source, fields=["f"], times=2) as sink,
            pytest.raises(ShapeError, match="extents must match"),
        ):
            volumes(source, sink, channels=["GFP"])
        with iohub.open_ome_zarr(tmp_path / "chunky.zarr", layout="fov", mode="w-", channel_names=["f"]) as sink:
            sink.create_zeros("0", shape=(3, 1, *SHAPE), dtype="float32", chunks=(3, 1, *SHAPE))
            with pytest.raises(ParameterError, match="span 3 time points") as caught:
                volumes(source, sink, channels=["GFP"])
    assert caught.value.path == "sink"
