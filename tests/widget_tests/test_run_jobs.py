"""RUN on a tiny simulated plate: ``wo reconstruct`` runs as a job, writes the output, and a failure is shown."""

import pytest

pytest.importorskip("napari")
pytest.importorskip("pytestqt")

import iohub

from tests.cli.plates import FIXED_PHASE, intensities, write_plate
from tests.widget_tests.conftest import set_input

JOB_TIMEOUT_MS = 300_000


@pytest.fixture
def plate(tmp_path, phase_volume):
    path = tmp_path / "in.zarr"
    write_plate(path, [intensities(phase_volume)[None], intensities(-phase_volume)[None]])
    return path


@pytest.fixture(scope="module")
def phase_volume():
    from tests.cli.plates import simulate_phase

    return simulate_phase()


def _load(tab, tmp_path, text, name="phase.yml"):
    path = tmp_path / name
    path.write_text(text)
    (entry,) = tab.open_model_files([str(path)])
    return entry


def test_run_reconstructs_the_selected_positions_and_shows_the_result(recon_tab, messages, plate, tmp_path, qtbot):
    set_input(recon_tab, plate)
    assert recon_tab.input_positions == ["A/1/0", "A/1/1"]
    assert recon_tab.data_output_LineEdit.value == str(tmp_path)
    recon_tab.data_positions_Select.value = ["A/1/0"]
    entry = _load(recon_tab, tmp_path, FIXED_PHASE)
    assert entry.output_line.value == "in_Pha_01.zarr"

    recon_tab.reconstruction_run_PushButton.native.click()
    (job,) = recon_tab.jobs.values()
    assert job.info.text().splitlines()[1].startswith(f"$ wo reconstruct -i {plate}/A/1/0 -c ")
    qtbot.waitUntil(lambda: not recon_tab.jobs, timeout=JOB_TIMEOUT_MS)

    assert [text for text, error in messages if error] == []
    output = tmp_path / "in_Pha_01.zarr"
    with iohub.open_ome_zarr(output, mode="r") as result:
        assert [name for name, _ in result.positions()] == ["A/1/0"]
        assert list(result.channel_names) == ["Phase3D"]
    (written,) = tmp_path.glob("phase-*-01.yml")
    assert "contrast: phase" in written.read_text()
    assert recon_tab.proc_table_QFormLayout.rowCount() == 0  # a job that succeeds leaves the queue
    assert len(recon_tab.viewer.layers) > 0  # Show after Reconstruction


def test_a_failing_job_keeps_its_entry_and_shows_its_error(recon_tab, messages, plate, tmp_path, qtbot):
    set_input(recon_tab, plate)
    entry = _load(recon_tab, tmp_path, FIXED_PHASE.replace("input_channel_names: [BF]", "input_channel_names: [GFP]"))
    entry.show.value = False

    recon_tab.reconstruction_run_PushButton.native.click()
    (job,) = recon_tab.jobs.values()
    qtbot.waitUntil(lambda: any("failed with return code" in text for text, _ in messages), timeout=JOB_TIMEOUT_MS)

    (failure,) = [text for text, error in messages if error]
    assert failure.startswith(f"{job.title} failed with return code ")
    assert "GFP" in failure
    assert "Process ended with return code" in job.info.text()
    assert recon_tab.proc_table_QFormLayout.rowCount() == 1
    assert not (tmp_path / "in_Pha_01.zarr" / "A" / "1" / "0" / "0").exists()


def test_run_without_a_model_asks_for_one(recon_tab, messages):
    recon_tab.reconstruction_run_PushButton.native.click()
    assert messages == [("Please create a processing model first !", True)]
    assert recon_tab.jobs == {}


def test_an_acquisition_in_progress_is_reconstructed_on_the_fly(recon_tab, plate, tmp_path, monkeypatch):
    with iohub.open_ome_zarr(plate, mode="r+") as dataset:
        dataset.zattrs["FinalDimensions"] = {"time": 3, "position": 2, "z": 8, "channel": 1}
        dataset.zattrs["CurrentDimensions"] = {"time": 1, "position": 2, "z": 8, "channel": 1}
    set_input(recon_tab, plate)
    assert recon_tab.poll_data  # the confirmation answered yes
    entry = _load(recon_tab, tmp_path, FIXED_PHASE)
    submitted = []
    monkeypatch.setattr(recon_tab, "submit", lambda _entry, config, **_: submitted.append(config.data.time_indices))

    recon_tab.build_model_and_run()
    assert entry.poll_data
    assert submitted == []  # time point 0 is still being acquired
    assert recon_tab.proc_OTF_table_QFormLayout.rowCount() == 1

    with iohub.open_ome_zarr(plate, mode="r+") as dataset:
        dataset.zattrs["CurrentDimensions"] = {"time": 3, "position": 2, "z": 8, "channel": 1}
    recon_tab.poll(str(plate))
    assert submitted == [[0, 1, 2]]
    assert recon_tab.polls == {}
    assert recon_tab.proc_OTF_table_QFormLayout.rowCount() == 0
