"""``wo rec``, ``wo apply-inv-tf`` and ``wo compute-tf``: reconstruct a dataset, or write its transfer functions.

Each command builds the model of the configuration (:meth:`waveorder.models.ReconstructionModel.from_config`).
``wo rec`` on a configuration with free quantities calibrates first (``model.calibrate_position``) and publishes the
store beside the output; ``wo apply-inv-tf`` loads the ``-t`` store's values and transfer functions
(``model.load_calibration``). Both then call ``model.reconstruct_positions`` with one argument per flag and print the
summary on stdout. ``wo compute-tf`` calls ``model.to_tf_store``. The ``data`` block gives the channels, the voxel
size and the time points. Every check runs before anything is computed, and what the store and the positions each
refuse is reported together.
"""

from waveorder.cli.options import (
    ConfigFilepath,
    Device,
    Devices,
    InputPositions,
    KeepGoing,
    NumProcesses,
    OutputDirpath,
    OutputOverwrite,
    Overwrite,
    Resume,
    Tile,
    TileOverlap,
    TransferFunctionDirpath,
    Wandb,
    WriteConfigScaleToOutput,
    parse_zyx,
)


def reconstruct(  # noqa: PLR0913 (one keyword per flag)
    *,
    input_position_dirpaths: InputPositions,
    config_filepath: ConfigFilepath,
    output_dirpath: OutputDirpath,
    device: Devices = None,
    tile: Tile = None,
    tile_overlap: TileOverlap = None,
    num_processes: NumProcesses = None,
    overwrite: OutputOverwrite = False,
    resume: Resume = False,
    keep_going: KeepGoing = False,
    write_config_scale_to_output: WriteConfigScaleToOutput = False,
    wandb: Wandb = None,
) -> None:
    """Reconstruct every position and time of a dataset into an output plate.

    Builds the model of the configuration and reconstructs each (position, time) unit, spread over the devices of
    --device. A configuration with free quantities is calibrated first, on the first position at its first selected
    time; the calibration is written beside the output as <output>_transfer_function_<config>.zarr. The output
    plate is created once, then filled unit by unit; --resume skips the units an interrupted run finished (and
    reuses the calibration it published), else a rerun reconstructs every unit it is given again.

    Example: wo rec -i 'in.zarr/*/*/*' -c phase.yml -o out.zarr
    """
    from waveorder.cli import _reconstruct

    _reconstruct.reconstruct(
        _reconstruct.Flags(
            command="reconstruct",
            inputs=tuple(str(path) for path in input_position_dirpaths),
            config=config_filepath,
            output=output_dirpath,
            store=None,
            device=device,
            tile=parse_zyx(tile, flag="--tile"),
            tile_overlap=parse_zyx(tile_overlap, flag="--tile-overlap", minimum=0),
            num_processes=num_processes,
            overwrite=overwrite,
            resume=resume,
            keep_going=keep_going,
            write_config_scale_to_output=write_config_scale_to_output,
            wandb=wandb,
        )
    )


def apply_inverse_transfer_function(  # noqa: PLR0913 (one keyword per flag)
    *,
    input_position_dirpaths: InputPositions,
    config_filepath: ConfigFilepath,
    transfer_function_dirpath: TransferFunctionDirpath,
    output_dirpath: OutputDirpath,
    device: Devices = None,
    tile: Tile = None,
    tile_overlap: TileOverlap = None,
    num_processes: NumProcesses = None,
    overwrite: OutputOverwrite = False,
    resume: Resume = False,
    keep_going: KeepGoing = False,
    write_config_scale_to_output: WriteConfigScaleToOutput = False,
    wandb: Wandb = None,
) -> None:
    """Reconstruct a dataset with a transfer-function store's parameters, without calibrating.

    Builds the model of the configuration, loads the store's parameters as they are and reconstructs each
    (position, time) unit, as wo rec does; --resume skips the units an interrupted run finished. A transfer function
    the store holds (wo compute-tf) is read, not computed, where it fits; a line on stderr says where.

    Example: wo apply-inv-tf -i 'in.zarr/*/*/*' -c phase.yml -t tf.zarr -o out.zarr
    """
    from waveorder.cli import _reconstruct

    _reconstruct.reconstruct(
        _reconstruct.Flags(
            command="apply-inverse-transfer-function",
            inputs=tuple(str(path) for path in input_position_dirpaths),
            config=config_filepath,
            output=output_dirpath,
            store=transfer_function_dirpath,
            device=device,
            tile=parse_zyx(tile, flag="--tile"),
            tile_overlap=parse_zyx(tile_overlap, flag="--tile-overlap", minimum=0),
            num_processes=num_processes,
            overwrite=overwrite,
            resume=resume,
            keep_going=keep_going,
            write_config_scale_to_output=write_config_scale_to_output,
            wandb=wandb,
        )
    )


def compute_transfer_function(
    *,
    input_position_dirpaths: InputPositions,
    config_filepath: ConfigFilepath,
    output_dirpath: OutputDirpath,
    device: Device = None,
    overwrite: Overwrite = False,
) -> None:
    """Write a tf.zarr at the configuration's nominal parameters, with transfer functions per shape.

    The shapes, and with data.pixel_size: auto the voxel size, come from the -i positions' metadata (no pixels are
    read). The transfer functions, on each shape after z padding and computed on --device, are stored when the
    configuration's outputs ask for them; wo apply-inv-tf then reads them from the store instead of computing them.

    Example: wo compute-tf -i in.zarr/A/1/0 -c phase.yml -o tf.zarr
    """
    from waveorder.cli import _reconstruct

    _reconstruct.compute_transfer_function(
        config_filepath, input_position_dirpaths, output_dirpath, device=device, overwrite=overwrite
    )
