import pytest
import torch

from tests.data.plates import raw_field, target_field
from waveorder.data import Recorded
from waveorder.data.sources import read_field
from waveorder.errors import ReadError


def test_read_field_reads_the_named_channels_in_order(plates, index):
    data = read_field(plates.inputs, "A/1/0", time=0, channels=["BF", "GFP"])
    assert torch.equal(data, raw_field("A/1/0")[0][[1, 0]])
    with pytest.raises(ReadError, match=r"A/1/0: it has no channel DAPI; it has GFP, BF"):
        read_field(plates.inputs, "A/1/0", time=0, channels=["DAPI"])
    data, target = Recorded().read(index[0], channels=["GFP"], target_channels=["Focus3D"])
    assert torch.equal(data, raw_field(index[0].position)[0][:1])
    assert torch.equal(target, target_field(index[0].position)[0][1:])
