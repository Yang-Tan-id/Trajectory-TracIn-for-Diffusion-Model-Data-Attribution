import pytest

from sd_artbench.train_lora import checkpoint_epochs


def test_twenty_five_checkpoints_across_one_hundred_epochs():
    epochs = checkpoint_epochs(100, 25)

    assert len(epochs) == 25
    assert sorted(epochs) == list(range(4, 101, 4))


def test_checkpoint_count_cannot_exceed_epoch_count():
    with pytest.raises(ValueError):
        checkpoint_epochs(10, 25)
