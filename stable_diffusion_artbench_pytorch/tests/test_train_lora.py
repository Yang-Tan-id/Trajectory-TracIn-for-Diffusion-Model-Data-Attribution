import pytest

from sd_artbench.train_lora import checkpoint_epochs, resolve_resume_checkpoint


def test_twenty_five_checkpoints_across_one_hundred_epochs():
    epochs = checkpoint_epochs(100, 25)

    assert len(epochs) == 25
    assert sorted(epochs) == list(range(4, 101, 4))


def test_checkpoint_count_cannot_exceed_epoch_count():
    with pytest.raises(ValueError):
        checkpoint_epochs(10, 25)


def test_zero_checkpoints_means_final_weights_only():
    assert checkpoint_epochs(100, 0) == set()


def test_resolve_latest_checkpoint(tmp_path):
    (tmp_path / "checkpoint-epoch-0004").mkdir()
    expected = tmp_path / "checkpoint-epoch-0008"
    expected.mkdir()

    assert resolve_resume_checkpoint(tmp_path, "latest") == expected
