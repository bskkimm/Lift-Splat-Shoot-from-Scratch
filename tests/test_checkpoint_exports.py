from lss.checkpoint import checkpoint_epoch, checkpoint_metadata_json, load_checkpoint, resume_checkpoint
def test_checkpoint_helpers_are_public(): assert all(callable(x) for x in (checkpoint_epoch,checkpoint_metadata_json,load_checkpoint,resume_checkpoint))
