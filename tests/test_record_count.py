from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_record_count_matches_length():
    dataset=NuScenesCameraDataset([]); assert dataset.record_count == 0 and len(dataset) == 0
