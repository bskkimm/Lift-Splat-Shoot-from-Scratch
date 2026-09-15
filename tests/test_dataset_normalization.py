from PIL import Image
from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_dataset_returns_normalized_images(tmp_path):
    p=tmp_path/"x.png"; Image.new("RGB",(2,2),(255,255,255)).save(p); r={"image_paths":[str(p)],"intrinsics":[[[1,0,0],[0,1,0],[0,0,1]]],"extrinsics":[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]}; assert NuScenesCameraDataset([r])[0]["images"].mean() > 1
