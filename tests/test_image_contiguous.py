from PIL import Image
from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_dataset_images_are_contiguous(tmp_path):
    p=tmp_path/"x.png"; Image.new("RGB",(1,1)).save(p); r={"image_paths":[str(p)],"intrinsics":[[[1,0,0],[0,1,0],[0,0,1]]],"extrinsics":[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]}; assert NuScenesCameraDataset([r])[0]["images"].is_contiguous()
