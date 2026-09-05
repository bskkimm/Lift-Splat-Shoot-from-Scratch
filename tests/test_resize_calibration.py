from PIL import Image
from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_resize_scales_intrinsics(tmp_path):
    p=tmp_path/"x.png"; Image.new("RGB",(10,20)).save(p); r={"image_paths":[str(p)],"intrinsics":[[[10,0,5],[0,20,10],[0,0,1]]],"extrinsics":[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]}; assert NuScenesCameraDataset([r],(10,5))[0]["intrinsics"][0,0,0] == 5
