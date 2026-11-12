"""BEV feature decoding and assembly of the complete segmentation model."""

from .canvas import Canvas


def bev_decoder(data):
    c = Canvas(8, "Decode BEV features with a long skip", "A ResNet-block encoder recovers detail through concatenation and bilinear interpolation.")
    c.text(34, 153, "Input: [B,64,200,200] → 7×7 stride-2 stem (no max-pool)", 18, "teal", True)
    cards = [(34, 218, 292, "Stem + layer1", "[B,64,100,100]", "blue"),
             (376, 218, 292, "layer2", "[B,128,50,50]", "purple"),
             (718, 218, 292, "layer3", "[B,256,25,25]", "purple"),
             (1060, 218, 306, "Upsample to 100²", "[B,256,100,100]", "purple")]
    for x, y, w, title, body, tone in cards:
        c.card(x, y, w, 102, title, body, tone, 18)
    for x in (326, 668, 1010):
        c.arrow((x, 269), (x + 48, 269), tone="purple")
    c.path([(180, 320), (180, 416), (474, 416)], "blue")
    c.text(192, 385, "64-channel skip", 16, "blue")
    c.path([(1213, 320), (1213, 416), (766, 416)], "purple")
    c.text(875, 385, "256 channels", 16, "purple")
    c.card(474, 369, 292, 94, "CONCAT: 64 + 256", "[B,320,100,100]", "teal", 18)
    c.arrow((620, 463), (620, 514), tone="teal")
    c.card(408, 514, 424, 100, "Two 3×3 Conv + BN + ReLU", "320 → 256 → 256 at 100 × 100", "teal", 17)
    c.arrow((832, 564), (914, 564), tone="teal")
    c.card(914, 514, 452, 100, "Upsample + output head", "200² → Conv 128 → BN/ReLU → Conv 1", "gold", 18)
    c.text(34, 664, "Residual blocks: zero-init residual normalization", 17, "muted")
    c.text(1140, 668, "raw logits [B,1,200,200]", 19, "gold", True, "center")
    c.note("Full practical shapes. Sigmoid is applied for display/decoding, not inside the model or before BCEWithLogits.")
    return c


def assembly(data):
    c = Canvas(9, "Assemble the model: values + coordinates", "Two streams must agree on B, N, depth and image sites before voxel reduction.")
    c.card(34, 169, 302, 103, "Images", "[1,6,3,128,352]", "blue")
    c.arrow((336, 220), (395, 220), tone="blue")
    c.card(395, 155, 435, 130, "Camera encoder", "depth p: [1,6,41,8,22]\ncontext c: [1,6,64,8,22]", "blue")
    c.arrow((830, 220), (890, 220), tone="blue")
    c.card(890, 169, 476, 103, "Outer product → lifted values", "[1,6,41,8,22,64]", "purple", 18)
    c.card(34, 345, 302, 130, "Calibration + aug", "K [B,N,3,3]\nT [B,N,4,4]", "teal", 16)
    c.arrow((336, 410), (395, 410))
    c.card(395, 345, 435, 130, "Pixel/depth frustum → ego XYZ", "[41,8,22,3] → [1,6,41,8,22,3]\nundo post-transform; K⁻¹; R,t", "teal", 17)
    c.arrow((830, 410), (890, 410))
    c.card(890, 342, 476, 134, "Voxel mask, indices, sum", "same sample sites as lifted values\n[B,Z·C,X,Y] = [1,64,200,200]", "teal", 18)
    c.arrow((1128, 272), (1128, 342), "values", "purple")
    c.arrow((1128, 476), (1128, 551))
    c.card(890, 551, 476, 111, "BEV decoder → logits", "[1,1,200,200]", "gold", 20)
    c.card(34, 550, 796, 112, "Z=1 pillar; x/y from −50 to +50 m at 0.5 m",
           "Depth: 4,…,44 m  ·  Context: 64 channels  ·  No supervised depth target", "purple", 18)
    c.note("Shape trace uses six validation cameras. The reference-data training recipe samples five; tiny CPU probes use smaller grids.")
    return c
