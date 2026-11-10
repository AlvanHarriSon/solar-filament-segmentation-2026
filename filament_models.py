"""Inference models and mask processing for full-disk solar filament segmentation.

The ROI network and crop/augmentation recipe adapt XingJia He's ref-train-v2
(MIT). See the third-party notices included with this repository.
"""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

MEAN = np.array([.485,.456,.406],np.float32)[:,None,None]
STD = np.array([.229,.224,.225],np.float32)[:,None,None]
CLAHE = cv2.createCLAHE(2.,(8,8))
FEATURES = ["detector_score", "mask_probability", "log_area", "fill_ratio", "box_fraction",
            "log_aspect", "ring_minus_inside_gray", "radial_distance"]
FUSION = dict(box_iou=.4, box_smaller_overlap=.8, mask_iou=.4,
              mask_smaller_overlap=.75, min_unoccupied_fraction=.7, min_area=96)


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        r = resnet18(weights=None)
        self.stem = nn.Sequential(r.conv1,r.bn1,r.relu)
        self.pool,self.l1,self.l2,self.l3,self.l4 = r.maxpool,r.layer1,r.layer2,r.layer3,r.layer4
        self.d3,self.d2,self.d1,self.d0 = self.up(768,256),self.up(384,128),self.up(192,64),self.up(128,32)
        self.head = nn.Conv2d(32,1,1)

    @staticmethod
    def up(cin,cout):
        return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(cin,cout,3,padding=1),nn.BatchNorm2d(cout),nn.ReLU(inplace=True),
            nn.Conv2d(cout,cout,3,padding=1),nn.BatchNorm2d(cout),nn.ReLU(inplace=True))

    def forward(self,x):
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0));s2 = self.l2(s1);s3 = self.l3(s2);s4 = self.l4(s3)
        d = self.d3(torch.cat([F.interpolate(s4,scale_factor=2,mode='bilinear',align_corners=False),s3],1))
        d = self.d2(torch.cat([d,s2],1));d = self.d1(torch.cat([d,s1],1));d = self.d0(torch.cat([d,s0],1))
        return self.head(d)


def patch_channels(gray,x0,y0,side,box):
    side = int(side)
    if side<96 or side>4096:
        raise ValueError('Invalid ROI side')
    h,w = gray.shape
    xs0,ys0,xs1,ys1 = max(0,x0),max(0,y0),min(w,x0+side),min(h,y0+side)
    patch = np.zeros((side,side),np.uint8)
    if xs1>xs0 and ys1>ys0:
        patch[ys0-y0:ys1-y0,xs0-x0:xs1-x0] = gray[ys0:ys1,xs0:xs1]
    seed = np.zeros((side,side),np.float32)
    bx0,by0,bx1,by1 = max(xs0,int(box[0])),max(ys0,int(box[1])),min(xs1,int(box[2])),min(ys1,int(box[3]))
    if bx1>bx0 and by1>by0:
        seed[by0-y0:by1-y0,bx0-x0:bx1-x0] = 1
    x = np.stack([patch.astype(np.float32)/255,CLAHE.apply(patch).astype(np.float32)/255,seed],axis=-1)
    return cv2.resize(x,(256,256),interpolation=cv2.INTER_LINEAR).transpose(2,0,1).copy()


@torch.inference_mode()
def refine_prob(gray,box,model,device='cuda'):
    bx0,by0,bx1,by1 = map(float,box)
    bside = max(bx1-bx0,by1-by0)
    side = max(int(bside*1.5),96)
    x0,y0 = int((bx0+bx1)/2-side//2),int((by0+by1)/2-side//2)
    channels = (patch_channels(gray,x0,y0,side,box)-MEAN)/STD
    tensor = torch.from_numpy(channels)[None].to(device)
    views = torch.cat([tensor,tensor.flip(3),tensor.flip(2),tensor.flip((2,3))],0)
    with torch.autocast('cuda',dtype=torch.float16,enabled=torch.device(device).type=='cuda'):
        p = model(views).float().sigmoid()[:,0]
    p = (p[0]+p[1].flip(1)+p[2].flip(0)+p[3].flip((0,1)))*.25
    probability = cv2.resize(p.cpu().numpy(),(side,side),interpolation=cv2.INTER_LINEAR)
    full = np.zeros(gray.shape,np.float32)
    xs0,ys0,xs1,ys1 = max(0,x0),max(0,y0),min(gray.shape[1],x0+side),min(gray.shape[0],y0+side)
    if xs1>xs0 and ys1>ys0:
        full[ys0:ys1,xs0:xs1] = probability[ys0-y0:ys1-y0,xs0-x0:xs1-x0]
    if not np.isfinite(full).all():raise FloatingPointError('Nonfinite refiner probability')
    return full,bside


def roi_geometry(box):
    side = max(96, int(1.5 * max(box[2] - box[0], box[3] - box[1])))
    return int((box[0] + box[2]) / 2 - side // 2), int((box[1] + box[3]) / 2 - side // 2), side


def box_duplicate(box, previous):
    x, y, right, bottom = box
    a = (right - x) * (bottom - y)
    for px, py, pr, pb in previous:
        b = (pr - px) * (pb - py)
        inter = max(0., min(right, pr) - max(x, px)) * max(0., min(bottom, pb) - max(y, py))
        if inter >= FUSION["box_iou"] * (a + b - inter) or inter >= FUSION["box_smaller_overlap"] * min(a, b):
            return True
    return False


def extra_boxes(original, secondary):
    accepted = []
    previous = [d["box"] for d in original]
    for detection in sorted(secondary, key=lambda d: -d["score"]):
        if not box_duplicate(detection["box"], previous):
            accepted.append(detection)
            previous.append(detection["box"])
    return accepted


def add_instance(labels, areas, mask):
    occupied = np.bincount(labels[mask], minlength=len(areas))
    area = int(occupied.sum())
    free = int(occupied[0])
    if free < FUSION["min_area"] or free < FUSION["min_unoccupied_fraction"] * area:
        return False
    for idx in np.flatnonzero(occupied[1:]) + 1:
        inter = occupied[idx]
        other = areas[idx]
        if (inter >= FUSION["mask_iou"] * (area + other - inter)
                or inter >= FUSION["mask_smaller_overlap"] * min(area, other)):
            return False
    labels[mask & (labels == 0)] = len(areas)
    areas.append(free)
    return True


def boxed_values(array, x, y, side):
    crop = np.zeros((side, side), array.dtype)
    h, w = array.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x+side), min(h, y+side)
    crop[y0-y:y1-y, x0-x:x1-x] = array[y0:y1, x0:x1]
    return crop


def mask_features(gray, labels, instance, detection, probability):
    mask = labels == instance
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    selection = np.s_[max(0, y-8):min(2048, y+h+8), max(0, x-8):min(2048, x+w+8)]
    inside = mask[selection]
    ring = cv2.dilate(inside.astype(np.uint8), np.ones((9,9), np.uint8)).astype(bool) & ~inside
    local = gray[selection]
    record = dict(detection, instance=instance, area=int(inside.sum()), mask_box=[x,y,w,h],
                  probability=float(probability[mask].mean()), inside_gray=float(local[inside].mean()),
                  ring_gray=float(local[ring].mean()) if ring.any() else None)
    x0, y0, side = roi_geometry(detection['box'])
    channels = patch_channels(gray,x0,y0,side,detection['box'])
    spatial = [cv2.resize(c,(32,32),interpolation=cv2.INTER_AREA) for c in channels]
    for c in (probability,mask.astype(np.uint8)):
        plane = boxed_values(c,x0,y0,side).astype(np.float32)
        spatial.append(cv2.resize(plane,(32,32),interpolation=cv2.INTER_AREA))
    return record, np.rint(np.clip(spatial,0,1)*255).astype(np.uint8)


def attributes(row):
    x,y,w,h=row['mask_box']
    x0,y0,x1,y1=row['box']
    area=row['area']
    diff=0. if row['ring_gray'] is None else row['ring_gray']-row['inside_gray']
    return [row['score'],row['probability'],np.log1p(area),area/(w*h),
            area/max(1.,(x1-x0)*(y1-y0)),np.log(max(w,h)/min(w,h)),diff,
            np.hypot(x+w*.5-1024,y+h*.5-1024)/1024.]


def quality_scores(records, model):
    if not records:
        return np.array([],float)
    x=np.asarray([attributes(row) for row in records])
    z=np.clip((x-np.array(model['center'])) / np.array(model['scale']), -4.,4.)
    return model['intercept']+z@np.array(model['coef'])


class QualityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(8))
        self.register_buffer("scale", torch.ones(8))
        self.spatial = nn.Sequential(
            nn.Conv2d(5, 12, 3, padding=1), nn.GroupNorm(3, 12), nn.SiLU(), nn.AvgPool2d(2),
            nn.Conv2d(12, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(), nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
        )
        self.outputs = nn.Sequential(nn.Linear(40, 24), nn.SiLU(), nn.Linear(24, 3))

    def forward(self, spatial, scalars):
        x = spatial.float() / 255.
        image = x[:, :2]
        contrast = (image - image.mean((2, 3), keepdim=True)) / image.std((2, 3), keepdim=True).clamp_min(.05)
        f = self.spatial(torch.cat((contrast.clamp(-4, 4), x[:, 2:]), 1))
        mask = F.adaptive_avg_pool2d(x[:, 4:5], f.shape[-2:])
        in_mask = (f * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(.01)
        scalars = ((scalars - self.center) / self.scale).clamp(-4, 4)
        return self.outputs(torch.cat((f.mean((2, 3)), in_mask, scalars), 1))


def expected_iou(output):
    return output[:, 0].sigmoid() * (.5 + .5 * output[:, 1].sigmoid())


def predict_scores(models, spatial_uint8, scalars_float):
    if len(spatial_uint8) == 0:
        return np.empty(0, dtype=np.float64)
    if spatial_uint8.shape != (len(spatial_uint8), 5, 32, 32) or spatial_uint8.dtype != np.uint8:
        raise ValueError("Invalid spatial candidate inputs")
    if scalars_float.shape != (len(spatial_uint8), 8) or not np.isfinite(scalars_float).all():
        raise ValueError("Invalid scalar candidate inputs")
    device = next(models[0].parameters()).device
    with torch.inference_mode():
        x = torch.as_tensor(spatial_uint8, device=device)
        a = torch.as_tensor(scalars_float, device=device, dtype=torch.float32)
        all_scores = [expected_iou(m(x, a)).cpu().numpy() for m in models]
    scores = np.mean(all_scores, axis=0, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite mask-quality score")
    return scores


def subset_labels(labels, keep):
    if labels.max() != len(keep):
        raise ValueError("Native label IDs and policy inputs disagree")
    mapping = np.zeros(len(keep)+1, dtype=np.int32)
    mapping[np.flatnonzero(keep)+1] = np.arange(1, np.count_nonzero(keep)+1)
    return mapping[labels]
