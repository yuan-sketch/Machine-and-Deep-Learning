# 袁超
# 开发时间：2026/5/25 12:32
import numpy as np
import math
import torch
from PIL import Image
import d2lzh8 as d2l

d2l.set_figsize()
img = Image.open('data/catdog.jpg')
w, h = img.size
print('w = %d, h = %d'%(w,h))

def MultiBoxPrior(feature_map_size, sizes=[0.75,0.5,0.25],ratios=[1,2,0.5]):
    pairs = []
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])
    pairs = np.array(pairs)
    # [[0.75       1.        ]
    #  [0.75       1.41421356]
    #  [0.75       0.70710678]
    #  [0.5        1.        ]
    #  [0.25       1.        ]]
    # pairs[:,0]:[0.75 0.75 0.75 0.5  0.25]
    # pairs[:,1]:[1.         1.41421356 0.70710678 1.         1.        ]
    ss1 = pairs[:,0]*pairs[:,1]
    ss2 = pairs[:,0]/pairs[:,1]
    # 除以2是因为像素点在锚框中心位置‘
    base_anchors = np.stack([-ss1, -ss2, ss1, ss2],axis=1) / 2
    if isinstance(feature_map_size, (tuple, list)):
        h, w = feature_map_size
    else:
        h, w = feature_map_size.shape[-2:]
    shifts_x = np.arange(0, w) / w
    shifts_y = np.arange(0,h) / h
    # 生成画布内所有坐标点
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    # 拉成1维，位置相同的元素组成画布内的一个点的表示
    shifts_x = shifts_x.reshape(-1)
    shifts_y = shifts_y.reshape(-1)
    shifts = np.stack((shifts_x, shifts_y, shifts_x, shifts_y), axis = 1)
    # 每个像素点添加所有类型的锚框
    anchors = shifts.reshape((-1,1,4))+base_anchors.reshape((1,-1,4))
    return torch.tensor(anchors, dtype=torch.float32).view(1,-1,4)
Y = MultiBoxPrior(feature_map_size=(h,w))
boxes = Y.reshape((h,w,5,4))
def show_bboxes(axes, bboxes, labels=None, colors=None):
    #  获取锚框上的坐标
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    # 框的颜色
    colors = _make_list(colors, ['b','g','r','m','c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(
            bbox.detach().cpu().numpy(), color
        )
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color=='w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=6,
                      color=text_color,bbox=dict(facecolor=color, lw=0))

d2l.set_figsize()
fig = d2l.plt.imshow(img)
bbox_scale = torch.tensor([[w,h,w,h]],dtype=torch.float32)
show_bboxes(fig.axes, boxes[250,250,:,:]*bbox_scale,
            ['s=0.75, r=1', 's=0.75, r=2','s=0.55, r=0.5',
             's=0.5, r=1','s=0.25, r=1'])
d2l.plt.show()

# 交并比
# 计算交集
def compute_intersection(set_1, set_2):
    # 左上角取最大(n1,n2,2)
    lower_bounds = torch.max(set_1[:,:2].unsqueeze(1),
                             set_2[:,:2].unsqueeze(0))
    # 右下角取最小(n1,n2,2)
    upper_bounds = torch.min(set_1[:,2:].unsqueeze(1),
                             set_2[:,2:].unsqueeze(0))
    # (n1,n2,2)
    intersection_dims = torch.clamp(
        upper_bounds - lower_bounds, min = 0
    )
    intersection = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]
    return intersection
# 计算并集
def compute_jaccard(set_1, set_2):
    intersection = compute_intersection(set_1, set_2)
    areas_set_1 = (set_1[:,2]-set_1[:,0])*(set_1[:,3]-set_1[:,1])
    areas_set_2 = (set_2[:,2]-set_2[:,0])*(set_2[:,3]-set_2[:,1])
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection
    return union

# 交并比
def iouu(set1,set2):
    return compute_intersection(set1,set2)/compute_jaccard(set1,set2)
# 为与真实边框最相似的锚框分配label
def match_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    num_anchors = anchors.shape[0]
    num_gt_boxes = ground_truth.shape[0]
    # 对于第i个锚框和第j个真实边框，用矩阵第i行第j列的元素表示其交并比(iou)
    jaccard = compute_intersection(anchors,ground_truth)/compute_jaccard(anchors,ground_truth)
    # 初始化一个张量储存每一个锚框对应的真实边框
    anchors_bbox_map = torch.full(
        (num_anchors,), -1, dtype=torch.long, device=device
    )
    # 根据阈值为锚框分配真是边框
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious>=0.5).reshape(-1)
    box_j = indices[max_ious>=0.5]
    anchors_bbox_map[anc_i] = box_j
    # 为每一个锚框匹配最大的iou对应的真实边框
    anc_i = torch.argmax(jaccard, dim=0)
    box_j = torch.arange(num_gt_boxes, device=device)
    anchors_bbox_map[anc_i] = box_j
    return anchors_bbox_map

# 将(左上角坐标，右下角坐标)转换为(中心点坐标，宽，长)
def box_corner_to_center(boxes):
    x1,y1,x2,y2 = boxes[:,0], boxes[:,1],boxes[:,2],boxes[:,3]
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    w = x2-x1
    h = y2-y1
    boxes = np.stack((cx,cy,w,h), axis=1)
    return boxes

# 为锚框标注偏移量
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:,:2]-c_anc[:,:2]) / c_anc[:,2:]
    offset_wh = 5 * np.log(eps + c_assigned_bb[:,2:] / c_anc[:,2:])
    offset = np.concatenate([offset_xy, offset_wh],axis=1)
    return offset

bbox_scale = torch.tensor(
    (w,h,w,h), dtype=torch.float32
)
# 第一列表示类别，之后分别是左上角x,y,右下角x,y
ground_truth = torch.tensor(
    [
        [0,0.1,0.08,0.52,0.92],
        [1,0.55,0.2,0.9,0.88]
    ]
)
anchors = torch.tensor(
    [
        [0,0.1,0.2,0.3],
        [0.15,0.2,0.4,0.4],
        [0.63,0.05,0.88,0.98],
        [0.66,0.45,0.8,0.8],
        [0.57,0.3,0.92,0.9]
    ]
)
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:,1:]*bbox_scale,
            ['dog','cat'],'k')
show_bboxes(fig.axes,anchors*bbox_scale,
            ['0','1','2','3','4'])
d2l.plt.show()

# 为锚框标注类别和偏移量
def multibox_target(anchors, labels):
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [],[],[]
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        # 取当前边框的坐标
        label = labels[i,:,:]
        # 第一列为标签
        anchors_bbox_map = match_anchor_to_bbox(
            label[:,1:],anchors,device
        )
        # 只保留匹配上的，未匹配上的为-1被过滤掉了
        bbox_mask = (
            (anchors_bbox_map>=0).float().unsqueeze(-1)
        ).repeat(1,4)
        # 初始化锚框标签
        class_labels = torch.zeros(
            num_anchors, dtype=torch.long, device=device
        )
        # 初始化偏移量
        assigned_bb = torch.zeros(
            (num_anchors, 4), dtype=torch.float32,device=device
        )
        # 为锚框标记类别
        indices_true = torch.nonzero(anchors_bbox_map>=0)
        bb_idx = anchors_bbox_map[indices_true]
        # 加1使原来的背景变为标签0
        class_labels[indices_true] = label[bb_idx,0].long()+1
        assigned_bb[indices_true] = label[bb_idx,1:]
        # 计算偏移量
        offset = torch.from_numpy(
            offset_boxes(anchors, assigned_bb)
        )*bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset,bbox_mask,class_labels)

labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))
print(labels[2])



# 将(中心点坐标，宽，长)转换为(左上角坐标，右下角坐标)
def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = np.stack((x1,y1,x2,y2),axis=1)
    return boxes

# 根据锚框和偏移量反推预测的边框
def offset_inverse(anchors, offset_preds):
    c_anc = torch.from_numpy(box_corner_to_center(anchors))
    c_pred_bb_xy = (offset_preds[:,:2] * c_anc[:,2:] / 10) + c_anc[:,:2]
    c_pred_bb_wh = np.exp(offset_preds[:,2:] / 5) * c_anc[:,2:]
    c_pred_bb = np.concatenate([c_pred_bb_xy, c_pred_bb_wh], axis=1)
    predicted_bb = box_center_to_corner(c_pred_bb)
    return predicted_bb

# 利用非极大值抑制移除相似的预测边界框
def nms(boxes, scores, iou_threshold):
    boxes = torch.tensor(boxes)
    # 降序排列
    B = torch.argsort(scores, dim=-1, descending=True)
    # 保存保留框的索引
    keep = []
    while B.numel() > 0:
        # 最大值的索引
        i = B[0]
        keep.append(i)
        # 终止条件，1个框计算并交比
        if B.numel() == 1:
            break
        # 当前框与剩余框的并交比
        iou = iouu(boxes[i,:].reshape(-1,4),boxes[B[1:],:].reshape(-1,4)).reshape(-1)
        # 筛选满足条件的框,返回索引值
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        # 加1是因为当前狂已经计算过，所以整体索引加1才是剩余框的索引
        B = B[inds+1]
    return torch.tensor(keep,device=boxes.device)


anchors = torch.tensor(
    [
        [0.1,0.08,0.52,0.92],
        [0.08,0.2,0.56,0.95],
        [0.15,0.3,0.62,0.91],
        [0.55,0.2,0.9,0.88]
    ]
)
offset_preds = torch.tensor([0.0]*(4*len(anchors)))
cls_probs = torch.tensor(
    [
        [0.,0.,0.,0.],          # 背景的预测概率
        [0.9,0.8,0.7,0.1],      # 狗的预测概率
        [0.1,0.2,0.3,0.9]       # 猫的预测概率
    ]
)
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors*bbox_scale,
            ['dog=0.9','dog=0.8','dog=0.7','cat=0.9'])
d2l.plt.show()

# 执行非最大值抑制
def multibox_detection(cls_probs, offset_preds, anchors,
                       nms_threshold=0.5, score_threshold=0.0099):
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1,4)
        # 每一列最大值及其索引，过滤掉背景概率
        conf, class_id = torch.max(cls_prob[1:],0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, 0.5)
        # 将所有未保留的框设为背景类
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        # 不重复元素(类别)，不重复元素个数(类别数)
        uniques, counts = combined.unique(return_counts=True)
        # 剔除只出现1次的索引
        non_keep = uniques[counts==1]
        all_id_sorted = torch.cat((keep, non_keep))
        # 设为背景
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        predicted_bb = torch.tensor(predicted_bb, device=device)
        pred_info = torch.cat(
            (class_id.unsqueeze(1).float(),
             conf[all_id_sorted].unsqueeze(1),
             predicted_bb[all_id_sorted]),
            dim=1
        )
        out.append(pred_info)
    return torch.stack(out)

# 预测类别索引、置信度、框坐标
output = multibox_detection(
    cls_probs.unsqueeze(dim=0),
    offset_preds.unsqueeze(dim=0),
    anchors.unsqueeze(dim=0),
    nms_threshold=0.5
)
print(output)
# 移除类别为-1的边框
fig = d2l.plt.imshow(img)
for i in output[0].detach().cpu().numpy():
    if i[0] == -1:
        continue
    label= ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:])*bbox_scale], label)
d2l.plt.show()