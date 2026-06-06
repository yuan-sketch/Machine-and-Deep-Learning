# 袁超
# 开发时间：2026/5/29 13:37
import importlib
import os
import json
import numpy as np
import torch
import torchvision
from PIL import Image
import d2lzh8 as d2l
import sys
data_dir = 'data/pikachu'

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
def match_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    num_anchors = anchors.shape[0]
    num_gt_boxes = ground_truth.shape[0]
    ground_truth = ground_truth.to(device)
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
def box_corner_to_center(boxes):
    x1,y1,x2,y2 = boxes[:,0], boxes[:,1],boxes[:,2],boxes[:,3]
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    w = x2-x1
    h = y2-y1
    boxes = torch.stack((cx,cy,w,h), axis=1)
    return boxes
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:,:2]-c_anc[:,:2]) / c_anc[:,2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:,2:] / c_anc[:,2:])
    offset = torch.cat([offset_xy, offset_wh],axis=1)
    return offset

def multibox_target(anchors, labels):
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [],[],[]
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        # 取当前边框的坐标
        label = labels[i,:,:]
        label = label.to(device)
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
        offset = offset_boxes(anchors, assigned_bb)*bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset,bbox_mask,class_labels)
class PikachuDetDataset(torch.utils.data.Dataset):
    """
        皮卡丘目标检测数据集类
        参数:
            data_dir: 数据集根目录路径
            part: 数据集划分，只能是'train'(训练集)或'val'(验证集)
            image_size: 输出图片的尺寸，格式为(高度, 宽度)
    """
    def __init__(self, data_dir, part, image_size=(256,256)):
        assert part in ['train', 'val']
        # 图片尺寸
        self.image_size = image_size
        # 图片路径
        self.image_dir = os.path.join(data_dir, part, 'images')
        # 图片标签
        # 读取标签文件：data_dir/part/label.json
        # 标签文件格式：{"1.png": {"class": 0, "loc": [x1, y1, x2, y2]}, ...}
        # 其中loc是归一化后的边界框坐标(0~1之间)
        with open(os.path.join(data_dir, part, 'label.json')) as f:
            self.label = json.load(f)
        self.transform = torchvision.transforms.Compose(
            [
                # ToTensor的作用：
                # 1. 将PIL图片(范围0-255, 形状H×W×C)转换为FloatTensor
                # 2. 像素值归一化到[0.0, 1.0]区间
                # 3. 通道顺序调整为C×H×W(PyTorch默认格式)
                # 将PIL图片转换成围殴于[0.0，1.0]的floattensor,shape(C*H*W)
                torchvision.transforms.ToTensor()
            ]
        )
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        # 构建图片文件名：数据集按1.png, 2.png...顺序命名
        image_path = str(index+1)+'.png'
        # 获取该图片的类别标签(皮卡丘类别为0)
        cls = self.label[image_path]['class']

        # 构建标签数组：[类别, x1, y1, x2, y2]
        # [None, :]将形状从(5,)变为(1, 5)，为后续批量处理做准备
        # dtype设为float32是为了和PyTorch模型输入类型匹配
        label = np.array(
            [cls]+self.label[image_path]['loc'],
            dtype='float32'
        )[None,:]
        # 读取图片并预处理
        # 1. 打开图片文件
        # 2. 转换为RGB格式(确保即使是灰度图也有3个通道)
        # 3. 调整到指定尺寸
        PIL_img = Image.open(
            os.path.join(self.image_dir, image_path)
        ).convert('RGB').resize(self.image_size)
        # 应用定义好的转换，将PIL图片转为PyTorch Tensor
        img = self.transform(PIL_img)
        sample = {
            'label':label,
            'image':img
        }
        return sample
# 随机读取训练集，按序读取测试集
def load_data_pikachu(batch_size, edge_size=256,
                          data_dir='data/pikachu'):
    """
        加载皮卡丘目标检测数据集
        参数:
            batch_size: 每个批次的样本数量
            edge_size: 图片的边长(正方形)
            data_dir: 数据集根目录
        返回:
            train_iter: 训练集数据迭代器
            val_iter: 验证集数据迭代器
    """
    image_size = (edge_size, edge_size)
    train_dataset = PikachuDetDataset(
            data_dir, 'train', image_size
        )
    val_dataset = PikachuDetDataset(
            data_dir, 'val', image_size
        )
    # 创建训练集DataLoader
    # shuffle=True: 训练时随机打乱样本顺序，防止模型学习到顺序相关的特征
    # num_workers=4: 使用4个子进程加载数据，提高加载速度
    train_iter = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4
        )
    # 创建验证集DataLoader
    # shuffle=False: 验证时不需要打乱样本，按顺序评估即可
    val_iter = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4
        )
    return train_iter, val_iter


# 类别预测层
def cls_predictor(num_inputs, num_anchors, num_classes):
    return torch.nn.Conv2d(
        num_inputs, num_anchors*(num_classes+1),
        kernel_size=3, padding=1
    )

# 边界框预测层——为每个锚框预测四个偏移量
def bbox_predictor(num_inputs, num_anchors):
    return torch.nn.Conv2d(
        num_inputs, num_anchors*4,
        kernel_size=3, padding=1
    )


def flatten_pred(pred):
    return torch.flatten(pred.permute(0,2,3,1),start_dim=1)
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

# 高宽减半块，串联两个填充为1得3*3卷积层和步幅为2得2*2最大池化层
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(torch.nn.BatchNorm2d(out_channels))     # 归一化层
        blk.append(torch.nn.ReLU())
        in_channels = out_channels
    blk.append(torch.nn.MaxPool2d(2))
    return torch.nn.Sequential(*blk)

# 基础网络块
def base_net():
    blk = []
    num_filters = [3,16,32,64]
    for i in range(len(num_filters)-1):
        blk.append(
            down_sample_blk(num_filters[i], num_filters[i+1])
        )
    return torch.nn.Sequential(*blk)

def get_blk(i):
    if i==0:
        blk = base_net()
    elif i==1:
        blk = down_sample_blk(64,128)
    elif i==4:
        blk = torch.nn.AdaptiveMaxPool2d((1,1))           #全局最大池化
    else:
        blk = down_sample_blk(128,128)
    return blk

# 返回特征图Y，锚框，锚框类别，偏移量
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.MultiBoxPrior(Y, size, ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

# 锚框归一化尺寸
sizes = [
    [0.2,0.272],[0.37,0.447],[0.54,0.619],[0.71,0.79],[0.88,0.961]
]
# 每个特征图对应的锚框宽高比（5个特征图，每个特征图3个宽高比）
ratios = [[1,2,0.5]] * 5
num_anchors = len(sizes[0])+len(ratios[0])-1


# 定义完整模型
class TinySSD(torch.nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD,self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64,128,128,128,128]
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(
                idx_to_in_channels[i], num_anchors, num_classes
            ))
            setattr(self, f'bbox_{i}', bbox_predictor(
                idx_to_in_channels[i], num_anchors
            ))
    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None]*5,[None]*5,[None]*5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}')
            )
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes+1
        )
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
if __name__ == '__main__':
    # Y1，Y2锚框数分别为5、3，类别个数是10时，输出通道数分别为55，33
    def forward(x, block):
        return block(x)


    Y1 = forward(
        torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10)
    )
    Y2 = forward(
        torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10)
    )

    net = TinySSD(num_classes=1)
    X= torch.zeros((32,3,256,256))
    anchors, cls_preds, bbox_preds = net(X)
    print('输出锚框：',anchors.shape)
    print('输出锚框类别：',cls_preds.shape)
    print('偏移量：',bbox_preds.shape)



    # 读取数据集
    batch_size = 32
    train_iter, _ = load_data_pikachu(batch_size)
    device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
    )
    net = TinySSD(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

    # 定义损失函数和评价函数
    # 锚框类别——交叉熵损失，正类锚框偏移量——L1范数损失即预测值与真实值之间差的绝对值
    cls_loss = torch.nn.CrossEntropyLoss(reduction='none')
    bbox_loss = torch.nn.L1Loss(reduction='none')
    def calc_loss(cls_preds, cls_labels, bbox_preds,
              bbox_labels, bbox_masks):
        batch_size, num_classes  = cls_preds.shape[0], cls_preds.shape[2]
        cls = cls_loss(cls_preds.reshape(-1,num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size,-1).mean(dim=1)
        bbox = bbox_loss(bbox_preds*bbox_masks,
                     bbox_labels*bbox_masks).mean(dim=1)
        return cls+bbox

    def cls_eval(cls_preds, cls_labels):
        return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype)==cls_labels).sum())

    def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
        return float((torch.abs((bbox_labels-bbox_preds)*bbox_masks)).sum())


    # 训练模型
    import time
    num_epochs = 30
    net = net.to(device)
    for epoch in range(num_epochs):
        accuracy_sum = 0
        mae_sum = 0
        num_examples = 0
        num_labels = 0
        net.train()
        for data in train_iter:
            features = data['image']
            target = data['label']
            begin = time.time()
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            bbox_labels = bbox_labels.to(device)
            bbox_masks = bbox_masks.to(device)
            cls_labels = cls_labels.to(device)
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            trainer.step()
            accuracy_sum += cls_eval(cls_preds, cls_labels)
            mae_sum += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            num_examples += bbox_labels.numel()
            num_labels += cls_labels.numel()
        cls_err, bbox_mae = 1-accuracy_sum/num_labels, mae_sum/num_examples
        if (epoch+1) % 5 == 0:
            print(f'epoch {epoch+1}, class err {cls_err:.2e},\
                    bbox mae {bbox_mae :.2e}, time {time.time()-begin:.1f} sec')

    # 预测目标——转成卷积层需要的四维格式
    from copy import deepcopy
    img = Image.open('data/pikachu.jpg')
    img = np.array(img, dtype=np.float32) / 255.0
    X = torch.from_numpy(img)
    show_img = deepcopy(X*255.0)
    X= X.permute(2,0,1).unsqueeze(0).float()
    img = X.squeeze(0).permute(1,2,0).long()
    show_img = show_img.permute(2,0,1).unsqueeze(0).float()
    show_img = show_img.squeeze(0).permute(1,2,0).long()
    def box_center_to_corner(boxes):
        cx, cy, w, h = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        boxes = np.stack((x1,y1,x2,y2),axis=1)
        return boxes
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
    # 预测边界框
    import torch.nn.functional as F
    def predict(X):
        net.eval()
        anchors, cls_preds, bbox_preds = net(X.to(device))
        cls_probs = F.softmax(cls_preds, dim=2).permute(0,2,1)
        output = multibox_detection(cls_probs.detach().cpu(),
                                    bbox_preds.detach().cpu(),
                                    anchors.detach().cpu())
        idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
        return output[0, idx]

    output = predict(X)
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
    def display(img, output, threshold):
        d2l.set_figsize((5,5))
        fig = d2l.plt.imshow(img)
        for row in output:
            score = float(row[1])
            if score < threshold:
                continue
            h, w = img.shape[:2]
            bbox = [row[2:6] * torch.tensor(
                (w,h,w,h), device=row.device
            ).float()]
            show_bboxes(fig.axes, bbox, '%.2f'% score)

    display(show_img, output.cpu(), threshold=0.46)
    d2l.plt.show()
