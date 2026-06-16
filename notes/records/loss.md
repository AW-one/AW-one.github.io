# Loss Functions: 常见损失函数与直觉解释

> **定位**：记录机器学习 / 深度学习中最常见的损失函数，重点放在“什么时候用、公式是什么、代码怎么写、直觉是什么”。
>
> **分类标签**：`Loss` `Optimization` `Regularization` `Classification` `Regression` `Style Transfer`

---

## 一句话总结

损失函数是训练模型时的“扣分规则”。模型输出越接近我们希望的结果，loss 越小；优化器做的事情就是不断调整参数，让这个扣分尽可能低。

---

## 一、回归类损失

### 1. MSE / L2 Loss

**适用场景**：回归任务、噪声预测、图像重建、扩散模型中的噪声回归。

**应用例子**：在 DDPM 中，模型输入 noisy image 和 timestep，输出预测噪声；训练目标常写成预测噪声和真实噪声之间的 MSE。

完整公式：

$$
\mathcal{L}_{\text{MSE}}
= \frac{1}{N}
\sum_{i=1}^{N}
\left(
y_i - \hat{y}_i
\right)^2
$$

其中 $y_i$ 是真实值，$\hat{y}_i$ 是预测值，$N$ 是样本数量。

直觉解释：

- 误差越大，惩罚增长越快；
- 对异常值比较敏感；
- 优化时梯度连续、稳定，所以很常用。

Python 代码：

```python
import numpy as np

def mse_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.mean((y_true - y_pred) ** 2)
```

出处：

- Gauss, 1809. 最小二乘思想的早期系统化来源之一。
- Ho et al., 2020. DDPM 中常用噪声预测 MSE 形式。

### 2. MAE / L1 Loss

**适用场景**：回归任务、图像重建、希望降低异常值影响的场景。

**应用例子**：在人脸重建、图像修复或深度估计中，L1 常用于减少过度平滑，让结果比纯 MSE 更锐利一些。

完整公式：

$$
\mathcal{L}_{\text{MAE}}
= \frac{1}{N}
\sum_{i=1}^{N}
\left|
y_i - \hat{y}_i
\right|
$$

直觉解释：

- 每个误差按绝对值惩罚；
- 比 MSE 更不怕少量极端异常值；
- 在误差为 0 附近不可导，但实际框架里通常可以处理。

Python 代码：

```python
import numpy as np

def mae_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.mean(np.abs(y_true - y_pred))
```

出处：

- Laplace, 1818. L1 误差与 Laplace 分布假设关系密切。

### 3. Huber Loss / Smooth L1 Loss

**适用场景**：鲁棒回归、目标检测 bbox regression、传感器数据回归、存在异常值但仍希望保留 MSE 平滑性的任务。

**应用例子**：Fast R-CNN / Faster R-CNN 系列中常使用 Smooth L1 做边界框回归，因为它比 MSE 更不容易被少数错误框支配，又比纯 L1 更平滑。

完整公式：

$$
\mathcal{L}_{\delta}(a)
=
\begin{cases}
\frac{1}{2}a^2, & |a| \le \delta \\
\delta\left(|a| - \frac{1}{2}\delta\right), & |a| > \delta
\end{cases}
$$

其中 $a = y - \hat{y}$ 是预测误差，$\delta$ 控制从二次惩罚切换到线性惩罚的位置。

直觉解释：

- 小误差时像 MSE，优化更平滑；
- 大误差时像 MAE，不会让异常值产生过大影响；
- 适合“多数样本比较干净，但少数样本可能非常离谱”的场景。

Python 代码：

```python
import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    error = np.asarray(y_true) - np.asarray(y_pred)
    abs_error = np.abs(error)
    quadratic = 0.5 * error ** 2
    linear = delta * (abs_error - 0.5 * delta)
    return np.mean(np.where(abs_error <= delta, quadratic, linear))
```

出处：

- Huber, 1964. Robust Estimation of a Location Parameter。
- Girshick, 2015. Fast R-CNN 中使用 Smooth L1 做边界框回归。

---

## 二、分类类损失

### 1. Softmax Cross Entropy

**适用场景**：多分类任务、语言模型 next-token prediction、图像分类。

**应用例子**：LLM 训练时，模型对词表中每个 token 输出一个 logit，softmax 后和真实下一个 token 做交叉熵。

先把 logits $z$ 转成概率：

$$
p_i
= \text{softmax}(z)_i
=
\frac{\exp(z_i)}
{\sum_{j=1}^{C}\exp(z_j)}
$$

交叉熵损失为：

$$
\mathcal{L}_{\text{CE}}
=
- \sum_{i=1}^{C}
y_i \log p_i
$$

如果标签是 one-hot，真实类别为 $k$，公式可以简化成：

$$
\mathcal{L}_{\text{CE}}
=
- \log p_k
$$

直觉解释：

- 如果真实类别概率 $p_k$ 很高，loss 很小；
- 如果模型把真实类别概率压得很低，loss 会迅速变大；
- 语言模型训练本质上就是对每个位置做一次“下一个 token 分类”。

Python 代码：

```python
import numpy as np

def softmax_cross_entropy(logits, target_index):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    return -np.log(probs[target_index] + 1e-12)
```

出处：

- Shannon, 1948. 信息熵与交叉熵的理论基础。
- Goodfellow et al., 2016. Deep Learning，第 6 章常见监督学习损失。

### 2. Binary Cross Entropy

**适用场景**：二分类、多标签分类。

**应用例子**：图像标签可以同时包含 `person`、`dog`、`outdoor`，每个标签都是独立的二分类判断，因此常用 BCE。

完整公式：

$$
\mathcal{L}_{\text{BCE}}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i \log \hat{y}_i
+
(1-y_i)\log(1-\hat{y}_i)
\right]
$$

其中 $y_i \in \{0,1\}$，$\hat{y}_i$ 是预测为正类的概率。

Python 代码：

```python
import numpy as np

def binary_cross_entropy(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1 - eps)
    loss = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return np.mean(loss)
```

出处：

- Bishop, 2006. Pattern Recognition and Machine Learning，逻辑回归与交叉熵。

### 3. Focal Loss

**适用场景**：类别极度不均衡的分类任务，尤其是一阶段目标检测、密集检测、前景背景比例悬殊的任务。

**应用例子**：RetinaNet 使用 Focal Loss 解决 dense object detection 中大量 easy negative 压过少量 foreground 的问题。

二分类形式：

$$
\mathcal{L}_{\text{focal}}
=
-
\alpha_t
(1-p_t)^\gamma
\log(p_t)
$$

其中：

$$
p_t =
\begin{cases}
p, & y = 1 \\
1-p, & y = 0
\end{cases}
$$

直觉解释：

- 当样本已经被分对时，$p_t$ 很大，$(1-p_t)^\gamma$ 很小，loss 被降低；
- 当样本很难分时，$p_t$ 小，loss 保持较大；
- 它让训练更关注 hard examples。

Python 代码：

```python
import numpy as np

def focal_loss_binary(y_true, y_prob, alpha=0.25, gamma=2.0):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-12, 1 - 1e-12)
    p_t = np.where(y_true == 1, y_prob, 1 - y_prob)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    return np.mean(-alpha_t * (1 - p_t) ** gamma * np.log(p_t))
```

出处：

- Lin et al., 2017. Focal Loss for Dense Object Detection。
- facebookresearch/Detectron 和后续 Detectron2 / MMDetection 中都有相关实现。

---

## 三、正则相关损失

### 1. L2 Regularization / Weight Decay

**适用场景**：控制参数规模，降低过拟合风险。

**应用例子**：Transformer / CNN 训练中常配合 AdamW 使用 weight decay，尤其是在中小数据集微调时可以提升泛化。

完整公式：

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{task}}
+
\lambda
\left\|
\theta
\right\|_2^2
=
\mathcal{L}_{\text{task}}
+
\lambda
\sum_{j}
\theta_j^2
$$

直觉解释：

- 参数越大，额外惩罚越大；
- 倾向于让模型使用更平滑、更小幅度的权重；
- 在深度学习中常以 weight decay 的形式出现。

Python 代码：

```python
import numpy as np

def l2_regularized_loss(task_loss, params, lambda_):
    penalty = sum(np.sum(np.asarray(w) ** 2) for w in params)
    return task_loss + lambda_ * penalty
```

出处：

- Hoerl and Kennard, 1970. Ridge Regression。
- Loshchilov and Hutter, 2019. AdamW 将 weight decay 与 Adam 更新解耦。

### 2. L1 Regularization

**适用场景**：希望参数更稀疏，例如特征选择、稀疏线性模型。

**应用例子**：传统机器学习里做特征选择时，L1 可以让不重要的特征权重变成 0，从而得到更容易解释的模型。

完整公式：

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{task}}
+
\lambda
\left\|
\theta
\right\|_1
=
\mathcal{L}_{\text{task}}
+
\lambda
\sum_j
\left|
\theta_j
\right|
$$

直觉解释：

- L1 会鼓励一部分参数变成 0；
- 适合希望模型“少用一些特征”的场景；
- 在大模型预训练里不是最常见主力正则，但在传统机器学习中很重要。

Python 代码：

```python
import numpy as np

def l1_regularized_loss(task_loss, params, lambda_):
    penalty = sum(np.sum(np.abs(np.asarray(w))) for w in params)
    return task_loss + lambda_ * penalty
```

出处：

- Tibshirani, 1996. LASSO。

---

## 四、分布相关损失

### KL Divergence

**适用场景**：概率分布匹配、VAE、知识蒸馏、RLHF / policy optimization 中的分布约束。

**应用例子**：在知识蒸馏中，学生模型不仅学习 hard label，还会学习老师模型输出的 soft distribution。

完整公式：

$$
D_{\text{KL}}(P \| Q)
=
\sum_{i}
P(i)
\log
\frac{P(i)}{Q(i)}
$$

连续形式为：

$$
D_{\text{KL}}(P \| Q)
=
\int
p(x)
\log
\frac{p(x)}{q(x)}
dx
$$

直觉解释：

- KL 衡量“如果真实分布是 $P$，却用 $Q$ 来近似，会多付出多少信息代价”；
- 它不是严格距离，因为通常 $D_{\text{KL}}(P\|Q) \neq D_{\text{KL}}(Q\|P)$；
- 在知识蒸馏中，可以让学生模型的输出分布接近老师模型。

Python 代码：

```python
import numpy as np

def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))
```

出处：

- Kullback and Leibler, 1951. On Information and Sufficiency。
- Kingma and Welling, 2014. Auto-Encoding Variational Bayes。

---

## 五、感知与风格相关损失

### 1. Perceptual Loss

**适用场景**：图像生成、超分辨率、风格迁移、图像编辑。

**应用例子**：超分辨率中，如果只用 MSE，图片可能数值接近但视觉偏糊；加入 perceptual loss 后，高层语义和纹理会更自然。

完整公式：

$$
\mathcal{L}_{\text{perceptual}}
=
\sum_{l}
\frac{1}{N_l}
\left\|
\phi_l(x)
-
\phi_l(\hat{x})
\right\|_2^2
$$

其中 $\phi_l(\cdot)$ 表示预训练视觉网络在第 $l$ 层提取的特征，$N_l$ 是该层特征元素数量。

直觉解释：

- 不直接比较像素，而是比较高层视觉特征；
- 两张图片像素不完全一致，但语义和结构相似时，perceptual loss 可以更合理；
- 常用 VGG 特征作为 $\phi$。

Python 伪代码：

```python
def perceptual_loss(vgg_features, image, target):
    image_feats = vgg_features(image)
    target_feats = vgg_features(target)
    loss = 0.0
    for a, b in zip(image_feats, target_feats):
        loss += ((a - b) ** 2).mean()
    return loss
```

出处：

- Johnson et al., 2016. Perceptual Losses for Real-Time Style Transfer and Super-Resolution。

### 2. Style Loss / Gram Matrix Loss

**适用场景**：神经风格迁移、纹理约束、服装纹理 / 材质一致性。

**应用例子**：服装生成或材质迁移中，可以用 style loss 约束生成结果保持参考图的纹理统计，比如针织、皮革、印花等。

先定义第 $l$ 层特征：

$$
F_l(x) \in \mathbb{R}^{C_l \times H_l W_l}
$$

Gram Matrix：

$$
G_l(x)
=
F_l(x)F_l(x)^{T}
$$

风格损失：

$$
\mathcal{L}_{\text{style}}
=
\sum_l
\left\|
G_l(x)
-
G_l(x_{\text{style}})
\right\|_F^2
$$

直觉解释：

- Gram Matrix 记录不同通道特征之间的相关性；
- 它不太关心物体在哪里，更关心纹理、颜色、笔触和整体风格；
- 所以风格迁移中常把 content loss 和 style loss 一起用。

Python 代码：

```python
import numpy as np

def gram_matrix(features):
    # features shape: [channels, height, width]
    c, h, w = features.shape
    flattened = features.reshape(c, h * w)
    return flattened @ flattened.T / (c * h * w)

def style_loss(features, style_features):
    g = gram_matrix(features)
    g_style = gram_matrix(style_features)
    return np.mean((g - g_style) ** 2)
```

出处：

- Gatys et al., 2016. Image Style Transfer Using Convolutional Neural Networks。

---

## 六、检测与分割相关损失

### 1. Dice Loss

**适用场景**：医学图像分割、前景区域很小的二分类分割、类别不均衡的语义分割。

**应用例子**：病灶、肿瘤、器官边界等目标通常只占图像很小一部分，Dice Loss 可以直接优化预测区域和真实区域的重叠程度。

Dice coefficient：

$$
\text{Dice}(P, G)
=
\frac{2|P \cap G|}
{|P| + |G|}
$$

常用 soft Dice loss：

$$
\mathcal{L}_{\text{Dice}}
=
1 -
\frac{
2\sum_i p_i g_i + \epsilon
}{
\sum_i p_i + \sum_i g_i + \epsilon
}
$$

其中 $p_i$ 是预测概率，$g_i$ 是真实 mask。

Python 代码：

```python
import numpy as np

def dice_loss(pred, target, eps=1e-6):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    intersection = np.sum(pred * target)
    dice = (2 * intersection + eps) / (np.sum(pred) + np.sum(target) + eps)
    return 1 - dice
```

出处：

- Milletari et al., 2016. V-Net 中使用基于 Dice coefficient 的目标函数。

### 2. Tversky Loss

**适用场景**：希望显式控制 false positive 和 false negative 权重的分割任务，尤其是医学病灶分割。

**应用例子**：在医学场景中，漏检病灶通常比多标一点背景更严重，因此可以调大 false negative 的惩罚。

Tversky index：

$$
\text{TI}
=
\frac{
TP
}{
TP + \alpha FP + \beta FN
}
$$

Tversky loss：

$$
\mathcal{L}_{\text{Tversky}}
=
1 - \text{TI}
$$

Python 代码：

```python
import numpy as np

def tversky_loss(pred, target, alpha=0.3, beta=0.7, eps=1e-6):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    tp = np.sum(pred * target)
    fp = np.sum(pred * (1 - target))
    fn = np.sum((1 - pred) * target)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - tversky
```

出处：

- Salehi et al., 2017. Tversky loss function for image segmentation using 3D fully convolutional deep networks。
- Abraham and Khan, 2018. Focal Tversky loss 用于 lesion segmentation。

### 3. IoU / GIoU Loss

**适用场景**：目标检测中的边界框回归。

**应用例子**：bbox 回归最终评测通常看 IoU，如果只优化坐标 L1/MSE，训练目标和评测指标不完全一致；GIoU 进一步解决无重叠框时 IoU 梯度为 0 的问题。

IoU：

$$
\text{IoU}
=
\frac{
|A \cap B|
}{
|A \cup B|
}
$$

GIoU：

$$
\text{GIoU}
=
\text{IoU}
-
\frac{
|C \setminus (A \cup B)|
}{
|C|
}
$$

其中 $C$ 是同时包住预测框 $A$ 和真实框 $B$ 的最小闭包框。

Loss：

$$
\mathcal{L}_{\text{GIoU}}
=
1 - \text{GIoU}
$$

Python 代码：

```python
def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def iou_loss(box_a, box_b, eps=1e-6):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = box_area([ix1, iy1, ix2, iy2])
    union = box_area(box_a) + box_area(box_b) - inter
    return 1 - inter / (union + eps)
```

出处：

- Rezatofighi et al., 2019. Generalized Intersection over Union。

### 4. Lovasz-Softmax Loss

**适用场景**：语义分割中直接优化 mIoU / Jaccard index 的 surrogate loss。

**应用例子**：Cityscapes、PASCAL VOC 等语义分割任务常用 mIoU 作为核心指标，Lovasz-Softmax 试图让训练目标更贴近评测目标。

对于类别 $c$，Jaccard loss 可以写作：

$$
\Delta_{J_c}(y^*, y)
=
1 -
\frac{
| \{y^*=c\} \cap \{y=c\} |
}{
| \{y^*=c\} \cup \{y=c\} |
}
$$

Lovasz-Softmax 使用 Lovasz extension 构造一个可优化 surrogate：

$$
\mathcal{L}_{\text{Lovasz-Softmax}}
=
\frac{1}{|C|}
\sum_{c \in C}
\overline{\Delta}_{J_c}(m(c))
$$

这里 $m(c)$ 是类别 $c$ 的 per-pixel margin error，$\overline{\Delta}_{J_c}$ 表示 Jaccard loss 的 Lovasz extension。

Python 伪代码：

```python
def lovasz_softmax_placeholder(probs, labels):
    # 实际实现需要对每个类别的像素误差排序，
    # 再按 Jaccard 梯度加权求和；建议直接使用成熟开源实现。
    raise NotImplementedError("Use a tested Lovasz-Softmax implementation.")
```

出处：

- Berman et al., 2017. The Lovasz-Softmax loss。

---

## 七、度量学习与对比学习损失

### 1. Contrastive Loss

**适用场景**：Siamese network、相似度学习、人脸验证、签名验证、检索 embedding 学习。

**应用例子**：输入两张图片，模型学习把同一人的图片 embedding 拉近，把不同人的图片 embedding 推远。

完整公式：

$$
\mathcal{L}_{\text{contrastive}}
=
yD^2
+
(1-y)
\max(0, m-D)^2
$$

其中 $D = \|f(x_1)-f(x_2)\|_2$，$y=1$ 表示正样本对，$y=0$ 表示负样本对，$m$ 是 margin。

Python 代码：

```python
import numpy as np

def contrastive_loss(z1, z2, y, margin=1.0):
    distance = np.linalg.norm(np.asarray(z1) - np.asarray(z2))
    positive = y * distance ** 2
    negative = (1 - y) * max(0.0, margin - distance) ** 2
    return positive + negative
```

出处：

- Hadsell, Chopra, LeCun, 2006. Dimensionality Reduction by Learning an Invariant Mapping。

### 2. Triplet Loss

**适用场景**：人脸识别、行人重识别、图像检索、句向量相似度学习。

**应用例子**：FaceNet 使用 anchor / positive / negative 三元组，让同一身份更近，不同身份至少远出一个 margin。

完整公式：

$$
\mathcal{L}_{\text{triplet}}
=
\max
\left(
0,
\|f(a)-f(p)\|_2^2
-
\|f(a)-f(n)\|_2^2
+
\alpha
\right)
$$

其中 $a$ 是 anchor，$p$ 是 positive，$n$ 是 negative，$\alpha$ 是 margin。

Python 代码：

```python
import numpy as np

def triplet_loss(anchor, positive, negative, margin=0.2):
    d_pos = np.sum((np.asarray(anchor) - np.asarray(positive)) ** 2)
    d_neg = np.sum((np.asarray(anchor) - np.asarray(negative)) ** 2)
    return max(0.0, d_pos - d_neg + margin)
```

出处：

- Schroff et al., 2015. FaceNet: A Unified Embedding for Face Recognition and Clustering。

### 3. InfoNCE / NT-Xent Loss

**适用场景**：自监督学习、对比学习、SimCLR、CPC、跨模态对齐。

**应用例子**：给定一个 query、一个 positive 和多个 negative，InfoNCE 让 positive 的相似度在所有候选里最高。

完整公式：

$$
\mathcal{L}_{\text{InfoNCE}}
=
-
\log
\frac{
\exp(\text{sim}(q, k^+)/\tau)
}{
\exp(\text{sim}(q, k^+)/\tau)
+
\sum_{j=1}^{K}
\exp(\text{sim}(q, k_j^-)/\tau)
}
$$

其中 $\tau$ 是 temperature。

Python 代码：

```python
import numpy as np

def info_nce_loss(query, positive, negatives, temperature=0.07):
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

    pos_logit = cosine(query, positive) / temperature
    neg_logits = np.array([cosine(query, n) / temperature for n in negatives])
    logits = np.concatenate([[pos_logit], neg_logits])
    logits = logits - np.max(logits)
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return -np.log(probs[0] + 1e-12)
```

出处：

- Oord, Li, Vinyals, 2018. Representation Learning with Contrastive Predictive Coding。
- Chen et al., 2020. SimCLR 使用 NT-Xent 形式的 contrastive loss。

### 4. CLIP Contrastive Loss

**适用场景**：图文对齐、多模态检索、零样本分类、多模态表示学习。

**应用例子**：CLIP 在一个 batch 内把第 $i$ 张图和第 $i$ 段文本作为正样本，其余图文组合作为负样本。

设图像 embedding 为 $v_i$，文本 embedding 为 $t_i$，相似度矩阵：

$$
s_{ij}
=
\frac{
v_i^T t_j
}{\tau}
$$

image-to-text loss：

$$
\mathcal{L}_{i2t}
=
-
\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{
\exp(s_{ii})
}{
\sum_{j=1}^{N}
\exp(s_{ij})
}
$$

text-to-image loss：

$$
\mathcal{L}_{t2i}
=
-
\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{
\exp(s_{ii})
}{
\sum_{j=1}^{N}
\exp(s_{ji})
}
$$

总损失：

$$
\mathcal{L}_{\text{CLIP}}
=
\frac{1}{2}
\left(
\mathcal{L}_{i2t}
+
\mathcal{L}_{t2i}
\right)
$$

Python 伪代码：

```python
import numpy as np

def clip_loss(image_embeds, text_embeds, temperature=0.07):
    image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=1, keepdims=True)
    text_embeds = text_embeds / np.linalg.norm(text_embeds, axis=1, keepdims=True)
    logits = image_embeds @ text_embeds.T / temperature
    logits = logits - np.max(logits, axis=1, keepdims=True)

    exp_logits = np.exp(logits)
    i2t = -np.mean(np.log(np.diag(exp_logits) / np.sum(exp_logits, axis=1)))

    exp_logits_t = np.exp(logits.T - np.max(logits.T, axis=1, keepdims=True))
    t2i = -np.mean(np.log(np.diag(exp_logits_t) / np.sum(exp_logits_t, axis=1)))
    return 0.5 * (i2t + t2i)
```

出处：

- Radford et al., 2021. Learning Transferable Visual Models From Natural Language Supervision。
- OpenAI CLIP GitHub 仓库提供了开源实现。

---

## 八、生成模型相关损失

### GAN Adversarial Loss

**适用场景**：生成对抗网络、图像生成、图像翻译、超分辨率、风格迁移。

**应用例子**：GAN 中 generator 试图生成让 discriminator 判断为真的样本；discriminator 试图区分真实样本和生成样本。

原始 minimax objective：

$$
\min_G \max_D
V(D, G)
=
\mathbb{E}_{x \sim p_{\text{data}}}
\left[
\log D(x)
\right]
+
\mathbb{E}_{z \sim p_z}
\left[
\log(1-D(G(z)))
\right]
$$

实际训练 generator 时常用 non-saturating loss：

$$
\mathcal{L}_{G}
=
-
\mathbb{E}_{z \sim p_z}
\left[
\log D(G(z))
\right]
$$

Python 伪代码：

```python
import numpy as np

def gan_discriminator_loss(d_real, d_fake):
    eps = 1e-12
    d_real = np.clip(d_real, eps, 1 - eps)
    d_fake = np.clip(d_fake, eps, 1 - eps)
    return -np.mean(np.log(d_real) + np.log(1 - d_fake))

def gan_generator_loss(d_fake):
    eps = 1e-12
    d_fake = np.clip(d_fake, eps, 1 - eps)
    return -np.mean(np.log(d_fake))
```

出处：

- Goodfellow et al., 2014. Generative Adversarial Nets。

---

## 九、快速选择表

| 目标 | 常用损失 | 适用场景 | 直觉 |
| --- | --- | --- | --- |
| 连续值回归 | MSE / MAE | 回归、重建、噪声预测 | 预测值接近真实值 |
| 鲁棒回归 | Huber / Smooth L1 | bbox 回归、异常值较多的回归 | 小误差像 MSE，大误差像 L1 |
| 单标签分类 | Cross Entropy | 图像分类、文本分类、next-token prediction | 真实类别概率越高越好 |
| 多标签分类 | Binary Cross Entropy | 多属性识别、多标签检索 | 每个标签分别判断是否存在 |
| 类别极不均衡分类 | Focal Loss | 一阶段目标检测、密集检测 | 降低 easy examples 权重 |
| 控制参数规模 | L2 / Weight Decay | 小数据微调、常规深度网络训练 | 避免权重过大 |
| 稀疏参数 | L1 | 特征选择、稀疏模型 | 鼓励部分参数为 0 |
| 匹配分布 | KL Divergence | VAE、蒸馏、RLHF | 让两个概率分布接近 |
| 图像语义相似 | Perceptual Loss | 超分、编辑、重建 | 比较高层视觉特征 |
| 风格 / 纹理一致 | Style Loss | 风格迁移、材质迁移 | 比较特征通道相关性 |
| 分割重叠优化 | Dice / Tversky / Lovasz | 医学分割、语义分割 | 直接优化区域重叠 |
| 检测框回归 | IoU / GIoU | 目标检测 | 训练目标更贴近 IoU 指标 |
| 相似度学习 | Contrastive / Triplet | 人脸识别、检索、ReID | 同类拉近，异类推远 |
| 自监督 / 多模态对齐 | InfoNCE / CLIP Loss | SimCLR、CLIP、图文检索 | 正样本相似度高于负样本 |
| 对抗生成 | GAN Loss | 图像生成、图像翻译、超分 | 生成器和判别器博弈 |

---

## 参考

- Shannon, 1948. [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)：信息熵、交叉熵等概念的基础。
- Bishop, 2006. [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)：概率建模、交叉熵、正则化等机器学习基础。
- Goodfellow, Bengio, Courville, 2016. [Deep Learning](https://www.deeplearningbook.org/)：深度学习损失函数、优化和正则化。
- Kullback and Leibler, 1951. [On Information and Sufficiency](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-1/On-Information-and-Sufficiency/10.1214/aoms/1177729694.full)：KL Divergence 的经典来源。
- Gatys et al., 2016. [Image Style Transfer Using Convolutional Neural Networks](https://arxiv.org/abs/1508.06576)：Gram Matrix 风格损失。
- Johnson et al., 2016. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)：Perceptual Loss 在图像生成中的常用形式。
- Huber, 1964. [Robust Estimation of a Location Parameter](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full)：Huber Loss 的统计学来源。
- Lin et al., 2017. [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)：Focal Loss 和 RetinaNet。
- Milletari et al., 2016. [V-Net](https://arxiv.org/abs/1606.04797)：Dice loss 在 3D 医学图像分割中的使用。
- Salehi et al., 2017. [Tversky loss function for image segmentation](https://arxiv.org/abs/1706.05721)：Tversky Loss 用于不平衡医学分割。
- Rezatofighi et al., 2019. [Generalized Intersection over Union](https://arxiv.org/abs/1902.09630)：GIoU 作为 bbox regression loss。
- Berman et al., 2017. [The Lovasz-Softmax loss](https://arxiv.org/abs/1705.08790)：直接优化 IoU surrogate 的语义分割损失。
- Schroff et al., 2015. [FaceNet](https://arxiv.org/abs/1503.03832)：Triplet Loss 的经典应用。
- Oord et al., 2018. [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)：InfoNCE 的代表性来源。
- Radford et al., 2021. [CLIP](https://arxiv.org/abs/2103.00020) 与 [OpenAI/CLIP](https://github.com/openai/CLIP)：图文对比学习损失和开源实现。
- Goodfellow et al., 2014. [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)：GAN adversarial objective。
