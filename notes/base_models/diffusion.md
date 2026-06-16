# Diffusion Models / DDPM: 从逐步加噪到逐步去噪的生成模型范式

> **作者**：Jonathan Ho, Ajay Jain, Pieter Abbeel 等
>
> **机构**：UC Berkeley, Google Research 等
>
> **发布时间**：DDPM 论文 2020 年；相关 score-based / DDIM / latent diffusion 工作集中在 2020-2022 年
>
> **论文链接**：[DDPM](https://arxiv.org/abs/2006.11239) | [DDIM](https://arxiv.org/abs/2010.02502) | [Score SDE](https://arxiv.org/abs/2011.13456) | [Latent Diffusion](https://arxiv.org/abs/2112.10752)
>
> **分类标签**：`Diffusion Model` `DDPM` `Score Matching` `Image Generation` `Denoising`

---

## 一句话总结

Diffusion Model 的核心思想是：训练时把真实数据逐步加噪成接近高斯噪声，生成时再让神经网络学习反过来一步步去噪。它用一个相对稳定的似然 / 去噪训练目标，把高质量图像生成从 GAN 式对抗训练转向了“可控的迭代生成”。

---

## 一、问题与动机

### 1. 生成模型想解决什么

图像生成模型的目标是学习真实数据分布：

$$
p_\theta(x) \approx p_{\text{data}}(x)
$$

也就是说，模型不是简单记住训练图片，而是学会“什么样的图像像真实数据”。早期常见路线包括：

- **VAE**：训练稳定，有显式 latent space，但生成图像容易偏糊；
- **GAN**：图像锐利，但训练不稳定，容易 mode collapse；
- **Autoregressive model**：概率建模清晰，但高分辨率图像生成速度慢；
- **Flow model**：似然计算方便，但架构限制较强。

Diffusion Model 的吸引力在于：训练目标稳定，覆盖模式能力强，并且随着模型和数据规模增长，生成质量可以持续提升。

### 2. 为什么要“先破坏再恢复”

直接从随机噪声生成一张高质量图像很难，因为噪声和图像之间的映射非常复杂。Diffusion 把这个困难问题拆成很多小问题：

1. 正向过程：一点点给真实图片加噪；
2. 反向过程：一点点预测如何去掉噪声；
3. 每一步只需要解决一个局部 denoising 任务；
4. 多步组合后，就能从噪声逐渐走回数据分布。

这个思路很像把“画一张完整图片”变成“不断修正当前图片”。每一步修正都不大，但足够多的步骤可以形成复杂结构。

---

## 二、核心方法

### 1. Forward process：固定的加噪过程

DDPM 先定义一个不可学习的正向马尔可夫链：

$$
q(x_t \mid x_{t-1}) =
\mathcal{N}
\left(
x_t;
\sqrt{1-\beta_t}x_{t-1},
\beta_t I
\right)
$$

其中：

| 符号 | 含义 |
| --- | --- |
| $x_0$ | 原始真实图像 |
| $x_t$ | 第 $t$ 步加噪后的图像 |
| $\beta_t$ | 第 $t$ 步噪声强度 |
| $T$ | 总扩散步数 |

经过足够多步后，$x_T$ 会接近标准高斯噪声：

$$
x_T \sim \mathcal{N}(0, I)
$$

一个关键技巧是可以直接从 $x_0$ 采样任意时间步 $x_t$，不需要真的循环加噪：

$$
q(x_t \mid x_0) =
\mathcal{N}
\left(
x_t;
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right)
$$

等价写法是：

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 +
\sqrt{1-\bar{\alpha}_t}\epsilon,
\quad \epsilon \sim \mathcal{N}(0, I)
$$

这里 $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$。

### 2. Reverse process：学习去噪分布

生成时要从噪声 $x_T$ 逐步走回 $x_0$。反向过程写作：

$$
p_\theta(x_{t-1} \mid x_t) =
\mathcal{N}
\left(
x_{t-1};
\mu_\theta(x_t, t),
\Sigma_\theta(x_t, t)
\right)
$$

直觉上，模型每一步都回答一个问题：

> 当前这张 noisy image 里，哪些部分像噪声，哪些部分像真实图像结构？

在 DDPM 中，常见做法不是直接预测 $x_{t-1}$，而是预测加入的噪声 $\epsilon$：

$$
\mathcal{L}_{\text{simple}} =
\mathbb{E}_{x_0, \epsilon, t}
\left[
\left\|
\epsilon -
\epsilon_\theta(x_t, t)
\right\|_2^2
\right]
$$

这个目标非常重要：它把复杂的概率建模问题变成了一个监督式的噪声回归问题。

### 3. U-Net backbone：为什么早期 diffusion 多用 U-Net

早期图像 diffusion 模型大多使用 U-Net，原因是它天然适合图像结构：

- 下采样路径提取全局语义；
- 上采样路径恢复空间细节；
- skip connection 保留局部纹理；
- attention block 可以增强长距离关系；
- timestep embedding 告诉网络当前噪声强度。

文本到图像模型通常会把文本 embedding 通过 cross-attention 注入 U-Net，使网络在去噪时同时考虑图像当前状态和文本条件。

### 4. Sampling：从慢速多步到加速采样

原始 DDPM 通常需要很多 denoising steps，生成质量高但速度慢。后续工作主要沿着三个方向加速：

| 方法 | 核心直觉 |
| --- | --- |
| DDIM | 把随机采样改成确定性或半确定性路径，减少步数 |
| Improved DDPM | 改进噪声 schedule、方差学习和训练细节 |
| DPM-Solver / ODE Solver | 把采样看成微分方程求解 |
| Distillation | 用少步学生模型拟合多步老师模型 |
| Latent Diffusion | 不在像素空间生成，而是在压缩 latent space 生成 |

Stable Diffusion、FLUX、SDXL 等现代图像模型都可以看成 diffusion / flow matching 思想在更大模型、更好文本编码器、更强数据和更高效采样器上的延伸。

### 5. 和 Score Matching / Flow Matching 的关系

Diffusion 也可以从 score matching 角度理解。模型学习的是数据分布在不同噪声尺度下的 score：

$$
\nabla_x \log p_t(x)
$$

这个 score 指向“更像真实数据”的方向。生成过程可以理解为：从噪声出发，沿着 score 指导的方向逐步移动到高概率数据区域。

Flow Matching / Rectified Flow 则进一步把生成过程写成从噪声到数据的连续运输路径，目标是学习速度场：

$$
v_\theta(x_t, t) \approx \frac{d x_t}{d t}
$$

所以 DDPM、Score SDE、Rectified Flow 不是完全割裂的路线，而是从不同角度描述“从噪声分布到数据分布”的生成过程。

---

## 三、实验结果与影响

### 1. DDPM 的意义

DDPM 证明了扩散模型可以在图像质量上接近甚至超过当时的强 GAN baseline，同时训练更加稳定。它的贡献不只是一个模型，而是让研究社区重新认识到：

- 多步迭代生成可以得到非常强的视觉质量；
- 简单的噪声预测损失足以训练高质量生成器；
- 生成质量可以通过模型规模、数据规模、采样器持续提升；
- 条件生成、编辑、超分、修复等任务可以共享同一套 denoising 框架。

### 2. 后续模型如何继承 diffusion

| 模型 / 范式 | 继承点 | 变化 |
| --- | --- | --- |
| Stable Diffusion | latent diffusion | 在 VAE latent 中去噪，显著降低成本 |
| SDXL | latent diffusion + 更大 U-Net | 提升分辨率、文本理解和美学质量 |
| ControlNet | conditional diffusion | 给 U-Net 增加可控结构条件 |
| DiT | diffusion transformer | 用 Transformer 替代 U-Net |
| FLUX | rectified flow transformer | 用 flow matching 和 DiT 进一步扩展 |

可以把 diffusion 看作现代视觉生成的底层语言：不管外层是 U-Net、DiT、ControlNet、LoRA 还是 multimodal editing，本质上经常仍是在学习某种“如何从噪声恢复数据”的路径。

---

## 四、局限性与未来方向

- **采样成本高**：多步去噪天然比一次前向生成慢，需要采样器、蒸馏或 latent space 降低成本；
- **精确控制困难**：纯文本条件不够精细，容易出现位置、数量、文字和空间关系错误；
- **训练数据依赖强**：生成质量高度依赖数据清洗、caption 质量和分布覆盖；
- **可解释性有限**：模型学到的 denoising trajectory 很强，但中间每一步的语义变化不总是可解释；
- **评测复杂**：图像生成不是单一指标问题，需要同时看 prompt following、真实感、美学、细节、文字、偏见与安全。

未来方向我认为主要在：

- 更少步数的高质量采样；
- 更强的结构控制和编辑一致性；
- 更好的 multimodal condition 表达；
- 从图像扩展到视频、3D、世界模型；
- 把 diffusion / flow 与 autoregressive 或 agentic workflow 结合。

---

## 五、个人思考

### 1. Diffusion 的核心不是“加噪技巧”，而是任务分解

最值得记住的点是：Diffusion 把一个极难的生成问题拆成了许多简单的去噪问题。每一步只做局部修正，最终组合出复杂样本。这种思想非常适合高维连续数据，比如图像、视频、音频和 3D。

### 2. 为什么它比 GAN 更适合成为基座模型

GAN 很擅长生成锐利图像，但训练依赖生成器和判别器的动态博弈。Diffusion 的训练更像标准监督学习：给定 noisy sample 和 timestep，预测噪声或速度。这让它更容易 scale，也更容易接入文本、图像、mask、depth、pose 等条件。

### 3. 和当前研究的联系

理解 DDPM 后，再看 Stable Diffusion、ControlNet、SDXL、DiT、FLUX 会清楚很多：

- Stable Diffusion 解决的是“在哪个空间去噪”；
- ControlNet 解决的是“用什么结构条件控制去噪”；
- DiT 解决的是“用什么 backbone 表达去噪函数”；
- FLUX / Rectified Flow 解决的是“用什么连续路径和训练目标连接噪声与数据”。

所以 DDPM 是后续视觉生成模型的地基，后面的许多工作都是在改空间、改条件、改架构、改路径、改采样器。

---

## 参考

- Ho et al., 2020. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)：DDPM 的核心论文，建立了噪声预测训练范式。
- Song et al., 2020. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)：提出 DDIM，加速采样并引入确定性生成路径。
- Song et al., 2021. [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)：从连续 SDE 角度统一 score-based generative modeling。
- Nichol and Dhariwal, 2021. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)：改进 DDPM 的训练和采样细节。
- Rombach et al., 2022. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)：Stable Diffusion 的基础思路，在 latent space 中进行扩散生成。
