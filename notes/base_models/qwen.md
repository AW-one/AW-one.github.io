# Qwen 系列：从通用 LLM 到多模态与推理模型的开源基座模型家族

> **作者**：Qwen Team, Alibaba Cloud
>
> **机构**：阿里巴巴通义千问团队
>
> **发布时间**：Qwen 技术报告 2023 年；Qwen2 / Qwen2.5 / Qwen3 主要发布于 2024-2025 年
>
> **论文链接**：[Qwen Technical Report](https://arxiv.org/abs/2309.16609) | [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) | [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) | [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | [代码](https://github.com/QwenLM)
>
> **分类标签**：`LLM` `Qwen` `Transformer` `Instruction Tuning` `Multilingual` `Reasoning`

---

## 一句话总结

Qwen 是阿里通义千问团队推出的一组开源基座模型家族，核心是 decoder-only Transformer，并在中文、英文、多语言、代码、数学、工具调用和多模态方向持续扩展。它不是单个模型，而是一条从 base model 到 instruction model、coder model、VL model、reasoning model 的完整路线。

---

## 一、问题与动机

### 1. 为什么需要 Qwen 这样的基座模型

大语言模型的基础能力来自大规模预训练，但不同地区和应用场景对模型有不同需求：

- 中文语义、中文知识和中英混合表达；
- 代码生成、数学推理、工具调用等专业能力；
- 较开放的权重和生态，方便本地部署、微调和研究；
- 从小模型到大模型的完整尺寸，适配不同算力预算；
- 面向真实应用的 chat、agent、RAG、多模态扩展。

Qwen 的定位可以理解为：在国际开源 LLM 生态中提供一个中文和多语言能力都较强、工程可用性较高的基座模型系列。

### 2. Base model 与 Chat model 的区别

Qwen 系列通常会同时提供 base 和 instruct / chat 版本：

| 类型 | 训练目标 | 使用场景 |
| --- | --- | --- |
| Base model | next-token prediction | 继续预训练、领域微调、研究实验 |
| Instruct / Chat model | 指令微调 + 对齐 | 对话、问答、工具调用、应用部署 |

Base model 更像“语言建模底座”，没有强行学会遵循用户指令；Chat model 则在 base model 上通过 SFT、偏好优化等流程，让模型更适合和人交互。

---

## 二、核心方法

### 1. 架构：decoder-only Transformer

Qwen 和主流 LLM 一样，使用 decoder-only Transformer。给定 token 序列：

$$
x = (x_1, x_2, ..., x_n)
$$

预训练目标是自回归 next-token prediction：

$$
\mathcal{L}_{\text{LM}} =
- \sum_{t=1}^{n}
\log p_\theta(x_t \mid x_{<t})
$$

也就是说，模型不断学习“根据前文预测下一个 token”。看起来简单，但当数据足够大、模型足够大时，这个目标会诱导出语法、知识、推理、代码、翻译和一定程度的工具使用能力。

### 2. Tokenizer：多语言与代码友好

Tokenizer 是 Qwen 系列比较重要的工程基础。一个好的 tokenizer 会影响：

- 中文是否被切得过碎；
- 代码符号和缩进是否保留得自然；
- 多语言文本是否有较高压缩率；
- 长上下文场景下 token budget 是否浪费。

Qwen 的 tokenizer 设计通常会兼顾中文、英文、多语言和代码场景，使模型在中文任务上不会像一些英文主导 tokenizer 那样付出过高 token 成本。

### 3. 训练数据：通用能力来自混合语料

Qwen 技术报告强调了大规模、多来源、多语言的预训练数据。可以把训练数据粗略分成几类：

| 数据类型 | 贡献能力 |
| --- | --- |
| Web text | 通用语言、世界知识、百科表达 |
| Books / Articles | 长文理解、叙事、专业文本 |
| Code | 编程、函数调用、结构化生成 |
| Math / STEM | 数学推理、符号表达、解题步骤 |
| Multilingual data | 翻译、多语言理解、跨语言迁移 |
| Synthetic / instruction data | 指令遵循、对话格式、工具调用 |

需要注意的是，模型能力不是由某一种数据单独决定的，而是数据规模、清洗质量、混合比例、训练阶段和后训练共同作用的结果。

### 4. 后训练：从会续写到会回答

Base model 只会预测下一个 token，不天然知道“用户希望我怎么回答”。因此 instruct / chat 模型通常需要后训练：

1. **SFT**：用高质量指令数据教模型遵循任务格式；
2. **Preference Optimization**：用人类或模型偏好让回答更有帮助、更安全；
3. **Tool / Function Calling Tuning**：让模型学会输出结构化工具调用；
4. **Long Context Tuning**：增强长文档、长对话和长代码处理能力；
5. **Reasoning Tuning**：强化数学、代码、规划和多步推理。

这个过程可以理解为：预训练让模型拥有“知识和语言能力”，后训练让模型学会“把能力用成用户想要的形状”。

### 5. Qwen 系列的分化路线

Qwen 不只是一个通用 chat 模型，而是逐渐形成了多个方向：

| 分支 | 关注能力 |
| --- | --- |
| Qwen / Qwen2 / Qwen2.5 / Qwen3 | 通用语言、知识、推理、长上下文 |
| Qwen-Coder | 代码补全、代码生成、代码推理 |
| Qwen-Math | 数学解题、符号推理 |
| Qwen-VL | 图文理解、视觉问答、OCR、定位 |
| Qwen-Audio | 语音与音频理解 |
| Qwen-Agent 生态 | 工具调用、RAG、工作流编排 |

这说明 Qwen 的路线不是只训练一个更大的模型，而是围绕同一个基础生态做能力专门化。

---

## 三、能力与实验结论

### 1. 通用语言与中文能力

Qwen 系列在中文知识、中文问答、中英混合表达和多语言理解上通常表现较强。对国内应用来说，这一点很重要，因为很多开源模型在英文 benchmark 上表现不错，但中文语境中的事实表达、口语习惯、成语、政策文本、教育题目和代码注释不一定稳定。

### 2. 代码与数学能力

Qwen2.5 之后，代码和数学分支的存在感明显增强。值得关注的点不是单一 benchmark 分数，而是能力结构：

- 能否理解已有代码上下文；
- 能否生成可运行、可维护的代码；
- 能否处理多步数学推理；
- 能否在解题时保持符号一致；
- 能否和工具调用、代码执行结合。

对于实际开发场景，模型的“可验证输出”比单纯回答流畅更重要。

### 3. 长上下文与工具调用

长上下文能力让模型可以处理长文档、长代码仓库、长对话历史，但它不等于真正理解所有细节。好的长上下文模型还需要：

- 能定位关键信息；
- 能跨段落整合；
- 能忽略无关上下文；
- 能在输出中保持引用和约束一致；
- 能和 RAG、搜索、代码工具配合。

工具调用能力则让模型从“只会说”变成“可以做”。在 agent 场景中，Qwen 这类模型常被用来做规划、函数参数生成、检索增强和本地自动化。

---

## 四、局限性与未来方向

- **版本差异需要仔细区分**：Qwen、Qwen2、Qwen2.5、Qwen3 不是同一个模型，能力和许可证细节也不应混写；
- **Benchmark 不等于真实应用**：高分模型在具体业务数据、工具链和长尾任务上仍可能不稳定；
- **推理能力仍依赖后训练策略**：是否显式 thinking、是否使用工具、是否允许多步草稿，会显著影响表现；
- **中文强不代表领域强**：医疗、法律、金融、科研等领域仍需要检索、校验和领域数据适配；
- **小模型部署要关注幻觉**：小尺寸模型成本低，但更容易在复杂任务中自信犯错。

未来方向可以继续看：

- 更强的原生多模态统一模型；
- 更稳定的长上下文检索与引用；
- 面向 agent 的工具调用和任务执行能力；
- 本地小模型的推理蒸馏；
- 面向行业场景的持续预训练和对齐。

---

## 五、个人思考

### 1. Qwen 的价值在于“生态完整”

单看某个 benchmark，Qwen 只是开源模型竞争中的一个选手；但从生态角度看，它的价值更大：通用模型、代码模型、数学模型、视觉语言模型、音频模型、agent 工具链都有覆盖。这让它很适合作为中文 AI 应用和本地部署实验的默认候选之一。

### 2. Base model 更适合研究，Instruct model 更适合应用

如果目标是继续预训练、做领域适配、研究模型内部能力，base model 更干净；如果目标是做问答助手、知识库、自动化工具，instruct model 会省很多后训练成本。很多应用失败不是因为模型太弱，而是没有选对 base / instruct 版本。

### 3. 和视觉生成基座模型的联系

Qwen 和 FLUX / Diffusion 看起来属于不同方向，一个是语言，一个是图像生成，但它们的趋势很像：

- 都在从单任务模型走向通用基座；
- 都依赖大规模数据和统一训练目标；
- 都通过后训练或条件注入适配具体任务；
- 都在向多模态和 agent workflow 扩展。

后续多模态系统很可能不是“一个 LLM + 一个图像模型”的简单拼接，而是语言、视觉、工具和记忆围绕统一上下文协同工作。

---

## 参考

- Qwen Team, 2023. [Qwen Technical Report](https://arxiv.org/abs/2309.16609)：Qwen 第一代技术报告，介绍模型、数据、训练和评测。
- Qwen Team, 2024. [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)：Qwen2 系列报告，重点关注模型扩展、多语言和长上下文能力。
- Qwen Team, 2024. [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)：Qwen2.5 系列报告，覆盖通用、代码、数学等方向的能力升级。
- Qwen Team, 2025. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)：Qwen3 系列报告，关注推理、混合 thinking/non-thinking 模式和模型族扩展。
- QwenLM. [GitHub Organization](https://github.com/QwenLM)：Qwen 系列模型、工具和示例代码入口。
