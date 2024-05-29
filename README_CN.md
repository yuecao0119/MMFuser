<h2 align="center">MMFuser: 多模态的多层特征融合器</h2>

<div align="center">

[English](README.md) | 简体中文

</div>

## 📣 新闻

- **[5月 30, 2024]**  🔥🔥🔥 源码已经发布。

## 目录
- [总览](#总览)
- [效果](#效果)
- [安装](#安装)
- [训练](#训练)
- [评估](#评估)

## 总览

<p align="center">
    <img src="images/method-comprehension.png" width="70%"></a>
</p>

与以前的体系结构相比：
（a） 仅CLIP：仅使用单层视觉特征，例如倒数第二层；
（b） 混合：集成多个视觉编码器以增强图像表示；
（c） MMFuser（我们的）：一个多层特征融合模块，用于处理来自视觉主干不同层的图像特征（如CLIP）。

<p align="center">
    <img src="images/mmfuser-diagram.png" width="100%"></a>
</p>

**MMFuser**是为**多**模态**多**层特征融合而设计的，它可以增强MLLMs的视觉表示。我们发现，CLIP最后几层的功能虽然与文本对齐，但缺乏详细信息。而与之相对的是，来自浅层和中间层的输出特征包含更多的图像细节，但语义对齐较差。因此，我们的MMFuser使用CLIP最后层的输出特征作为查询（Q），使用浅层和中间层的特征作为键（K）和值（V），来改善前面几层的语义对齐程度。通过应用注意力机制，我们融合了多层特征，以获得丰富详细并和文本对齐的图像表示。

## 效果

与传统 VQA 基准测试和最近的多模式基准测试上最先进的 VLLM 进行比较。最佳结果以 **粗体** 标记，第二好结果以 <u>下划线</u> 标记。

<p align="center">
    <img src="images/eval-13b.png" width="100%"></a>
</p>

加入MMFuser后，LLaVA-1.5的性能得到了大幅提升，在多个基准测试中超越了LLaVA-1.5。
其中，在Vizwiz、MME和MMBench上的得分分别为57.4、
1585.2和69.9，分别比LLaVA-1.5高出3.8分、53.9分和2.2分。

7B规模模型及更多基准测试正在测试中。

## 安装

1. 克隆此存储库并切换到MMFuser文件夹
    ```bash
    git clone git@github.com:yuecao0119/MMFuser.git
    cd MMFuser
    ```

2. 安装程序包

    我们的项目基于[LLaVA-1.5](https://github.com/haotian-liu/LLaVA)并根据[LLaVA-1.5安装](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install)创建相关环境。

    ```bash
    conda create -n MMFuser python=3.10 -y
    conda activate MMFuser
    pip install --upgrade pip  # enable PEP 660 support
    pip install -e .
    ```

3. 安装其他软件包

    安装Flash-Attention：

    ```bash
    pip install -e ".[train]"
    pip install flash-attn==2.3.6 --no-build-isolation
    ```

    在我们的项目中使用了[Deformation-DETR](https://github.com/fundamentalvision/Deformable-DETR/tree/main)中的可变形注意力机制。需要运行以下脚本编译CUDA算子：

    ```bash
    cd llava/model/multimodal_projector/deformable_attention/ops
    sh ./make.sh
    # unit test
    python test.py
    ```


## 训练

我们的训练管线和数据集直接取自[LLaVA-v1.5](https://github.com/haotian-liu/LLaVA). 训练包括两个阶段：
- *预训练*: 在~558K图像-文本对的子集上训练projector，以连接冻结的预训练视觉编码器和冻结的大语言模型。
    ```bash
    sh scripts/mmfuser/pertrain.sh
    ```
- *指令微调*: 利用多模态指令数据LLaVA-665K对整个MLLM进行微调。
    ```bash
    sh scripts/mmfuser/finetune.sh
    ```

## 评估

我们遵循[LLaVA-v1.5](https://github.com/haotian-liu/LLaVA/tree/main)进行评估。您应该下载[eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)，并将其解压缩到`./playground/data/eval`。此外，我们还进一步实施了refcoco、mmmu和xxx的评估。请参考[Evaluation.md](./docs/Evaluation.md)准备数据。

然后，您可以在`scripts/eval`中运行我们的评估脚本。

## 👍 致谢

- [LLaVA](https://github.com/haotian-liu/LLaVA) ：我们基于该代码库改进。

## 🔒 许可证

- 该项目的大部分内容都是在[LICENSE](https://github.com/yuecao0119/MMFuser/blob/main/LICENSE)文件中的Apache 2.0许可证下发布的。
- 该服务是一个仅用于非商业用途的研究预览，受LLaMA的[License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)模型和OpenAI生成的数据的[Terms of Use](https://openai.com/policies/terms-of-use)约束。如果您发现任何潜在的违规行为，请与我们联系。