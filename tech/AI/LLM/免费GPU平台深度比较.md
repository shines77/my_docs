- # 免费GPU平台：Google Colab、Kaggle 和 Gradient 的深度比较

## 1. Kaggle

官网：[https://www.kaggle.com/](https://www.kaggle.com/)

Kaggle 是一个以举办顶级机器学习和深度学习竞赛而闻名的数据科学社区空间。它提供 Kaggle 笔记本，这些是由免费 GPU 和 CPU 支持的可共享 Jupyter 笔记本。

Kaggle 提供至少30小时的动态GPU配额，该配额每周六午夜 UTC 时间重置。我通常每周获得30至41小时的GPU配额。我喜欢 Kaggle 笔记本的一个功能是其 commit 模式，该模式允许用户在后台运行整个笔记本，这意味着我们不必在代码运行时保持浏览器打开。Kaggle 还通过简单地进行快速保存或在 commit 模式下运行笔记本来允许笔记本版本控制，这使我们可以查看以前的笔记本版本并跟踪我们的进度。

Kaggle 提供配备 16GB 内存的 P100 GPU。

**优点**

- Commit 模式允许代码在后台运行
- 允许其他用户对笔记本进行评论
- 16GB 的 GPU 内存，三者中最高
- 支持公共和私有笔记本
- 访问 Kaggle 数据集
- 笔记本版本控制

**缺点**

- 每周动态 GPU 配额为 30~40 小时
- 交互式 GPU 模式中的短暂空闲时间，为 20 分钟

## 2. Google Colab

官网：[https://colab.google/](https://colab.google/)

由 Google 推出的 Google Collaboratory 是一个配备免费 GPU 和 TPU 的 Jupyter Notebook 集成开发环境。您只需一个 Google 账户即可开始使用。Google Colab 允许您将 Google Drive 作为 Colab 项目的存储文件夹，这让数据的读取和保存变得非常简单。并且，您也可以免费获得附带 15 GB存储空间的 Google Drive 。

**优点**

- 与 Google Drive 集成，便于数据存储
- 无使用限制
- 支持私有和公共笔记本
- 执行时间长达12小时

**缺点**

- 短暂的空闲时间，约 30 分钟
- 每个会话前都需要重新挂载和验证 Google Drive
- 在 Colab 中使用 GPU，请在顶部工具栏中选择"运行时" → "更改运行时类型"，然后选择 GPU 作为硬件加速器。

根据 Google Colab 的常见问题解答，Colab 提供多种 GPU 选项，如 Nvidia K80s、T4s、P4s 和 P100s，但您不能选择特定的 GPU 类型。接下来，我们来看一下 Colab 提供的一些硬件规格。

## 3. Gradient

官网：[https://www.paperspace.com/artificial-intelligence](https://www.paperspace.com/artificial-intelligence) | [https://www.paperspace.com/gradient/deployments](https://www.paperspace.com/gradient/deployments)

Paperspace 的 Gradient 提供了端到端的基于云的 MLOps 解决方案。作为其产品系列的一部分，Gradient 提供社区笔记本，这些笔记本是运行在免费云端 GPU 和 CPU 上的公共且可共享的 Jupyter 笔记本。

**优点**

- 8 个 CPU 与 30GB RAM，比所有三者都高
- 1-6 小时的长空闲时间

**缺点**

- 如果使用免费 GPU，没有私人笔记本
- GPU 取决于可用性
- GPU 内存较低，为 8 GB
- 免费存储空间较少，为 5 GB

## 4. 其他

这个三个提供免费支持 GPU 在注册时不需要信用卡，没有免费额度限制或试用期。

如果打算创建可共享的笔记本，内哥建议用Kaggle。虽然每周的GPU配额可能对重度GPU用户来说是个问题，但是Kaggle提供了一些如何更高效地使用GPU的建议，是一个不错的起步平台。

## 5. 参考文章

- [免费GPU平台大对决：Google Colab、Kaggle和Gradient的深度比较](https://zhuanlan.zhihu.com/p/652607393)
