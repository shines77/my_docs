# 如果卸载 ComfyUI

## 1. Uninstall ComfyUI

在 `C:\Users\<你的用户名>\AppData\Local\Programs\@comfyorgcomfyui-electron` 目录中，有一个 `Uninstall ComfyUI.exe`，双击执行卸载。

## 2. 删除 ComfyUI 用户数据

ComfyUI 用户数据包含以下内容：

* .venv：Python 虚拟环境
* input：输入目录
* output：输出目录
* models：模型文件
* users：用户数据，日志，工作流等。
* custom nodes：自定义节点

用户数据默认安装在这个目录：

```
C:\Users\<你的用户名>\Documents\ComfyUI
```

用户可以自定义该目录的路径。

## 3. 删除相关目录

如果你想要完全删除 ComfyUI 桌面版 的所有文件，你可以手动删除以下文件夹：

* `C:\Users\<你的用户名>\AppData\Local\@comfyorgcomfyui-electron-updater`
* `C:\Users\<你的用户名>\AppData\Local\Programs\@comfyorgcomfyui-electron`
* `C:\Users\<你的用户名>\AppData\Roaming\ComfyUI`

以上的操作并不会删除以下你的以下文件夹，如果你需要删除对应文件的话，请手动删除：

* models：模型文件
* custom nodes：自定义节点
* input/output directories：图片输入/输出目录
