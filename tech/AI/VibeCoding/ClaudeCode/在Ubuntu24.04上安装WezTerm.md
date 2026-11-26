# Ubuntu 24.04 上安装 WezTerm

## 1. 安装 WezTerm

在 Ubuntu 24.04 上安装 WezTerm，有一个需要特别注意的地方：**官方目前没有为 Ubuntu 24.04 提供稳定版的 `.deb` 安装包或 APT 源**，只提供了 Nightly（每夜构建）版本。因此，你需要根据自己的需求在几种方式中选择。

### 方案一：使用 APT 仓库（推荐，获取最新版本）

这是最方便、也最推荐的方式，可以让你通过 `apt` 命令安装和更新。不过，由于 Ubuntu 24.04 的官方稳定版包缺失，通过这个方式安装的其实是 **Nightly** 版本。

1. **添加 GPG 密钥和软件源**：在终端中执行以下命令。

   ```bash
   curl -fsSL https://apt.fury.io/wez/gpg.key | sudo gpg --yes --dearmor -o /usr/share/keyrings/wezterm-fury.gpg
   
   echo 'deb [signed-by=/usr/share/keyrings/wezterm-fury.gpg] https://apt.fury.io/wez/ * *' | sudo tee /etc/apt/sources.list.d/wezterm.list
   
   sudo chmod 644 /usr/share/keyrings/wezterm-fury.gpg
   ```

2.  **更新并安装**：

    ```bash
    sudo apt update
    sudo apt install wezterm-nightly
    ```

    **注意**：安装的是 `wezterm-nightly`，而不是 `wezterm`。Nightly 版本功能最新，但理论上可能不如稳定版稳定。

### 方案二：使用 AppImage（免安装，尝鲜用）

如果你不想动系统的软件源，或者只想临时用一下，AppImage 是最干净的选择。

1.  前往 WezTerm 的 GitHub Releases 页面，下载适用于 Linux 的 AppImage 文件。
2.  赋予执行权限并运行：

    ```bash
    chmod +x WezTerm-*.AppImage
    ./WezTerm-*.AppImage
    ```
    
    为了方便，你可以把它移动到 `~/bin` 或 `/usr/local/bin` 目录下，并重命名为 `wezterm`。

### 方案三：使用 Flatpak（沙盒环境）

如果你习惯使用 Flatpak，且系统已经配置好了 Flathub，也可以用它安装。但需要注意，Flatpak 的沙盒特性可能会限制 WezTerm 的部分功能（例如，某些 CLI 子命令的首次运行会阻塞，或无法准确获取当前面板的工作目录），因此官方更推荐原生安装方式。

```bash
flatpak install flathub org.wezfurlong.wezterm
flatpak run org.wezfurlong.wezterm
```

### 总结建议

结合你之前的需求（在 Ubuntu 上运行 Claude Code），**推荐使用方案一（APT 源安装 Nightly 版）**。这样安装的 `wezterm` 命令能更好地与系统集成，也方便后续更新。如果你对稳定性有极高要求，可以观望官方后续是否会为 24.04 提供稳定版包，或者暂时使用 Ubuntu 22.04 的稳定版 `.deb` 包尝试安装（存在兼容性风险，需自行测试）。
