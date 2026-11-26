# 在 Windows Terminal 中使用 SSH

## 1. 远程登陆 Ubuntu 服务器使用 Claude Code

看你用什么系统，TUI类工具（OpenCode/Claude Code）最看重真彩色、Unicode 宽度、Nerd Font、性能：

- Windows：Windows Terminal（自带、免费、体验已很好）；想更强用 WezTerm。
- macOS：Ghostty（快、原生）或 iTerm2（功能全）；也可 WezTerm。
- Linux：kitty 或 WezTerm；GNOME自带终端也能用。
- 跨平台统一：WezTerm 或 Ghostty 都支持 Win/mac/Linux。

通用建议：

- 装一款  Nerd  Font（如  JetBrainMono Nerd Font）并在终端里设为默认字体，避免图标乱码。
- 确认 echo #TERM 输出含 256color，COLORTERM=truecolor。
- 若在远程服务器上跑，最终显示取决于你的本地终端，服务器端只影响能力探测。

### 1.1 装 Windows Terminal

最好以管理员身份打开终端，更新或安装 Microsoft WindowsTerminal：

```powershell
winget install --id Microsoft.WindowsTerminal -e
```

执行结果：

```bash
找到已安装的现有包。正在尝试升级已安装的包...
找不到可用的升级。
配置的源中没有可用的较新的包版本。
```

### 1.2 装 Nerd Font （选一个即可）

安装 `JetBrainsMono Nerd Font` 字体（[DEVCOM.JetBrainsMonoNerdFont] 版本 3.3.0）

```powershell
winget install --id DEVCOM.JetBrainsMonoNerdFont -e
```

执行结果：

```bash
已找到 JetBrainsMono Nerd Font [DEVCOM.JetBrainsMonoNerdFont] 版本 3.3.0
此应用程序由其所有者授权给你。

Microsoft 对第三方程序包概不负责，也不向第三方程序包授予任何许可证。
正在下载 https://github.com/DevelopersCommunity/wix-JetBrainsMonoNF/releases/download/v3.3.0/JetBrainsMono.msi
  ██████████████████████████████   112 MB /  112 MB

已成功验证安装程序哈希
正在启动程序包安装...
安装程序将请求以管理员身份运行。期待提示。
已成功安装
```

### 1.3 配置字体

打开 WT -> 设置 -> Ubuntu/SSH 配置文件 -> 外观 -> 字体选 `JetBrainsMono Nerd Font`，字号11-12。

### 1.4 SSH登录(在WT里)

```bash
ssh user@你的服务器IP
```

建议在 `C:\Users\你\.ssh\config` 写个别名：

```bash
Host ubuntu
    User 用户名
    HostName 服务器IP
    ServerAliveInterval 120
```

之后输入 `ssh ubuntu` 即可。

### 1.5 服务器端确认

```bash
echo $TERM         # 应含 256color, xterm-256color
echo $COLORTERM    # truecolor
```

这样 OpenCode/ClaudeCode 的颜色、图标、按键都会正常。

## 2. WT 存在的问题，换 WezTerm

### 2.1 安装 WezTerm
问题：

但是 WT 也有一个问题，就是我的 QQ 拼音输入法在WT里看不到选字的候选框。

回答：

这是 WT 在 Win10 上对部分第三方输入法（QQ拼音）的已知 IME 兼容问题，候选框定位/渲染会出错。

按成本从低到高：

1. 更新 Windows Terminal：升级到最新版 (`winget upgrade Microsoft.WindowsTerminal`)，新版修了不少 IME 问题
2. 临时换输入法：切到微软拼音，它在 WT 里候选框正常；QQ 拼音可留着在别处用。
3. 更新/重装 QQ 拼音：到最新版，并在其设置里试试「兼容模式」。
4. 换终端：WezTerm 对中文 IME 支持最好，候选框正常，且同样支持真彩色 + Nerd Font：`winget install --id wez.wezterm -e`，Ghostty（Win 版较新）也可试。
5. 终极方案：装 WSL2 + Ubuntu，在 WSL 里跑 TUI，输入走 Windows IME，体验更稳。

建议顺序：先升级 WT，不行就换 WezTerm，基本能解决候选框问题。要我给你 WezTerm 连Ubuntu 的配置吗？

---

```powershell
winget install --id WezTerm.WezTerm -e

找不到与输入条件匹配的程序包。
```

回答：

包 ID不对，WezTerm 的 winget ID 是 `wez.wezterm`。先搜索确认再装：

```powershell
winget search wezterm
winget install --id wez.wezterm -e
```

如果还是搜不到，可能是 winget 源太旧，先更新源：

```powershell
winget source update
```

再不行就直接去官网下安装包：[https://wezterm.org/install/windows.html](https://wezterm.org/install/windows.html) 。

### 2.2 配置 WezTerm SSH Domain（推荐，支持会话保持）

如果你希望网络波动或断线后，远程 Ubuntu 上的工作现场（如正在跑的 Claude Code 会话）不丢失，可以配置 SSH Domain。但这需要**本地和远程 Ubuntu 上都安装了 WezTerm**（远程端用于运行 mux 服务）。

**步骤 1：修改本地 WezTerm 配置**

打开你的 `~/.wezterm.lua`（或 `~/.config/wezterm/wezterm.lua`），添加 `ssh_domains` 配置：

```lua
local wezterm = require 'wezterm'
local config = wezterm.config_builder()

config.ssh_domains = {
  {
    name = 'my-ubuntu',
    -- 换成你的 Ubuntu IP
    -- 如果端口非 22，可以加 remote_address = '192.168.x.x:2222'
    remote_address = '192.168.x.x',
    username = '你的用户名',
  },
}

return config
```

**步骤 2：连接**

重启 WezTerm 后，运行：

```bash
wezterm connect my-ubuntu
```

这样连接后，即使本地网络断开，远程 Ubuntu 上的 WezTerm 守护进程仍在运行，重新连接即可恢复之前的会话。

------

#### 一个便捷技巧：自动生成 Domain

如果你的 `~/.ssh/config` 里已经配好了 Ubuntu 的别名，WezTerm 从较新版本开始会自动把这些主机生成对应的 Domain。你不需要手写 Lua 配置，直接尝试：

```bash
# 普通 SSH 会话（不保持）
wezterm connect SSH:你的 ssh 配置别名

# 多路复用会话（保持，需远程也装了 WezTerm）
wezterm connect SSHMUX:你的 ssh 配置别名
```

**建议**：如果你只是日常连上去跑跑命令，**方案一**足够。如果你打算在 Ubuntu 上长时间运行 Claude Code 或其他任务，担心网络闪断导致前功尽弃，花几分钟配置**方案二**会更省心。

### 2.3 Ubuntu 24.04 上安装 WezTerm

关于如何在 Ubuntu 24.04 上安装 WezTerm，请查阅另一篇文章：

[Ubuntu24.04上安装WezTerm.md](./Ubuntu24.04上安装WezTerm.md)
