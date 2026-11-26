# 在 Windows Terminal 中使用 SSH

## 1. 远程登陆 Ubuntu 服务器使用 Claude Code

看你用什么系统，TUI类工具（OpenCode/Claude Code）最看重真彩色、Unicode 宽度、Nerd Font、性能：

- Windows：Windows Terminal（自带、免费、体验已很好）；想更强用 WezTerm。
- macOS：Ghostty（快、原生）或 iTerm2（功能全）；也可 WezTerm。
- Linux：kitty 或 WezTerm；GNOME自带终端也能用。
- 跨平台统一：WezTerm 或 Ghostty 都支持 Win/mac/Linux。

通用建议：

- 装一款  Nerd  Font（如  JetBrainMono Nerd Font）并在终端里设为默认字体，避免图标乱码。
- 确认 echo $TERM 输出含 256color，COLORTERM=truecolor。
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

如果 `echo $COLORTERM` 显示为空，看后面的第三小节： `[3. 关于 setenv 的报错]` 。

也可以使用如下命令：

```bash
export COLORTERM=truecolor
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

[在Ubuntu24.04上安装WezTerm.md](./在Ubuntu24.04上安装WezTerm.md)

事实上，Linux 上的 WezTerm 只能工作在 GUI 模式，没有 GUI 的服务器版就不要折腾了，直接使用 `wezterm ssh 你的SSH domain` 连接服务器即可。

## 3. 关于 setenv 的报错

在 Windows 上用 wezterm 以 SSH 的方式登陆服务器的时候，会看到如下的报错信息：

```text
wezterm_ssh::pty[0m > ssh: setenv COLORTERM=truecolor failed: RequestDenied: Channel request env failed. Check the AcceptEnv setting on the ssh server side. Additional errors with setting env vars in this session will be logged at debug log level.
```

这个报错本身**不是致命错误**，它只是 WezTerm 尝试通过 SSH 把本地的 `COLORTERM=truecolor` 环境变量发送到远程服务器时，被服务器拒绝了。远程服务器仍然可以正常连接和使用，只是可能无法正确识别你的终端支持真彩色（truecolor）。

### 为什么会出现这个报错

WezTerm 内置的 SSH 客户端为了确保远程端的程序能正确显示颜色和终端特性，会自动尝试发送一些环境变量（如 `COLORTERM`、`TERM_PROGRAM` 等）。但是，SSH 服务器出于安全考虑，默认只接收白名单里的环境变量。如果服务器的 `sshd_config` 中没有配置 `AcceptEnv COLORTERM`，它就会拒绝这个请求，从而产生你看到的日志。

### 解决方法

**方案一：修改远程服务器的 SSH 配置（推荐）**

如果你拥有远程服务器的管理权限，并且希望彻底消除这个报错，让 WezTerm 能正常传递颜色信息，可以在服务器端进行配置。

1.  登录到远程服务器。
2.  编辑 SSH 配置文件（通常位于 `/etc/ssh/sshd_config`）：

    ```bash
    sudo nano /etc/ssh/sshd_config
    或
    sudo vim /etc/ssh/sshd_config
    ```
    
3.  在文件中找到或添加 `AcceptEnv` 这一行。你可以专门针对 WezTerm 发送的变量进行放行：

    ```text
    # 例如原来是：
    AcceptEnv LANG LC_*
    
    # 在后面添加我们的 Env 变量，改为：
    AcceptEnv LANG LC_* TERM COLORTERM TERM_PROGRAM TERM_PROGRAM_VERSION WEZTERM_* WSLENV WT_SESSION WT_PROFILE_ID
    ```

    如果文件中已经存在 `AcceptEnv LANG LC_*` 之类的行，直接在后面追加即可，`AcceptEnv` 指令是累加的。

4.  保存文件后，重启 SSH 服务使配置生效：

    ```bash
    sudo systemctl restart ssh
    
    # 保险一点，再执行一次
    sudo systemctl daemon-reload
    ```

    完成这些步骤后，再次通过 WezTerm 连接，该报错就不会再出现了。

**方案二：忽略报错（如果你不介意颜色可能不准）**

如果你没有服务器权限，或者觉得这个报错很烦但不想改配置，其实**完全可以忽略它**。这只是一个警告级别的日志，不会影响你正常登录和使用远程服务器。只是某些依赖 `COLORTERM` 变量来判断是否启用真彩色的程序（如 Vim、Neovim 或某些 CLI 工具）在远程端可能会退回到 256 色模式。

**补充说明**：根据 WezTerm 的代码提交记录，开发者也在后续版本中调整了逻辑，遇到这种被服务器拒绝的情况时会仅记录日志而不再“硬失败”（hard fail），所以你看到的是一个警告而非错误。
