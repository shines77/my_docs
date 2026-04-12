# OpenAI Codex CLI

## npm 安装

```bash
# Install using npm
npm install -g @openai/codex
```

或者

```bash
# Install using Homebrew
brew install --cask codex
```

更新版本时，也使用同样的命令。

## 安装 Rust

因为要用源码来编译，是需要 Rust 的，Codex 内核是用 Rust 写的，nmp 只是一个壳而已。

在 Ubuntu 24.04 上安装 Rust，最标准、最灵活的方式是使用官方工具 rustup。它会帮你安装最新稳定版的 Rust 编译器 (rustc) 和包管理工具 (cargo)。

### 安装步骤

1. 打开终端，首先需要安装 curl 和编译工具，这能为后续安装和编译 Rust 代码提供基础环境。

```bash
sudo apt update
sudo apt install -y curl build-essential
```

`build-essential` 包含了 Rust 编译代码时需要的链接器等工具。

2. 下载并运行安装脚本

使用 `curl` 命令下载 Rust 官方安装脚本并执行。

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

命令执行后，安装程序会给出选项。直接按回车键选择默认安装（选项 1）即可。

3. 配置环境变量

安装脚本会自动配置 PATH 环境变量，但为了让当前终端立即生效，需要手动运行以下命令。

```bash
source "$HOME/.cargo/env"
```

你也可以关闭并重新打开终端，效果是一样的。

4. 验证安装

最后，通过检查版本号来确认 Rust 是否安装成功。

```bash
rustc --version
cargo --version
```

如果屏幕上正常显示了 rustc 和 cargo 的版本号，就说明安装已经完成了。

## 实用建议

- **为什么不用 apt 安装**：Ubuntu 的软件源中虽然也有 Rust，但版本通常比较旧（例如 24.04 源中是 1.75 版），而通过 rustup 安装可以确保你使用的是最新的稳定版，能用到最新的语言特性和工具。

- **国内用户加速（可选）**：如果你在中国大陆，下载可能会比较慢。可以在安装前运行下面两行命令，切换到国内的镜像源，能显著提升下载速度。

```bash
export RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
export RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup
```

- **后续更新**：Rust 每六周发布一个新版本。之后想更新到最新版时，只需在终端执行 `rustup update` 这条命令就可以了。

成功安装后，就可以运行 `cargo new hello-rust` 来创建你的第一个 Rust 项目了。

## Git编译安装

[Codex 官网](https://github.com/openai/codex)

### 系统需求

| 需求 | 详细 |
| ---- | :--: |
| 操作系统 | macOS 12+, Ubuntu 20.04+/Debian 10+, or Windows 11 via WSL2 |
| Git (可选，推荐) | 2.23+ for built-in PR helpers |
| 内存 | 至少 4GB (推荐 8GB) |

注：如果是 Windows，可以借助 WSL2 来编译。

### DotSlash

GitHub 版本还包含了一个名为 `codex` 的 [DotSlash](https://dotslash-cli.com/) 文件，用于 Codex 命令行界面（CLI）。使用 DotSlash 文件可以在源代码管理中进行轻量级提交，以确保所有贡献者无论使用哪个平台进行开发，都使用同一版本的可执行文件。

### 从源码构建

编译和构建：

```bash
# Clone the repository and navigate to the root of the Cargo workspace.
git clone https://github.com/openai/codex.git
cd codex/codex-rs

# Install the Rust toolchain, if necessary.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup component add rustfmt
rustup component add clippy
# Install helper tools used by the workspace justfile:
cargo install just
# Optional: install nextest for the `just test` helper
cargo install --locked cargo-nextest

# Build Codex.
cargo build

# Launch the TUI with a sample prompt.
cargo run --bin codex -- "explain this codebase to me"

# After making changes, use the root justfile helpers (they default to codex-rs):
just fmt
just fix -p <crate-you-touched>

# Run the relevant tests (project-specific is fastest), for example:
cargo test -p codex-tui
# If you have cargo-nextest installed, `just test` runs the test suite via nextest:
just test
# Avoid `--all-features` for routine local runs because it increases build
# time and `target/` disk usage by compiling additional feature combinations.
# If you specifically want full feature coverage, use:
cargo test --all-features
```

## 追踪/详细日志记录

Codex 是用 Rust 编写的，因此它遵循 RUST_LOG 环境变量来配置其日志记录行为。

TUI（文本用户界面）默认设置为 `RUST_LOG=codex_core=info,codex_tui=info,codex_rmcp_client=info`，并且日志消息默认会写入 `~/.codex/log/codex-tui.log`。对于单次运行，您可以使用 `-c log_dir=...`（例如，`-c log_dir=./.codex-log`）来覆盖日志目录。

```bash
tail -F ~/.codex/log/codex-tui.log
```

相比之下，非交互模式（codex exec）默认为 `RUST_LOG=error`，但消息会直接打印出来，因此无需监控单独的文件。

有关配置选项的更多信息，请参阅 [RUST_LOG](https://docs.rs/env_logger/latest/env_logger/#enabling-logging) 的 Rust 文档。

## 参考文章

- [Codex 官网](https://github.com/openai/codex)

- [Codex: Installing & building](https://github.com/openai/codex/blob/main/docs/install.md)
