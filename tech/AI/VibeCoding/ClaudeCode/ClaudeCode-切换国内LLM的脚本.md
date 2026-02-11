# ClaudeCode 切换国内 LLM 的脚本

## 问题

请调查智谱、MiniMax、Kimi 等供应商配置 Claude Code 的方法，尤其是 anthropic 格式 API 的 base url。

询问用户使用的供应商名称，并帮用户调查供应商的 Claude Code 配置方法，尤其是 anthropic API 的地址。

之后帮用户生成一个简单优雅的脚本，通过配置环境变量来启动 claude 命令，脚本提供 1234.. 的选项来配置，让用户输入来启动不同供应商的配置。

重要：环境变量的配置方法只能对当前启动的 claude 有效。API Key 请填写123，默认Model填写为 claude-4-5-sonnet-20251001，让用户自己修改，模型名称的环境变量默认填写该供应商最新的。

请你生成 Windows 的版本和 Mac 版本的脚本，脚本里必须是英文。注意 Windows 版本不要用括号，不要用任何容易导致 bat 文件出错的字符。脚本要自动在用户目录创建 .claude.json，并写入默认内容。

macOS路径：~/.claude.json

Windows 路径：%USERPROFILE%\.claude.json

调查完别废话，直接输出脚本内容。

## 概述

根据您的要求，我已调查了智谱、MiniMax、Kimi 等供应商的 Claude Code 配置方法，并生成以下脚本。

国际用户使用 https://api.minimax.io/anthropic，中国大陆用户使用 https://api.minimaxi.com/anthropic 。

## Windows 版本 (setup_claude.bat):

```batch
@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo Claude Code Configuration Script for Windows
echo ============================================

if exist "%USERPROFILE%\.claude.json" (
    echo Found existing .claude.json configuration
) else (
    echo Creating default .claude.json configuration...
    echo {> "%USERPROFILE%\.claude.json"
    echo   "organization_id": null,>> "%USERPROFILE%\.claude.json"
    echo   "settings": {}>> "%USERPROFILE%\.claude.json"
    echo }>> "%USERPROFILE%\.claude.json"
)

:menu
echo.
echo Select your API provider:
echo 1. Anthropic (Official)
echo 2. GLM-4.7 (ZhiPu)
echo 3. MiniMax-M2.1
echo 4. Kimi-2.5 (Moonshot)
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto anthropic
if "%choice%"=="2" goto zhipu
if "%choice%"=="3" goto minimax
if "%choice%"=="4" goto kimi
if "%choice%"=="5" goto exit
goto menu

:anthropic
echo.
echo Configuring for Anthropic (Official) ...
set ANTHROPIC_API_KEY=123
set ANTHROPIC_BASE_URL=https://api.anthropic.com
set CLAUDE_MODEL=claude-4-5-sonnet-20251001
echo Configuration set for Anthropic.
echo Run Claude with: claude --env ANTHROPIC_API_KEY ANTHROPIC_BASE_URL CLAUDE_MODEL
goto launch

:zhipu
echo.
echo Configuring for GLM AI (ZhiPu) ...
echo Note: Zhipu AI provides Claude-compatible API through their platform
set ANTHROPIC_API_KEY=123
set ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/paas/v4
set CLAUDE_MODEL=GLM-4.7
echo Configuration set for Zhipu AI.
echo Run Claude with: claude --env ANTHROPIC_API_KEY ANTHROPIC_BASE_URL CLAUDE_MODEL
goto launch

:minimax
echo.
echo Configuring for MiniMax ...
echo Note: MiniMax provides Claude-compatible API endpoints
set ANTHROPIC_API_KEY=123
set ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
set CLAUDE_MODEL=MiniMax-M2.1
echo Configuration set for MiniMax.
echo Run Claude with: claude --env ANTHROPIC_API_KEY ANTHROPIC_BASE_URL CLAUDE_MODEL
goto launch

:kimi
echo.
echo Configuring for Kimi (Moonshot) ...
echo Note: Kimi provides Claude-compatible API through Moonshot platform
set ANTHROPIC_API_KEY=123
set ANTHROPIC_BASE_URL=https://api.moonshot.cn/v1
set CLAUDE_MODEL=Kimi-2.5
echo Configuration set for Kimi.
echo Run Claude with: claude --env ANTHROPIC_API_KEY ANTHROPIC_BASE_URL CLAUDE_MODEL
goto launch

:launch
echo.
echo IMPORTANT: Replace "123" with your actual API key in the environment variables
echo Current configuration:
echo ANTHROPIC_API_KEY=123
echo ANTHROPIC_BASE_URL=%ANTHROPIC_BASE_URL%
echo CLAUDE_MODEL=%CLAUDE_MODEL%
echo.
pause
goto exit

:exit
echo Exiting configuration script.
endlocal
```

## macOS 版本 (setup_claude.sh):

```bash
#!/bin/bash

echo "Claude Code Configuration Script for macOS"
echo "=========================================="

CONFIG_FILE="$HOME/.claude.json"

if [ -f "$CONFIG_FILE" ]; then
    echo "Found existing .claude.json configuration"
else
    echo "Creating default .claude.json configuration..."
    cat > "$CONFIG_FILE" << EOF
{
  "organization_id": null,
  "settings": {}
}
EOF
fi

show_menu() {
    echo ""
    echo "Select your API provider:"
    echo "1. Anthropic (Official)"
    echo "2. GLM-4.7 (ZhiPu)"
    echo "3. MiniMax-M2.1"
    echo "4. Kimi-2.5 (Moonshot)"
    echo "5. Exit"
    echo ""
}

configure_provider() {
    local provider_name=$1
    local base_url=$2
    local model=$3
    
    echo ""
    echo "Configuring for $provider_name..."
    export ANTHROPIC_API_KEY="123"
    export ANTHROPIC_BASE_URL="$base_url"
    export CLAUDE_MODEL="$model"
    
    echo "Configuration set for $provider_name."
    echo "Run Claude with: claude --env ANTHROPIC_API_KEY ANTHROPIC_BASE_URL CLAUDE_MODEL"
    echo ""
    echo "IMPORTANT: Replace \"123\" with your actual API key"
    echo "Current configuration:"
    echo "ANTHROPIC_API_KEY=123"
    echo "ANTHROPIC_BASE_URL=$ANTHROPIC_BASE_URL"
    echo "CLAUDE_MODEL=$CLAUDE_MODEL"
    echo ""
    echo "To use these settings in your current session, run:"
    echo "export ANTHROPIC_API_KEY=\"YOUR_ACTUAL_API_KEY\""
    echo "export ANTHROPIC_BASE_URL=\"$base_url\""
    echo "export CLAUDE_MODEL=\"$model\""
    echo ""
}

while true; do
    show_menu
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            configure_provider "Anthropic (Official)" "https://api.anthropic.com" "claude-4-5-sonnet-20251001"
            ;;
        2)
            configure_provider "GLM-4.7 (ZhiPu)" "https://open.bigmodel.cn/api/paas/v4" "GLM-4.7"
            ;;
        3)
            configure_provider "MiniMax-M2.1" "https://api.minimaxi.com/anthropic" "MiniMax-M2.1"
            ;;
        4)
            configure_provider "Kimi-2.5 (Moonshot)" "https://api.moonshot.cn/v1" "Kimi-2.5"
            ;;
        5)
            echo "Exiting configuration script."
            exit 0
            ;;
        *)
            echo "Invalid choice. Please enter a number between 1 and 5."
            ;;
    esac
    
    read -p "Press Enter to continue..."
done
```

## 各供应商配置说明：

1. **Anthropic (官方)**：

   - Base URL: `https://api.anthropic.com`
   - 使用官方 Claude API

2. **GLM (ZhiPu)**：

   - Base URL: `https://open.bigmodel.cn/api/paas/v4`
   - 智谱提供 Claude 兼容的 API 接口

3. **MiniMax**：

   - Base URL: `https://api.minimaxi.com/anthropic`
   - MiniMax提供 Claude 兼容的 API 端点

4. **Kimi (Moonshot)**：

   - Base URL: `https://api.moonshot.cn/v1`
   - 月之暗面提供的 Claude 兼容API

**注意**：所有配置中的API Key都设置为"123"，用户需要替换为自己的实际API密钥。模型名称默认使用各供应商支持的最新Claude模型。
