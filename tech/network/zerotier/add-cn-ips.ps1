
# 以管理员身份运行 PowerShell

# 你本地上网的网卡 IP
$eth0_IP = "192.168.5.1"

# 你的 ZeroTier IP
$zt_IP = "10.144.172.100"

# 读取 CIDR 列表
$cidrs = Get-Content ".\simple_cn_ips.txt"

foreach ($cidr in $cidrs) {
    # 将 CIDR 转换为目标网络和子网掩码
    $parts = $cidr -split "/"
    $network = $parts[0]
    $maskLen = [int]$parts[1]
    
    # 计算子网掩码
    $maskBinary = "1" * $maskLen + "0" * (32 - $maskLen)
    $maskBytes = for ($i = 0; $i -lt 4; $i++) {
        [Convert]::ToByte([Convert]::ToInt32($maskBinary.Substring($i*8, 8), 2))
    }
    $mask = [string]::Join(".", $maskBytes)
    
    # 添加永久路由
    route -p add $network mask $mask $eth0_IP metric 5
    Write-Host "已添加: $network/$maskLen"
}

# 添加默认路由
route -p add 0.0.0.0/0 mask $mask $zt_IP metric 5
