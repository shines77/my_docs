# Fail2ban 阻拦信息查看

## 查看方法

要知道 Fail2ban 在什么时候阻挡了哪些 IP，最直接的方法是**查看它自己的日志文件**，其次是**查看防火墙规则**。这两种方式分别对应“历史记录”和“当前状态”。

这里有 4 种最实用的查看方法，按推荐度排序：

### 方法 1：查看 Fail2ban 日志（最推荐，信息最全）

Fail2ban 会将每一次封禁、解封和重试操作都记录在 `/var/log/fail2ban.log` 中。这里能看到**精确到秒的时间**、**被阻挡的 IP** 以及**触发的服务（如 sshd）**。

-   **查看完整日志**：

    ```bash
    sudo cat /var/log/fail2ban.log
    ```

-   **只看“封禁”和“解封”的操作（过滤掉杂讯）**：

    ```bash
    sudo grep -E "Ban|Unban" /var/log/fail2ban.log
    ```

    输出示例：

    ```text
    2026-09-04 08:23:15,123 fail2ban.actions [12345]: NOTICE  [sshd] Ban 192.168.1.100
    2026-09-04 08:33:15,456 fail2ban.actions [12345]: NOTICE  [sshd] Unban 192.168.1.100
    ```

-   **实时监控（动态滚动显示新封禁）**：

    ```bash
    sudo tail -f /var/log/fail2ban.log | grep "Ban"
    ```

---

### 方法 2：查看防火墙当前的“黑名单”（看当下正被封禁的 IP）

如果你使用的是 `iptables` 或 `ufw`，Fail2ban 实际上是把 IP 扔进了防火墙的规则链中。查看防火墙就能知道**当前这一刻**哪些 IP 还在被关“小黑屋”。

-   **如果你使用的是 UFW（Ubuntu 默认推荐）**：

    ```bash
    sudo ufw status
    ```

    注意看底部的 `Actions` 部分，或者直接看 `iptables` 规则。

-   **如果你使用的是 iptables（更底层）**：
    Fail2ban 默认会创建一个名为 `f2b-sshd`（或其他服务名）的链，查看它即可：

    ```bash
    sudo iptables -L f2b-sshd -n --line-numbers
    ```

    （`-n` 表示不解析域名，显示纯数字 IP）

-   **一招查看所有被 Fail2ban 封禁的 IP（通用）**：

    ```bash
    sudo iptables -L | grep -E "DROP|REJECT" | grep -v "0.0.0.0"
    ```

---

### 方法 3：使用 fail2ban-client 查看特定服务的“囚犯列表”

如果你想看 SSH 服务当前封禁了哪些 IP，可以用客户端命令。不过这个方法**只显示当前正在被封禁的 IP**，不显示历史解封记录。

```bash
sudo fail2ban-client status sshd
```

在输出的 `Banned IP list:` 后面就是当前被封的 IP 列表。

---

### 方法 4：查看系统认证日志（查看“犯罪证据”）

如果你想了解这些 IP **在被封之前**具体做了什么（比如尝试了哪些用户名），可以查看 SSH 的认证日志。这能帮你判断是暴力破解还是误封。

```bash
sudo grep "Failed password" /var/log/auth.log | tail -20
```

或者把 IP 带上去查：

```bash
sudo grep "Failed password" /var/log/auth.log | grep "192.168.1.100"
```

---

## 进阶技巧：写个脚本一次性看清“谁在挨打”

如果你觉得上面几条命令太分散，可以用下面这个组合命令，一次性列出**当前所有正在被封禁的 IP 及其封禁剩余时间**（需要 `fail2ban-client` 配合）：

```bash
sudo fail2ban-client banned
```

*（注：Ubuntu 24.04 的较新版本支持 `banned` 命令，如果不支持，请使用 `fail2ban-client status` 查看）*

系统自带的 `lastb` 也可以看到哪些 IP 登陆过 SSH，例如：

```bash
lastb

lisi     ssh:notty    180.76.236.71    Fri Sep  4 15:14 - 15:14  (00:00)
lisi     ssh:notty    180.76.236.71    Fri Sep  4 15:14 - 15:14  (00:00)
student  ssh:notty    139.59.80.46     Fri Sep  4 15:11 - 15:11  (00:00)
student  ssh:notty    139.59.80.46     Fri Sep  4 15:11 - 15:11  (00:00)
jenkins  ssh:notty    152.32.218.149   Fri Sep  4 15:10 - 15:10  (00:00)
jenkins  ssh:notty    152.32.218.149   Fri Sep  4 15:10 - 15:10  (00:00)
root     ssh:notty    36.95.15.107     Fri Sep  4 15:08 - 15:08  (00:00)
root     ssh:notty    219.151.150.30   Fri Sep  4 14:57 - 14:57  (00:00)
root     ssh:notty    217.60.255.130   Fri Sep  4 14:38 - 14:38  (00:00)
root     ssh:notty    161.248.37.160   Fri Sep  4 14:33 - 14:33  (00:00)
administ ssh:notty    195.178.110.231  Fri Sep  4 14:30 - 14:30  (00:00)
administ ssh:notty    195.178.110.231  Fri Sep  4 14:30 - 14:30  (00:00)

btmp begins Sun Dec 26 22:43:12 1915
```

---

## 总结建议

-   **平时巡检**：用 `sudo grep "Ban" /var/log/fail2ban.log` 查看历史封禁时间点。
-   **发现连不上，怀疑自己被误封**：用 `sudo iptables -L -n | grep <你的IP>` 检查防火墙。
-   **想要可视化**：如果你觉得敲命令麻烦，可以安装 `fail2ban-web` 等图形化面板，不过对于服务器运维，直接用 `tail -f /var/log/fail2ban.log` 是最直观高效的。
