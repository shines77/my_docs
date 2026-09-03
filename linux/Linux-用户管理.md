# Ubuntu 24.04 用户与用户组管理完整指南

本指南涵盖了在 Ubuntu 24.04 系统中管理用户和用户组的常用命令与操作，包括查询、增删改、密码管理及相关配置文件等。

## 1. 用户与组的基本概念

- **用户**：系统中的一个账户，用于登录和执行操作。
- **用户组**：一组用户的集合，用于统一管理权限。
- **主组（primary group）**：用户创建文件时默认所属的组。
- **附加组（supplementary groups）**：用户可以属于多个附加组，以获得额外权限。

主要配置文件：

- `/etc/passwd`：用户账户信息（用户名、UID、GID、家目录、shell等）。
- `/etc/shadow`：用户密码的加密信息及密码策略。
- `/etc/group`：用户组信息。
- `/etc/gshadow`：用户组密码及组管理员信息。
- `/etc/login.defs`：用户账户的默认配置（如UID范围、密码过期策略等）。
- `/etc/default/useradd`：`useradd` 命令的默认设置。
- `/etc/skel`：新用户家目录的模板目录。

---

## 2. 查询用户信息

### 2.1 查看当前用户

```bash
whoami          # 显示当前用户名
id              # 显示当前用户的 UID、GID 及所属组
id username     # 显示指定用户的 UID、GID 及所属组
```

### 2.2 查看登录用户

```bash
who             # 显示当前登录的用户
w               # 显示登录用户及其正在执行的任务
users           # 简单列出登录用户
```

### 2.3 查看用户详细信息

```bash
finger username           # 显示用户详细信息（需安装 finger：sudo apt install finger）
getent passwd username    # 从 NSS 数据库获取用户信息
grep '^username:' /etc/passwd   # 直接查看 passwd 文件
lslogins username         # 显示用户登录信息（属于 util-linux 包）
```

### 2.4 查看所有用户

```bash
cat /etc/passwd           # 列出所有用户（包括系统用户）
getent passwd             # 类似 cat /etc/passwd
compgen -u                # 列出所有本地用户（bash 内置）
```

### 2.5 查看用户登录历史

```bash
last              # 显示所有用户的登录历史
last username     # 显示指定用户的登录历史
lastb             # 显示失败登录尝试（需要 root 权限）
```

---

## 3. 查询用户组信息

### 3.1 查看当前用户的组

```bash
groups                  # 显示当前用户所属的所有组
id -Gn                  # 同上，仅显示组名
id -G                   # 显示组 ID
```

### 3.2 查看指定用户的组

```bash
groups username         # 显示指定用户所属的组
id username             # 显示用户的 UID、GID 及所有组
```

### 3.3 查看所有组

```bash
cat /etc/group          # 列出所有组
getent group            # 类似 cat /etc/group
compgen -g              # 列出所有本地组（bash 内置）
```

### 3.4 查看组详细信息

```bash
getent group groupname          # 获取组信息
grep '^groupname:' /etc/group   # 直接查看组文件
```

---

## 4. 添加用户

Ubuntu 提供两种方式：`useradd`（低级工具）和 `adduser`（交互式友好脚本，Debian/Ubuntu 推荐）。

### 4.1 使用 `adduser`（推荐）

```bash
sudo adduser username
```

该命令会交互式地设置密码、用户信息，并自动创建家目录、复制 `/etc/skel` 内容、创建用户私有组。

常用选项：

- `--home DIR`：指定家目录。
- `--shell SHELL`：指定登录 shell。
- `--uid UID`：指定 UID。
- `--gid GROUP`：指定主组。
- `--ingroup GROUP`：将用户添加到已存在的组作为主组。
- `--disabled-password`：创建用户但不设置密码（账户锁定）。
- `--gecos GECOS`：设置用户全名等信息。

示例：

```bash
sudo adduser --home /opt/user1 --shell /bin/bash --uid 1500 --ingroup developers user1
```

### 4.2 使用 `useradd`

```bash
sudo useradd [options] username
```

常用选项：

- `-m`：创建家目录（默认可能不创建）。
- `-d DIR`：指定家目录。
- `-s SHELL`：指定登录 shell。
- `-u UID`：指定 UID。
- `-g GROUP`：指定主组（组名或 GID）。
- `-G GROUPS`：指定附加组列表（逗号分隔）。
- `-c COMMENT`：添加用户描述（GECOS）。
- `-e DATE`：账户过期日期（YYYY-MM-DD）。
- `-f DAYS`：密码过期后多少天禁用账户。
- `-k DIR`：指定骨架目录（默认 `/etc/skel`）。
- `-p PASSWORD`：指定加密后的密码（不推荐，使用 `passwd` 设置）。
- `-r`：创建系统用户（UID 小于 1000）。

示例：

```bash
sudo useradd -m -s /bin/bash -G sudo,developers -c "John Doe" johndoe
sudo passwd johndoe   # 设置密码
```

### 4.3 批量创建用户

使用 `newusers` 命令可以从一个格式化的文件批量创建用户：

```bash
sudo newusers users.txt
```

文件格式与 `/etc/passwd` 相同（用户名:密码:UID:GID:GECOS:家目录:shell）。

---

## 5. 删除用户

### 5.1 使用 `deluser`（推荐）

```bash
sudo deluser username
```

常用选项：

- `--remove-home`：同时删除用户家目录和邮件目录。
- `--remove-all-files`：删除系统中属于该用户的所有文件。
- `--backup`：删除前备份用户文件。
- `--group`：删除一个组（与 `delgroup` 相同）。

示例：

```bash
sudo deluser --remove-home johndoe
```

### 5.2 使用 `userdel`

```bash
sudo userdel [options] username
```

常用选项：

- `-r`：删除用户家目录和邮件目录。
- `-f`：强制删除，即使用户已登录。

示例：

```bash
sudo userdel -r johndoe
```

---

## 6. 修改用户属性

### 6.1 使用 `usermod`

```bash
sudo usermod [options] username
```

常用选项：
- `-d DIR`：修改家目录（通常配合 `-m` 移动内容）。
- `-m`：将家目录内容移动到新位置（与 `-d` 一起使用）。
- `-s SHELL`：修改登录 shell。
- `-l NEWNAME`：修改用户名。
- `-u UID`：修改 UID。
- `-g GROUP`：修改主组。
- `-G GROUPS`：设置附加组（覆盖原有附加组）。
- `-aG GROUPS`：将用户追加到附加组（保留原附加组）。
- `-L`：锁定用户密码（账户禁用）。
- `-U`：解锁用户密码。
- `-e DATE`：设置账户过期日期。
- `-c COMMENT`：修改用户描述。

示例：

```bash
sudo usermod -aG sudo,docker johndoe   # 将用户添加到 sudo 和 docker 组
sudo usermod -l newname oldname        # 修改用户名
sudo usermod -d /new/home -m username  # 修改家目录并移动内容
```

### 6.2 修改用户 Shell

```bash
chsh -s /bin/zsh username    # 修改用户登录 shell
```

### 6.3 修改用户 GECOS 信息（全名、电话等）

```bash
sudo chfn username
```

---

## 7. 管理用户密码

### 7.1 设置或修改密码

```bash
sudo passwd username    # 为指定用户设置密码
passwd                  # 修改当前用户密码
```

### 7.2 锁定/解锁账户

```bash
sudo passwd -l username   # 锁定密码（账户无法用密码登录，但可用 SSH 密钥等）
sudo passwd -u username   # 解锁密码
sudo usermod -L username  # 锁定密码
sudo usermod -U username  # 解锁密码
```

### 7.3 设置密码策略（过期等）

`chage` 命令用于管理密码过期信息：
```bash
sudo chage [options] username
```

常用选项：
- `-l`：显示密码过期信息。
- `-M DAYS`：设置密码最大有效天数。
- `-m DAYS`：设置密码最小有效天数（两次修改的最小间隔）。
- `-W DAYS`：设置过期前警告天数。
- `-I DAYS`：设置密码过期后账户锁定的宽限天数。
- `-E DATE`：设置账户过期日期（YYYY-MM-DD）。
- `-d LASTDAY`：设置上次密码修改日期（从 1970-01-01 起的天数，0 表示下次登录强制改密）。

示例：

```bash
sudo chage -M 90 -W 7 -I 14 johndoe   # 密码 90 天过期，提前 7 天警告，过期后 14 天锁定
sudo chage -d 0 johndoe               # 强制用户下次登录修改密码
```

### 7.4 批量设置密码

使用 `chpasswd`：

```bash
echo "username:password" | sudo chpasswd
# 或从文件读取
sudo chpasswd < passwords.txt   # 文件每行 "username:password"
```

---

## 8. 添加用户组

### 8.1 使用 `addgroup`

```bash
sudo addgroup groupname
```

常用选项：
- `--gid GID`：指定 GID。
- `--system`：创建系统组（GID < 1000）。

### 8.2 使用 `groupadd`

```bash
sudo groupadd [options] groupname
```

常用选项：
- `-g GID`：指定 GID。
- `-r`：创建系统组。

示例：

```bash
sudo groupadd -g 2000 developers
```

---

## 9. 删除用户组

### 9.1 使用 `delgroup`

```bash
sudo delgroup groupname
```

### 9.2 使用 `groupdel`

```bash
sudo groupdel groupname
```

注意：不能删除仍作为用户主组的组，除非先删除或修改这些用户。

---

## 10. 修改用户组

### 10.1 使用 `groupmod`

```bash
sudo groupmod [options] groupname
```

常用选项：
- `-n NEWNAME`：修改组名。
- `-g GID`：修改 GID。

示例：

```bash
sudo groupmod -n newgroupname oldgroupname
sudo groupmod -g 3000 groupname
```

---

## 11. 管理组成员

### 11.1 将用户添加到组

- 使用 `usermod -aG`（推荐）：

```bash
sudo usermod -aG groupname username
```

- 使用 `gpasswd -a`：

```bash
sudo gpasswd -a username groupname
```

### 11.2 从组中移除用户

- 使用 `gpasswd -d`：

```bash
sudo gpasswd -d username groupname
```
- 或者手动编辑 `/etc/group`（不推荐）。

### 11.3 设置组管理员

`gpasswd` 可以指定组管理员，管理员可以管理组成员：

```bash
sudo gpasswd -A user1,user2 groupname   # 设置组管理员
```

### 11.4 设置组密码

```bash
sudo gpasswd groupname   # 交互式设置组密码
```

组密码用于 `newgrp` 命令切换到非主组时验证身份。

---

## 12. 切换用户和提权

### 12.1 `su` 切换用户

```bash
su - username       # 切换到指定用户并加载其环境
su username         # 切换用户但保留当前环境变量
su -                # 切换到 root（需 root 密码）
```

### 12.2 `sudo` 以管理员权限执行

```bash
sudo command           # 以 root 权限执行命令
sudo -u username command   # 以指定用户身份执行命令
sudo -i                # 进入 root 交互式 shell
sudo -s                # 以 root 运行 shell（保留当前环境）
```

### 12.3 配置 sudo 权限

- 将用户添加到 `sudo` 组（Ubuntu 默认允许 sudo 组成员）：

```bash
sudo usermod -aG sudo username
```

- 或编辑 `/etc/sudoers` 文件（使用 `visudo` 安全编辑）：

```bash
sudo visudo
```

添加类似行：

```
username ALL=(ALL:ALL) ALL
```

---

## 13. 检查用户和组配置

### 13.1 检查 `/etc/passwd` 和 `/etc/group` 文件完整性

```bash
sudo pwck    # 检查 /etc/passwd 和 /etc/shadow
sudo grpck   # 检查 /etc/group 和 /etc/gshadow
```

### 13.2 手动编辑用户/组文件

- `vipw`：安全编辑 `/etc/passwd`。
- `vigr`：安全编辑 `/etc/group`。

---

## 14. 默认配置与模板

### 14.1 查看/修改 `useradd` 默认设置

```bash
useradd -D                 # 显示默认值
sudo useradd -D -s /bin/bash   # 修改默认 shell
sudo useradd -D -b /home   # 修改默认家目录基础路径
```

配置文件：`/etc/default/useradd`。

### 14.2 修改全局默认值

编辑 `/etc/login.defs` 可以设置：

- `UID_MIN`、`UID_MAX`：普通用户 UID 范围。
- `PASS_MAX_DAYS`、`PASS_MIN_DAYS`、`PASS_WARN_AGE`：密码过期策略。
- `CREATE_HOME`：是否默认创建家目录（对 `useradd` 有效）。

### 14.3 家目录模板

`/etc/skel` 中的文件和目录会在使用 `adduser` 或 `useradd -m` 创建用户时复制到新用户的家目录。

---

## 15. 常见任务示例

### 15.1 创建新用户并加入 sudo 组

```bash
sudo adduser newuser
sudo usermod -aG sudo newuser
```

### 15.2 创建系统用户（无登录权限）

```bash
sudo useradd -r -s /usr/sbin/nologin serviceuser
```

### 15.3 强制用户下次登录修改密码

```bash
sudo chage -d 0 username
```

### 15.4 锁定用户账户

```bash
sudo usermod -L username   # 或 sudo passwd -l username
```

### 15.5 删除用户及其所有文件

```bash
sudo deluser --remove-all-files username
```

---

## 16. 总结

Ubuntu 24.04 的用户和组管理命令丰富且灵活。推荐使用 `adduser`、`deluser`、`addgroup`、`delgroup` 等友好命令进行日常操作，而底层工具如 `useradd`、`usermod`、`groupadd` 则适合脚本或高级需求。务必小心修改系统文件，使用 `vipw`、`vigr`、`visudo` 等安全编辑工具。

以上命令均需相应权限（通常为 root 或通过 sudo 执行）。
