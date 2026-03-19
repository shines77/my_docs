# 如何优化低配置服务器上的 PostgreSQL 性能？

## 前言

（来自 DeepSeek-V3.2 ）

在低配置服务器（1核CPU、1-2GB内存）上优化 PostgreSQL，核心思路是 **严格控制内存使用、优化检查点行为、调整查询规划器**。与MySQL不同，PostgreSQL的体系结构依赖**操作系统缓存**，因此内存分配策略有所不同。

以下是针对低配置环境的详细优化方案：

## 1. 内存配置（最关键！）

PostgreSQL 有两个重要的内存区域：**共享内存** 和 **每个连接私有的内存** 。

### 共享缓冲区 （Shared Buffers）

- **配置项：** `shared_buffers`
- **建议值：** 可用内存的 **15%-25%**
  - *例如：* 1GB 内存 → 设置为 **256MB**
  - *例如：* 512MB 内存 → 设置为 **128MB**
- **原理：** PG 依赖操作系统的缓存，所以 `shared_buffers` 不需要像 MySQL 的 InnoDB Buffer Pool 那样设置得很大。设太大会导致内存双重缓存和内部管理开销增加。

### 工作内存 （Work Mem）

- **配置项：** `work_mem`
- **建议值：** **4MB - 8MB**（默认 4MB）
  - *注意：* 这是 **每个连接、每个排序/哈希操作** 可以使用的内存。如果有 20 个连接同时做排序，可能消耗 20 × 8MB = 160MB。
  - **危险信号：** 不要设得过高，否则内存会瞬间被耗尽，导致 OOM（内存不足） killer 杀掉 PostgreSQL 进程。

### 维护工作内存 （Maintenance Work Mem）

- **配置项：** `maintenance_work_mem`
- **建议值：** **64MB - 128MB**（默认 64MB）
- **用途：** 用于 VACUUM、CREATE INDEX 等维护操作。适当增大可以让日常清理更快完成。

### 有效缓存大小 （Effective Cache Size）

- **配置项：** `effective_cache_size`
- **建议值：** 可用内存的 **50%-75%**
  - *例如：* 1GB 内存 → 设置为 **512MB - 768MB**
- **原理：** 这不是分配内存，而是告诉查询规划器："操作系统能缓存多少文件？"。规划器根据这个值更倾向于选择索引扫描还是顺序扫描。

## 2. 连接配置

低配置服务器无法处理大量并发连接。

### 最大连接数

- **配置项：** `max_connections`
- **建议值：** **20 - 50**（默认通常是 100）
- **配合策略：** 如果应用需要更多连接，建议在应用层使用 **连接池**（如 PgBouncer），而不是增加 `max_connections`。每个连接即使空闲，也会消耗内存。

## 3. 检查点与 WAL（预写式日志）优化

这是低配置机器磁盘 I/O 压力的主要来源。频繁的检查点会导致所有进程的磁盘写入风暴。

### 调整检查点间隔

- **配置项：** `checkpoint_timeout`
- **建议值：** **15min**（默认 5min）
- **配置项：** `max_wal_size`
- **建议值：** **2GB - 4GB**（默认 1GB）
- **原理：** 允许检查点间隔更长，让后台写进程更平稳地刷脏页，避免瞬间 I/O 高峰。

### 减少 WAL 日志刷新频率

- **配置项：** `wal_buffers`
- **建议值：** **4MB - 8MB**（默认 -1 自动，通常偏小）
- **配置项：** `synchronous_commit`
- **建议值：** 如果数据不是绝对关键（如日志分析、个人应用），可以设为 **off**
  - *注意：* 设为 off 可能导致操作系统崩溃时丢失少量最近的事务（约 0.5 秒的数据），但能大幅提升写入性能。

## 4. 查询规划器调整

低配置机器上，顺序扫描可能比复杂的索引扫描更慢，但有时规划器会误判。

### 随机页成本

- **配置项：** `random_page_cost`
- **建议值：** **2.0**（如果使用机械硬盘） / **1.1**（如果使用 SSD）
- **原理：** 默认是 4.0（针对老式机械硬盘），这对现代硬盘过于悲观。降低这个值可以让规划器更愿意使用索引。

### 连接查询优化

- **配置项：** `join_collapse_limit`
- **建议值：** 保持默认 **8**
- **说明：** 如果查询涉及大量表连接，低配置服务器可能耗尽内存。检查慢查询日志，简化复杂连接。

## 5. 自动清理 （Autovacuum）—— 必须保持运行！

PostgreSQL 依赖 VACUUM 来清理死元组。如果 autovacuum 跟不上，会导致表膨胀，索引变大，性能急剧下降。

- **配置项：** `autovacuum = on`（必须开启）
- **配置项：** `autovacuum_vacuum_scale_factor`
- **建议值：** **0.01**（默认 0.2，对于小表来说太高了）
- **配置项：** `autovacuum_work_mem`
- **建议值：** **64MB**
- **注意：** 在低配置机器上，要避免 autovacuum 太频繁，但也要避免它来不及清理。建议监控 `pg_stat_user_tables` 中 `n_dead_tup` 的数量。

## 6. 操作系统级别优化

### 交换分区 （Swap）

- **建议：** 预留少量 Swap（如 1GB），但将 `vm.swappiness` 调低
  ```bash
  # 临时设置
  sysctl vm.swappiness=10
  
  # 永久设置（/etc/sysctl.conf）
  vm.swappiness = 10
  ```
- **原理：** 避免内核过早地将 PostgreSQL 的共享内存交换到磁盘。

### 大页 （Huge Pages）

- **建议：** **关闭**（在低配置机器上）
  - 在 `/etc/sysctl.conf` 中设置 `vm.nr_hugepages = 0`
  - 在 PostgreSQL 中设置 `huge_pages = off`
- **原理：** 小内存机器不需要大页，反而会造成浪费。

## 7. 应用层配合

与 MySQL 类似，但有一些 PG 特有的技巧：

### 索引类型

- 如果查询主要是等值查询（`=`），使用默认的 B-tree。
- 如果有全文搜索需求，使用 GIN 索引。
- **注意：** 低配置机器上，**不要滥用索引**。每个索引都会增加写入和 VACUUM 的开销。

### 避免长事务

- 长事务会阻止 autovacuum 回收死元组，导致表膨胀。
- 监控：`SELECT * FROM pg_stat_activity WHERE state = 'idle in transaction';`

### 使用 PREPARE 语句

- 对于重复执行的查询，使用 Prepared Statements 可以减少 SQL 解析和计划生成的开销。

## 一个示例配置（适用于 1GB 内存服务器，PostgreSQL 12+）

编辑 `postgresql.conf`：

```conf
# 连接设置
max_connections = 40
superuser_reserved_connections = 3

# 内存设置
shared_buffers = 256MB          # 1GB * 0.25
effective_cache_size = 768MB     # 1GB * 0.75
work_mem = 8MB
maintenance_work_mem = 128MB
huge_pages = off

# WAL/检查点设置
wal_buffers = 4MB
synchronous_commit = off        # 如果不需要绝对持久化
checkpoint_timeout = 15min
max_wal_size = 2GB
min_wal_size = 1GB

# 查询规划器
random_page_cost = 1.1          # 如果使用 SSD
effective_io_concurrency = 2     # 如果使用 SSD

# 自动清理
autovacuum = on
autovacuum_vacuum_scale_factor = 0.01
autovacuum_analyze_scale_factor = 0.05
autovacuum_vacuum_cost_limit = 200  # 降低清理对 I/O 的压力

# 日志
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h'
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0
log_autovacuum_min_duration = 0

# 慢查询日志
log_min_duration_statement = 2000  # 记录超过 2 秒的查询
```

## 监控和诊断工具

在低配置机器上，你需要知道瓶颈在哪里：

1.  **查看当前活动：**

    ```sql
    SELECT pid, state, query, age(now(), query_start) FROM pg_stat_activity;
    ```

2.  **查看表膨胀情况：**

    ```sql
    SELECT schemaname, tablename, n_dead_tup, n_live_tup, 
           round(n_dead_tup::numeric / n_live_tup * 100, 2) as dead_pct
    FROM pg_stat_user_tables 
    WHERE n_live_tup > 0 
    ORDER BY dead_pct DESC;
    ```

3.  **查看索引使用情况（找出无效索引）：**

    ```sql
    SELECT schemaname, tablename, indexname, idx_scan 
    FROM pg_stat_user_indexes 
    ORDER BY idx_scan ASC;
    ```

## 总结：低配置 PG 优化的黄金法则

1.  **内存控制：** `shared_buffers` 给 25%，`effective_cache_size` 给 75%，告诉系统："内存很小，但 OS 缓存很重要"。
2.  **连接池化：** 尽量用 PgBouncer 把并发连接数控制在 20-30 以内。
3.  **减少写入放大：** `synchronous_commit = off` + 延长 `checkpoint_timeout`，让 I/O 更平滑。
4.  **保持整洁：** 确保 autovacuum 能及时清理，否则表膨胀会拖垮一切。
5.  **索引优化：** 定期检查 `pg_stat_user_indexes`，删除从未使用的索引，减少写入负担。

**最立竿见影的操作：** 

- 先设置 `shared_buffers = 256MB` 
- 设置 `synchronous_commit = off` 
- 重启服务，观察磁盘 I/O 压力是否下降
