# PostgreSQL配置优化中有哪些关键参数需要关注？

## 计划

计划：

1. **分析 PostgreSQL 性能关键点**：确定影响查询性能、内存使用、I/O 效率及并发处理的核心参数。
2. **分类关键参数**：将参数按功能分组，如内存管理、并发控制、日志与检查点、自动清理等。
3. **列出核心参数并简述作用**：选取对性能影响最大且常需调优的参数，说明其用途和优化方向。

## 关键参数

关键参数包括：

- **内存相关**：

  - `shared_buffers`：数据库共享内存缓冲区大小，建议设为物理内存的 25%~40% 。
  - `work_mem`：单个查询操作（如排序、哈希）可使用的内存，影响排序和连接性能，通常设为 4MB~16MB 。  
  - `maintenance_work_mem`：维护操作（如 VACUUM、索引创建）可用内存，建议设为较大值（如几百 MB）。
  - `effective_cache_size`：告诉 PostgreSQL 的查询优化器，操作系统和 PostgreSQL 自身的缓存一共有多少内存可用。一般情况下，可以将其设置为总 RAM 的 50%-75%。

- **并发与连接**：

  - `max_connections`：最大客户端连接数，过多会导致内存压力，建议配合连接池使用。
  - `effective_cache_size`：优化器假设的操作系统和 PostgreSQL 可用磁盘缓存大小，通常设为物理内存的 50%~75% 。

- **日志与检查点**：

  - `checkpoint_timeout` 和 `checkpoint_completion_target`：控制检查点频率和写入平滑度，避免 I/O 突增。
  - `wal_buffers`：WAL（预写式日志）缓冲区大小，默认通常足够，高负载可适当增加。

- **自动清理（Autovacuum）**：

  - `autovacuum_vacuum_cost_limit`、`autovacuum_naptime` 等：控制自动清理行为，避免表膨胀，根据负载调整。

这些参数应根据硬件配置、业务负载和监控数据逐步调优。
