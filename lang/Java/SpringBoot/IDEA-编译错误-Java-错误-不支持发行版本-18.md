# Idea IDE 编译错误

## 简介

Idea IDE 编译错误：

```bash
java: 错误: 不支持发行版本 18
```

## 解决办法

在使用 IntelliJ IDEA 编译 Java 项目时，出现错误 `java: 错误: 不支持发行版本 18`，通常是因为项目的 JDK 版本设置与实际使用的 JDK 版本不匹配。以下是解决此问题的步骤：

---

### 1. **检查项目的 JDK 版本**

确保项目使用的 JDK 版本与编译器要求的版本一致。

#### 方法 1：检查项目的 `Project SDK`

1. 打开 IntelliJ IDEA。
2. 进入 **File > Project Structure**（快捷键：`Ctrl + Alt + Shift + S`）。
3. 在左侧选择 **Project**。
4. 检查 **Project SDK** 是否设置为 JDK 18。如果没有，点击 **New...** 选择 JDK 18 的路径。
5. 确保 **Project language level** 设置为 **18**（或与 JDK 版本匹配的选项）。

#### 方法 2：检查模块的 `Module SDK`

1. 在 **Project Structure** 中，选择左侧的 **Modules**。
2. 选择你的模块，检查 **Module SDK** 是否设置为 JDK 18。
3. 确保 **Language level** 设置为 **18**。

---

### 2. **检查编译器设置**

确保编译器的 JDK 版本与项目一致。

#### 方法 1：检查 `Settings` 中的编译器设置
1. 打开 **File > Settings**（快捷键：`Ctrl + Alt + S`）。
2. 导航到 **Build, Execution, Deployment > Compiler > Java Compiler**。
3. 选择你的模块，确保 **Target bytecode version** 设置为 **18**。

#### 方法 2：检查 `pom.xml`（Maven 项目）

如果你使用的是 Maven 项目，确保 `pom.xml` 中指定了正确的 JDK 版本：

```xml
<properties>
    <maven.compiler.source>18</maven.compiler.source>
    <maven.compiler.target>18</maven.compiler.target>
</properties>
```

#### 方法 3：检查 `build.gradle`（Gradle 项目）
如果你使用的是 Gradle 项目，确保 `build.gradle` 中指定了正确的 JDK 版本：

```groovy
sourceCompatibility = '18'
targetCompatibility = '18'
```

---

### 3. **检查 IntelliJ IDEA 的全局 JDK 设置**

确保 IntelliJ IDEA 的全局 JDK 设置正确。

1. 打开 **File > Project Structure**。
2. 在左侧选择 **Platform Settings > SDKs**。
3. 确保已添加 JDK 18，并且它是默认的 SDK。

---

### 4. **检查环境变量**

确保系统环境变量中配置了正确的 JDK 版本。

1. 打开命令行（Windows：`cmd`，macOS/Linux：`Terminal`）。
2. 输入以下命令检查 JDK 版本：
   ```bash
   java -version
   ```
   输出应显示 JDK 18。如果不是，请安装 JDK 18 并配置环境变量。

---

### 5. **清理和重建项目**

在完成上述设置后，清理和重建项目：

1. 打开 **Build > Clean Project**。
2. 打开 **Build > Rebuild Project**。

---

### 6. **常见问题**

- **问题**：安装了多个 JDK 版本，IDE 选择了错误的版本。
  - **解决方法**：确保在 IDE 中明确指定 JDK 18。
- **问题**：项目依赖的库不支持 JDK 18。
  - **解决方法**：检查项目依赖的库是否兼容 JDK 18，必要时升级库。

---

### 7. **总结**

- 确保项目的 `Project SDK` 和 `Module SDK` 设置为 JDK 18。
- 确保编译器的 `Target bytecode version` 设置为 18。
- 确保 Maven 或 Gradle 配置文件中指定了正确的 JDK 版本。
- 清理和重建项目以确保设置生效。

如果你按照上述步骤操作后问题仍未解决，请提供更多详细信息（如项目类型、JDK 版本等），我可以进一步帮助你！
