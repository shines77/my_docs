# MyBaties 中关于自增 id

在 MyBatis 中，获取自动增长（Auto Increment）的 ID 值取决于数据库的类型以及 MyBatis 的配置。以下是常见的几种方法：

---

### 1. **使用 `useGeneratedKeys` 和 `keyProperty`**

   这是 MyBatis 推荐的方式，适用于支持自动生成主键的数据库（如 MySQL、PostgreSQL 等）。

   - 在 `insert` 语句中，设置 `useGeneratedKeys="true"`，并指定 `keyProperty` 为实体类中对应的属性。
   - 插入数据后，自动生成的 ID 会直接赋值到实体类的属性中。

   **示例：**

   ```xml
   <insert id="insertUser" useGeneratedKeys="true" keyProperty="id">
       INSERT INTO user (name, age) VALUES (#{name}, #{age})
   </insert>
   ```

   **Java 代码：**

   ```java
   User user = new User();
   user.setName("John");
   user.setAge(25);
   userMapper.insertUser(user); // 插入数据
   System.out.println("生成的 ID: " + user.getId()); // 获取自动生成的 ID
   ```

---

### 2. **使用 `selectKey`**

   如果数据库不支持 `useGeneratedKeys`，或者需要更复杂的逻辑（如获取序列值），可以使用 `<selectKey>`。

   - `<selectKey>` 可以在插入数据之前或之后执行一个 SQL 语句，将结果赋值给实体类的属性。

   **示例（MySQL）：**

   ```xml
   <insert id="insertUser">
       INSERT INTO user (name, age) VALUES (#{name}, #{age})
       <selectKey keyProperty="id" resultType="int" order="AFTER">
           SELECT LAST_INSERT_ID()
       </selectKey>
   </insert>
   ```

   **示例（Oracle）：**

   ```xml
   <insert id="insertUser">
       <selectKey keyProperty="id" resultType="int" order="BEFORE">
           SELECT user_seq.NEXTVAL FROM dual
       </selectKey>
       INSERT INTO user (id, name, age) VALUES (#{id}, #{name}, #{age})
   </insert>
   ```

   **Java 代码：**

   ```java
   User user = new User();
   user.setName("John");
   user.setAge(25);
   userMapper.insertUser(user); // 插入数据
   System.out.println("生成的 ID: " + user.getId()); // 获取自动生成的 ID
   ```

---

### 3. **数据库特定的函数**

   某些数据库提供了特定的函数来获取自动生成的 ID，例如：
   - MySQL：`LAST_INSERT_ID()`
   - PostgreSQL：`RETURNING id`
   - SQL Server：`SCOPE_IDENTITY()`

   **示例（PostgreSQL）：**

   ```xml
   <insert id="insertUser">
       INSERT INTO user (name, age) VALUES (#{name}, #{age})
       RETURNING id
   </insert>
   ```

   **Java 代码：**

   ```java
   User user = new User();
   user.setName("John");
   user.setAge(25);
   int generatedId = userMapper.insertUser(user); // 插入数据并返回 ID
   System.out.println("生成的 ID: " + generatedId);
   ```

---

### 4. **注意事项**

   - **数据库支持**：不同数据库对自动生成 ID 的支持不同，请根据数据库类型选择合适的方案。
   - **事务**：确保插入操作在事务中执行，以避免并发问题。
   - **多行插入**：如果插入多行数据，`useGeneratedKeys` 和 `selectKey` 可能无法正确获取所有生成的 ID。

---

### 总结

- 对于大多数场景，推荐使用 `useGeneratedKeys` 和 `keyProperty`，简单且高效。
- 如果需要更复杂的逻辑，可以使用 `<selectKey>`。
- 根据数据库类型选择合适的方案，并确保在事务中操作。

希望这些方法能帮助你解决问题！如果还有其他疑问，请随时提问。
