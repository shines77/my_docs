
# Git 如何只拉取某个远程分支

## 1. 拉取远程分支

```bash
# --tags 只拉取远程分支的分支信息(仅分支列表),
# --all 拉取所有远程分支的所有数据

$ git fetch --tags -v --progress
```

1. 方法一

    ```bash
    # git clone $REMOTE_REPO --branch $BRANCH --single-branch $CREATE_FOLDER

    $ git clone https://github.com/google/protobuf.git --branch v2.4.1 --single-branch
    $ git clone https://github.com/google/protobuf.git -b v2.4.1 --single-branch

    $ git checkout -f -b v2.4.1     ## -f 命令表示强制覆盖已有文件
    ```

2. 方法二

    ```bash
    $ git init

    # git remote add -t $BRANCH -f origin $REMOTE_REPO
    $ git remote add -t v2.4.1 -f origin https://github.com/google/protobuf.git

    # git checkout -f -b $BRANCH
    $ git checkout -f -b v2.4.1     ## -f 命令表示强制覆盖已有文件
    ```
