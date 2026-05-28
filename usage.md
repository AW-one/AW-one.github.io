# Usage

## 1. 编辑本地文件

1. 在本地仓库中打开需要修改的文档，例如 `index.md`、`README.md` 或项目目录下的其他文件。
2. 使用文本编辑器完成编辑并保存文件。

## 2. 查看修改内容

在仓库根目录打开终端，运行以下命令：

```bash
git status
```

这会显示已修改、已新增或已删除的文件。

## 3. 添加修改到暂存区

将要提交的文件加入暂存区，例如：

```bash
git add .
```

或只添加特定文件：

```bash
git add index.md README.md usage.md
```

## 4. 提交更改

为本次修改写一个清晰的提交信息：

```bash
git commit -m "更新个人主页内容并添加 usage 文档"
```

## 5. 推送到 GitHub

将本地提交推送到远程仓库：

```bash
git push origin main
```

如果你的默认分支不是 `main`，请把 `main` 替换为实际分支名，比如 `master`。

## 6. 在 GitHub 上确认

1. 打开浏览器，访问你的 GitHub 仓库页面。
2. 检查你刚刚提交的文件是否已出现在仓库中。
3. 如果你的网站使用 GitHub Pages，确认页面已经更新。

---

以上步骤适用于将编辑好的文件上传到 GitHub。