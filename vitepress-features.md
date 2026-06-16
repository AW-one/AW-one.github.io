# VitePress 功能指南

VitePress 提供了丰富的功能来美化和增强个人主页。以下是主要功能及使用方法。

运行在线编辑时使用
```
npm run docs:dev
```
## 1. 首页特殊布局

使用 `layout: home` 和 `hero` 配置创建炫彩首页：

```yaml
---
layout: home

hero:
  name: "Kingzilv"
  text: "AI & Fashion Innovation"
  tagline: "AIGC, Computer Vision, Virtual Try-on
  image:
    src: /logo.svg
    alt: Logo
  actions:
    - theme: brand
      text: 查看项目
      link: /projects/
    - theme: alt
      text: GitHub
      link: https://github.com/AW-one

features:
  - title: 🤖 AI Generation
    details: 深度学习与生成模型研究
  - title: 👗 Fashion Tech
    details: 虚拟试穿与服装生成
  - title: 📚 Knowledge Base
    details: 技术笔记与学习记录
---
```

## 2. 自定义主题色彩

在 `config.mts` 中配置主题颜色：

```javascript
themeConfig: {
  // 深色模式下的颜色
  colorScheme: 'dark',
  
  // 自定义主题变量
  colors: {
    'c-brand': '#00d4ff',
    'c-brand-light': '#00d4ff99',
  }
}
```

## 3. 警告框与提示

在 markdown 中使用容器语法：

```markdown
::: warning 警告
这是一个警告信息
:::

::: danger 危险
这是危险操作提示
:::

::: tip 提示
这是有用的提示
:::

::: info 信息
这是信息提示
:::

::: details 点击展开
隐藏的详细内容
:::
```

## 4. 代码块增强

支持代码高亮、行号、代码聚焦：

````markdown
```python {1,3}
def hello():
    print("Hello")
    print("World")
```

```typescript [1,3-5]
interface User {
  id: number
  name: string
  email: string
  age: number
}
```
````

## 5. 自定义导航栏

添加下拉菜单和外部链接：

```javascript
nav: [
  { text: 'Home', link: '/' },
  {
    text: 'Projects',
    items: [
      { text: 'RedesignNet', link: '/projects/redesignnet' },
      { text: 'ConceptCloth', link: '/projects/conceptcloth' }
    ]
  },
  { text: 'External', link: 'https://example.com', target: '_blank' }
]
```

## 6. 页面 Frontmatter 元数据

在 markdown 文件顶部添加元数据：

```yaml
---
title: 自定义标题
description: 页面描述
layout: custom
image: /path/to/image.png
---
```

## 7. 编辑链接与最后更新

配置 git 仓库信息，自动显示编辑链接和更新时间：

```javascript
themeConfig: {
  editLink: {
    pattern: 'https://github.com/AW-one/AW-one.github.io/edit/main/:path',
    text: '在 GitHub 上编辑此页'
  },
  lastUpdated: {
    text: '最后更新',
    formatOptions: {
      dateStyle: 'short',
      timeStyle: 'short'
    }
  }
}
```

## 8. 自定义 CSS 样式

在 `.vitepress/theme` 目录中创建 `custom.css`：

```css
:root {
  --vp-c-brand: #00d4ff;
  --vp-c-brand-light: #00d4ff99;
  --vp-c-bg: #1a1a1a;
}

.custom-class {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 20px;
  border-radius: 8px;
}
```

## 9. 表格与图表

支持 Markdown 表格和 Mermaid 图表：

```markdown
| 功能 | 描述 |
|------|------|
| AIGC | AI生成内容 |
| CV | 计算机视觉 |

\`\`\`mermaid
graph LR
  A[开始] --> B{决策}
  B -->|是| C[结果1]
  B -->|否| D[结果2]
\`\`\`
```

## 10. 搜索功能

已在 config.mts 中配置本地搜索：

```javascript
search: {
  provider: 'local'
}
```

## 11. 页面布局选项

控制单个页面的显示元素：

```yaml
---
layout: page
aside: false  // 关闭侧边栏
editLink: false  // 关闭编辑链接
lastUpdated: false  // 关闭最后更新时间
---
```

## 12. 社交链接扩展

增加更多社交媒体链接：

```javascript
socialLinks: [
  { icon: 'github', link: 'https://github.com/AW-one' },
  { icon: 'twitter', link: 'https://twitter.com/yourhandle' },
  { icon: 'linkedin', link: 'https://linkedin.com/in/yourprofile' },
  { icon: 'mastodon', link: 'https://mastodon.social/@yourhandle' }
]
```

## 13. 自定义组件

在 `.vitepress/theme/components` 中创建 Vue 组件，直接在 markdown 中使用：

```markdown
<CustomCard 
  title="项目标题"
  description="项目描述"
  link="/projects/xxx"
/>
```

## 14. 多语言支持

配置多语言路由：

```javascript
locales: {
  root: {
    label: 'English',
    lang: 'en'
  },
  zh: {
    label: '中文',
    lang: 'zh',
    title: "个人主页"
  }
}
```

## 15. 部署与构建

构建静态网站：

```bash
npm run docs:build
```

部署到 GitHub Pages（在 `config.mts` 中设置）：

```javascript
export default defineConfig({
  base: '/',  // 或使用仓库名
  // ...
})
```

---

## 推荐改进方案

1. 使用 `layout: home` 创建动画首页
2. 添加自定义 CSS 实现渐变、阴影效果
3. 在项目卡片中使用图片和描述
4. 添加 Mermaid 图表展示技术栈
5. 配置编辑链接方便维护
6. 添加更多社交媒体链接

