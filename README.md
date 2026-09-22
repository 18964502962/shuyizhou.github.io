# 周舒怡的个人作品集

基于 [simplefolio](https://github.com/cobiwave/simplefolio) 模板构建的个人作品集网站。

---

## 🚀 一键部署到 GitHub Pages

### 第一步：创建 GitHub 仓库

1. 打开 https://github.com/new
2. **仓库名称必须为：`zy-shirley.github.io`**（把你自己的 GitHub 用户名替换 `zy-shirley`）
3. 选 **Public**（公开仓库）
4. 不要勾选 "Add a README" 或 "Add .gitignore"
5. 点击 **Create repository**

### 第二步：推送代码

仓库创建后会显示一个链接，类似：
```
https://github.com/你的用户名/zy-shirley.github.io.git
```

把下面这行命令里的 `你的用户名` 换成你的实际 GitHub 用户名，然后运行：

```bash
cd C:\Users\10029\WorkBuddy\2026-09-21-17-02-03\zy-shirley-portfolio

# 把远程仓库地址换成你自己的
git remote add origin https://github.com/你的用户名/zy-shirley.github.io.git
git branch -M main
git push -u origin main
```

### 第三步：开启 GitHub Pages（通常会自动开启）

1. 进入你的 GitHub 仓库页面
2. 点击 **Settings** → 左侧 **Pages**
3. 确保 **Source** 选的是 `Deploy from a branch`，分支选 `main`
4. 保存后，等待约 1-2 分钟

### 第四步：访问你的网站

浏览器打开：
```
https://zy-shirley.github.io
```

---

## 📝 自定义修改指南

### 修改个人信息
编辑 `index.html`，搜索以下关键词替换：
- `周舒怡` → 你的英文名/中文名
- `1002968845@qq.com` → 你的邮箱
- `zsyymm2962` → 你的微信号
- `18964502962` → 你的手机号

### 添加头像图片
1. 把你的照片放在 `assets/images/` 目录
2. 在 `index.html` 中找到 `<div class="avatar-placeholder">` 替换为：
```html
<img src="assets/images/avatar.jpg" alt="Profile" class="avatar-img">
```

### 添加项目截图
每个项目卡片里有 `.project-placeholder` 元素，替换为：
```html
<img src="assets/images/project-screenshot.png" alt="Project Screenshot" class="project-img">
```

### 修改配色
编辑 `assets/css/style.css` 顶部的 CSS 变量：
```css
:root {
    --primary-color: #6c63ff;   /* 主色调 - 可改成你喜欢的颜色 */
    --accent-color: #00d4aa;    /* 强调色 */
}
```

---

## 📋 项目结构

```
zy-shirley-portfolio/
├── index.html          # 主页面
├── README.md           # 说明文档
├── .gitignore          # Git 忽略配置
├── .nojekyll           # 禁用 Jekyll 处理
└── assets/
    ├── css/
    │   └── style.css   # 样式（含暗色/亮色主题）
    ├── js/
    │   └── main.js     # 交互逻辑（打字机/动画/主题切换）
    └── images/         # ← 你的图片放这里
```

---

## 🔧 常见问题

**Q: 网站打不开？**
→ 等待 1-2 分钟让 GitHub Pages 生效，或者检查一下 Steps 1-3 是否都完成了

**Q: 想更新内容？**
→ 本地修改文件 → `git add -A` → `git commit -m "更新说明"` → `git push`

**Q: 想要更短的链接？**
→ 自定义域名：Settings → Pages → Custom domain，填入你买的域名并配置 DNS

---

> 原始模板：[simplefolio](https://github.com/cobiwave/simplefolio) (MIT License)
