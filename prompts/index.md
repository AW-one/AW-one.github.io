---
layout: page
sidebar: false
aside: false
title: Prompt Gallery
---

<div class="pg-wrap">
  <div class="pg-mast">
    <div>
      <h1>Prompt <em>Gallery</em></h1>
      <div class="pg-tag">收藏在网上看到的 AIGC 提示词：一张图，一段词，附出处 · 持续收集中</div>
    </div>
    <div class="pg-stats">已收藏 1 条</div>
  </div>

  <div class="pg-card">
    <div class="pg-img pg-img-empty"><span>图片待补<br><small>生成图随后补上</small></span></div>
    <div class="pg-body">
      <div class="pg-meta"><span class="pg-no">No.001</span><span class="pg-chip">产品广告</span><span class="pg-chip">微缩景观</span><span class="pg-chip">写实渲染</span></div>
      <h3>微缩工地 × 护肤瓶</h3>
      <p class="pg-prompt">A hyper-realistic miniature diorama product advertisement featuring an oversized luxury skincare pump bottle labeled "LUXEVEIL Skin Science - Radiance Nourishing Body Lotion" in cream/beige with a polished gold pump top, placed on a circular platform. Tiny figurine construction workers dressed in yellow coveralls and white hard hats swarm around the bottle climbing scaffolding, painting the bottle with rollers, operating a tower crane, working near industrial tanks and pipework, and unloading a miniature flatbed truck. The scene includes metal scaffolding structures, industrial silos, orange traffic cones, wooden barricades, and storage barrels. The overall color palette is warm beige, cream, gold, and mustard yellow. Studio photography style with soft diffused lighting, no shadows, clean beige background. The concept metaphorically shows workers "crafting" or "building" the perfect lotion. Tilt-shift miniature aesthetic, ultra-detailed, commercial product photography, 8K resolution, photorealistic CGI render.</p>
      <div class="pg-src">来源：我的 <a href="https://github.com/AW-one/Prompt-Collecting-for-AIGC-in-image-generation" target="_blank" rel="noreferrer">Prompt-Collecting</a> 仓库收藏</div>
    </div>
  </div>

  <p class="pg-more">新的收藏会陆续补进来：看到好的图和词，就往这里放一条。</p>
</div>

<style>
.pg-wrap { max-width: 1040px; margin: 8px auto 0; padding: 48px 42px 56px; background-color: #FFFBF4; border-radius: 24px; color: #33302A; }
.pg-mast { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 30px; gap: 20px; }
.pg-mast h1 { font-size: 42px; font-weight: 900; margin: 0; border: none; padding: 0; color: #33302A; }
.pg-mast h1 em { color: #A4133C; font-style: normal; }
.pg-tag { color: #9A8F78; font-size: 14.5px; margin-top: 8px; }
.pg-stats { background: #fff; border: 2px dashed #FFB84D; color: #A86E12; border-radius: 14px; padding: 9px 16px; font-size: 13.5px; font-weight: 700; transform: rotate(1.2deg); white-space: nowrap; }
.pg-card { display: flex; gap: 0; background: #fff; border-radius: 20px; overflow: hidden; box-shadow: 0 14px 30px rgba(164,19,60,.08); margin-bottom: 26px; }
.pg-img { flex: 0 0 42%; min-height: 320px; background-size: cover; background-position: center; }
.pg-img img { width: 100%; height: 100%; object-fit: cover; display: block; }
.pg-img-empty { background: linear-gradient(135deg, #F6E7C8 0%, #FFE3EC 55%, #DCEEFF 100%); display: flex; align-items: center; justify-content: center; text-align: center; color: #8A6D1F; font-weight: 700; font-size: 17px; line-height: 1.6; }
.pg-img-empty small { font-weight: 500; font-size: 13px; color: #A79B7F; }
.pg-body { flex: 1; padding: 26px 30px; min-width: 0; }
.pg-meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.pg-no { font-family: "Songti SC", serif; font-weight: 700; background: #FFE3EC; color: #B03A5B; border-radius: 10px; padding: 3px 11px; transform: rotate(-2deg); display: inline-block; }
.pg-chip { background: #F3EFE6; color: #6E6759; border-radius: 999px; padding: 3px 11px; font-size: 12.5px; }
.pg-body h3 { font-family: "Songti SC", "Noto Serif CJK SC", serif; font-size: 22px; margin: 14px 0 10px; border: none; padding: 0; color: #33302A; }
.pg-prompt { font-size: 13.5px; line-height: 1.85; color: #55503F; background: #FFFBF0; border: 1.5px dashed #E8D9B0; border-radius: 12px; padding: 14px 16px; margin: 0; max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }
.pg-src { margin-top: 14px; font-size: 13px; color: #A79B7F; }
.pg-src a { color: #A4133C; font-weight: 700; text-decoration: none; }
.pg-src a:hover { text-decoration: underline; }
.pg-more { color: #A79B7F; font-size: 14px; margin: 6px 0 0; }
.dark .pg-wrap { background-color: #2A2118; color: #EDE4D3; }
.dark .pg-mast h1, .dark .pg-body h3 { color: #F3EAD8; }
.dark .pg-card { background: #33291D; }
.dark .pg-prompt { background: #2A2118; color: #CBBFA9; }
.dark .pg-stats { background: #33291D; }
@media (max-width: 720px) {
  .pg-wrap { padding: 32px 20px 40px; }
  .pg-mast { flex-direction: column; align-items: flex-start; }
  .pg-card { flex-direction: column; }
  .pg-img { flex: none; min-height: 220px; }
  .pg-body { padding: 20px 22px; }
}
</style>
