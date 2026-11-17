<template>
  <div v-if="visible" class="site-stats">
    <span>总访问量 {{ sitePv }} 次</span>
    <span class="dot">·</span>
    <span>访客数 {{ siteUv }} 人</span>
    <template v-if="pagePv !== null">
      <span class="dot">·</span>
      <span>本页阅读 {{ pagePv }} 次</span>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { useRoute } from 'vitepress'

const sitePv = ref<number | null>(null)
const siteUv = ref<number | null>(null)
const pagePv = ref<number | null>(null)
const visible = ref(false)
const route = useRoute()
let lastPath = ''

async function logVisit() {
  if (typeof window === 'undefined') return
  const host = window.location.host || 'unknown-host'
  const cookieKey = `vercount_uv_${host.replace(/[^a-zA-Z0-9_-]/g, '_')}`
  const isNewUv = !document.cookie
    .split('; ')
    .some((item) => item.startsWith(`${cookieKey}=`))
  if (isNewUv) {
    document.cookie = `${cookieKey}=1; path=/; max-age=31536000; samesite=lax`
  }
  try {
    const res = await fetch('https://events.vercount.one/api/v2/log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: window.location.href, isNewUv })
    })
    if (!res.ok) return
    const json = await res.json()
    const d = json && json.data
    if (d && d.site_pv != null && d.site_uv != null) {
      sitePv.value = Number(d.site_pv)
      siteUv.value = Number(d.site_uv)
      pagePv.value = d.page_pv != null ? Number(d.page_pv) : null
      visible.value = true
    }
  } catch {
    // 统计服务不可用时整行不显示，不影响阅读
  }
}

onMounted(() => {
  lastPath = route.path
  logVisit()
})

watch(
  () => route.path,
  (p) => {
    if (p !== lastPath) {
      lastPath = p
      logVisit()
    }
  }
)
</script>

<style scoped>
.site-stats {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 4px 16px 28px;
  font-size: 13px;
  color: #8a8a8a;
  text-align: center;
}
.site-stats .dot {
  opacity: 0.6;
}
</style>
