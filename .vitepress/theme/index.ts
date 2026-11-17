import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import SiteStats from './components/SiteStats.vue'

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'layout-bottom': () => h(SiteStats)
    })
  }
}
