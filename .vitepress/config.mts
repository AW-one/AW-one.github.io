import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Kingzilv_tobe_better",
  description: "Personal Homepage | AIGC, Computer Vision, Fashion Intelligence and Multimodal AI",

  base: "/",

  cleanUrls: true,

  head: [
    ['meta', { name: 'author', content: '王泯皓' }],
    ['meta', { name: 'keywords', content: 'AIGC, Computer Vision, Garment Generation, Virtual Try-on, Diffusion Models, Multimodal AI' }]
  ],

  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Projects', link: '/projects/' },
      { text: 'Publications', link: '/publications/' },
      { text: 'Notes', link: '/notes/' },
      { text: 'Resume', link: '/resume/' },
      { text: 'GitHub', link: 'https://github.com/AW-one' }
    ],

    sidebar: {
      '/projects/': [
        {
          text: 'Projects',
          items: [
            { text: 'Overview', link: '/projects/' },
            { text: 'RedesignNet', link: '/projects/redesignnet' },
            { text: 'ConceptCloth', link: '/projects/conceptcloth' },
            { text: 'ComfyUI LoRA Workflow', link: '/projects/comfyui-lora' }
          ]
        }
      ],
      '/publications/': [
        {
          text: 'Publications',
          items: [
            { text: 'Overview', link: '/publications/' }
          ]
        }
      ],
      '/notes/': [
        {
          text: 'Notes',
          items: [
            { text: 'Overview', link: '/notes/' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/AW-one' }
    ],

    footer: {
      message: 'Personal homepage built with VitePress.',
      copyright: 'Copyright © 2026 KINGZILV'
    },

    search: {
      provider: 'local'
    }
  }
})
