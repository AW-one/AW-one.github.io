import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Kingzilv_tobe_better",
  description: "Personal Homepage | AIGC, Computer Vision, Fashion Intelligence and Multimodal AI",

  base: "/",

  cleanUrls: true,
  ignoreDeadLinks: true,
  markdown: {
    math: true
  },
  
  head: [
    ['meta', { name: 'author', content: 'KINGZILV' }],
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
            { text: 'Overview', link: '/publications/' },
            { text: 'RedesignNet', link: '/publications/redesignnet' },
            { text: 'GTDNet', link: '/publications/NCAA' }
          ]
        }
      ],
      '/notes/': [
        {
          text: 'Notes',
          items: [
            { text: 'Overview', link: '/notes/' },
            { text: 'BASE Models', 
              collapsed: false,
              link: '/notes/base_models/',
              items: [
                { text: 'Diffusion Models / DDPM', link: '/notes/base_models/diffusion' },
                { text: 'DiT', link: '/notes/base_models/dit' },
                { text: 'FLUX', link: '/notes/base_models/flux' },
                { text: 'Qwen', link: '/notes/base_models/qwen' },
              ]
            },
            { text: 'Style Transfer Models', 
              collapsed: false,
              link: '/notes/style_transfer_models/',
            },
            { text: 'Record', 
              collapsed: true,
              link: '/notes/records/',
              items: [
                { text: 'LLM 面试题记录', link: '/notes/records/llms' },
                {
                  text: '基础概念',
                  collapsed: false,
                  items: [
                    { text: 'Loss Functions', link: '/notes/records/loss' },
                    { text: 'Deep Learning Concepts', link: '/notes/records/basic_concepts' },
                    { text: 'Large Model Concepts', link: '/notes/records/large_model_concepts' },
                    { text: '概念笔记模板', link: '/notes/records/concept_template' },
                  ]
                },
              ]
            },
            { text: '模板', link: '/notes/paper_template' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/AW-one' }
    ],

    footer: {
      message: 'Personal homepage built with VitePress.',
      copyright: 'Copyright © 2026 王泯皓 · Allen Wang'
    },

    search: {
      provider: 'local'
    }
  }
})
