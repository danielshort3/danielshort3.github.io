/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        deep: {
          slate: '#0f172a',
          accent: '#00bcd4',
          text: '#e2e8f0',
          start: '#004d40',
          end: '#00bcd4'
        }
      },
      fontFamily: {
        sans: ['Inter', 'Geist Sans', 'system-ui', 'sans-serif']
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(0, 188, 212, 0.32), 0 20px 36px rgba(0, 188, 212, 0.2)'
      }
    }
  },
  plugins: []
};
