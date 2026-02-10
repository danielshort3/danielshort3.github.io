/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        deep: {
          bg: "#0f172a",
          surface: "#13203a",
          panel: "#1b2f54",
          accent: "#00bcd4",
          accentDark: "#004d40",
          text: "#e2e8f0",
          muted: "#90a4bf"
        }
      },
      fontFamily: {
        sans: ["Inter", "Geist Sans", "ui-sans-serif", "system-ui", "sans-serif"]
      },
      boxShadow: {
        bento: "0 0 0 1px rgba(0, 188, 212, 0.18), 0 18px 40px rgba(2, 12, 27, 0.38)"
      },
      backgroundImage: {
        "deep-grid":
          "radial-gradient(circle at 15% 20%, rgba(0, 188, 212, 0.18), transparent 40%), radial-gradient(circle at 90% 5%, rgba(0, 77, 64, 0.35), transparent 42%)",
        "action-gradient": "linear-gradient(120deg, #004d40 0%, #00bcd4 100%)"
      }
    }
  },
  plugins: []
};
