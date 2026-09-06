/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        thervo: {
          cool: '#6F9BA8',
          amber: '#B58A4A',
          orange: '#C56A38',
          critical: '#B84A43',
          airflow: '#6C8FA0',
          background: '#1A1C23', // industrial dark background
          panel: '#252830',
          border: '#3A3F4C',
          text: '#D1D5DB'
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
