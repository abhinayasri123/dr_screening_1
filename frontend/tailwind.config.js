/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        midnight: {
          base: '#070707',
          sidebar: '#0B0B0B',
          card: '#121212',
          cardElevated: '#191919',
        },
        ivory: {
          50: '#FDFBF7',
          100: '#F5F0E6',
          200: '#E8E1D5',
          300: '#D0CCC3',
          400: '#85817A',
        },
        sage: {
          50: '#F4F6F5',
          100: '#E4EAE7',
          200: '#C5D3CD',
          300: '#A4B9B0',
          400: '#81958D',
          500: '#687E76',
          600: '#52645E',
        },
        accent: {
          sage: '#81958D',
          ivory: '#E8E1D5',
          muted: '#292929',
        },
        primary: {
          50: '#FDFBF7',
          100: '#F5F0E6',
          200: '#E8E1D5',
          300: '#D0CCC3',
          400: '#85817A',
          500: '#81958D',
          600: '#E8E1D5',
          700: '#F5F0E6',
          800: '#191919',
          900: '#121212',
        },
        border: '#292929',
        card: '#121212',
        background: '#070707',
      },
      fontFamily: {
        sans: ['Inter', 'Manrope', 'system-ui', '-apple-system', 'sans-serif'],
        heading: ['Manrope', 'Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
