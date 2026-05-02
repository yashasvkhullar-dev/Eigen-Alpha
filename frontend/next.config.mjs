/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  async rewrites() {
    // Only proxy API calls to FastAPI backend in local development.
    // On Vercel (production), the dashboard runs standalone with mock
    // data fallbacks — no backend required.
    if (process.env.NODE_ENV === "development") {
      return [
        {
          source: "/api/:path*",
          destination: "http://localhost:8000/api/:path*",
        },
      ]
    }
    return []
  },
}

export default nextConfig
