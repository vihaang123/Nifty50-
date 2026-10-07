import type { NextConfig } from "next";

// The backend URL is NOT configured here: it comes from NEXT_PUBLIC_API_URL (see .env.example).
const nextConfig: NextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
