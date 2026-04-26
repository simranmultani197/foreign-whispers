import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    // Long-form videos with diarization on can produce 1k+ TTS segments;
    // observed ~14 min for a 13 min source with 360 segments on a 4070 Ti.
    // Set to 1 hour to give headroom for ~60 min source clips.
    proxyTimeout: 3_600_000, // 60 minutes
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.API_URL || "http://localhost:8080"}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
