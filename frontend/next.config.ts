import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    // Long-form videos with diarization + per-speaker voice cloning can
    // exceed 60 min on a single 4070 Ti (Chatterbox upload-mode is ~3× slower
    // than default-voice). Bumped to 3 h to fit longer interviews end-to-end.
    // Proper fix is to convert TTS to async job + poll, but this unblocks
    // long-form runs in the meantime.
    proxyTimeout: 10_800_000, // 3 hours
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
