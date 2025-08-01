import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

/**
 * Normalize image URLs coming from the backend.
 * If the URL is relative or points to localhost, prefix it with the
 * API base URL so images are served from the correct backend host.
 */
export function normalizeImageUrl(url: string): string {
  if (!url) return ""
  try {
    const api = new URL(API_BASE_URL)
    const parsed = new URL(url, API_BASE_URL)
    parsed.protocol = api.protocol
    parsed.host = api.host
    return parsed.toString()
  } catch {
    const cleanUrl = url.startsWith("/") ? url : `/${url}`
    return `${API_BASE_URL}${cleanUrl}`
  }
}
