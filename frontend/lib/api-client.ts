/**
 * EigenAlpha API Client
 *
 * Thin fetch wrapper for the FastAPI backend.
 * Falls back to the provided default value if the backend
 * is unreachable or returns an error.
 */

const API_BASE = "/api"

export async function fetchAPI<T>(endpoint: string, fallback: T): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    })
    if (!res.ok) throw new Error(`API ${res.status}`)
    const data = await res.json()
    // If the backend reports data is not available, use fallback
    if (data && data.available === false) {
      return fallback
    }
    return data as T
  } catch {
    // Backend unreachable — use mock data
    return fallback
  }
}
