const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

interface Face {
  id: string
  path: string
  similarity: number
  cluster_id: number
  original_image: string
  bounding_box: number[]
  landmarks: number[][]
}

interface Cluster {
  id: number
  faces: Face[]
  representative_face: Face
  size: number
}

interface Stats {
  totalFaces: number
  totalClusters: number
  totalPhotos: number
}

class ApiService {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`
    console.log(`🔄 API Request: ${url}`)

    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout

      const response = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
        signal: controller.signal,
        ...options,
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()
      console.log(`✅ API Response: ${url}`, data)
      return data
    } catch (error) {
      console.error(`❌ API Error: ${url}`, error)
      throw error
    }
  }

  async getClusters(): Promise<Cluster[]> {
    return this.request<Cluster[]>("/api/clusters")
  }

  async getStats(): Promise<Stats> {
    return this.request<Stats>("/api/stats")
  }

  async searchSimilarFaces(file: File, threshold: number): Promise<Face[]> {
    const formData = new FormData()
    formData.append("file", file)
    formData.append("threshold", threshold.toString())

    const url = `${API_BASE_URL}/api/search`
    console.log(`🔄 API Upload Request: ${url}`)

    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Search request failed: ${response.statusText}`)
      }

      const data = await response.json()
      console.log(`✅ API Search Response:`, data)
      return data
    } catch (error) {
      console.error(`❌ API Search Error:`, error)
      throw error
    }
  }

  async getCluster(clusterId: number): Promise<Cluster> {
    return this.request<Cluster>(`/api/clusters/${clusterId}`)
  }

  async uploadGroupPhoto(file: File): Promise<{ message: string; faces_detected: number }> {
    const formData = new FormData()
    formData.append("file", file)

    const url = `${API_BASE_URL}/api/upload`
    console.log(`🔄 API Upload Request: ${url}`)

    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Upload request failed: ${response.statusText}`)
      }

      const data = await response.json()
      console.log(`✅ API Upload Response:`, data)
      return data
    } catch (error) {
      console.error(`❌ API Upload Error:`, error)
      throw error
    }
  }

  async processGroupPhotos(): Promise<{ message: string; total_faces: number }> {
    return this.request<{ message: string; total_faces: number }>("/api/process", {
      method: "POST",
    })
  }

  async updateSimilarityThreshold(threshold: number): Promise<{ message: string }> {
    return this.request<{ message: string }>("/api/similarity-threshold", {
      method: "PUT",
      body: JSON.stringify({ threshold }),
    })
  }

  // Check if API is available
  async checkApiHealth(): Promise<boolean> {
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 3000)

      const response = await fetch(`${API_BASE_URL}/health`, {
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (response.ok) {
        console.log("✅ Backend API is online")
        return true
      }

      console.warn("⚠️ Backend health check failed")
      return false
    } catch (error) {
      console.error("❌ Backend API is offline", error)
      return false
    }
  }

  async getApiStatus(): Promise<{
    isOnline: boolean
    endpoint: string
    lastChecked: Date
    error?: string
  }> {
    const lastChecked = new Date()

    try {
      const isOnline = await this.checkApiHealth()
      return {
        isOnline,
        endpoint: API_BASE_URL,
        lastChecked,
        error: isOnline ? undefined : "API health check failed",
      }
    } catch (error) {
      return {
        isOnline: false,
        endpoint: API_BASE_URL,
        lastChecked,
        error: error instanceof Error ? error.message : "Unknown error",
      }
    }
  }

  async getDebugInfo(): Promise<any> {
    return this.request<any>("/api/debug/images")
  }
}

export const apiService = new ApiService()
