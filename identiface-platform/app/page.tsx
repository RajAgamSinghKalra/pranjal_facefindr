"use client"

import { useState, useEffect } from "react"
import {
  Search,
  Users,
  Upload,
  Moon,
  Sun,
  WifiOff,
  RefreshCw,
  Settings,
  HelpCircle,
  ImageIcon,
  Eye,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { useTheme } from "next-themes"
import { motion, AnimatePresence } from "framer-motion"
import { UploadZone } from "@/components/upload-zone"
import { ProcessingModal } from "@/components/processing-modal"
import { BackendSetupGuide } from "@/components/backend-setup-guide"
import { ConnectionDiagnostics } from "@/components/connection-diagnostics"
import { PeopleGallery } from "@/components/people-gallery"
import { SearchResults } from "@/components/search-results"
import { SearchFilters } from "@/components/search-filters"
import { ClusterView } from "@/components/cluster-view"
import { apiService } from "@/lib/api-service"

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

// Jigly animation variants
const jiglyAnimation = {
  hover: {
    scale: 1.05,
    rotate: [0, -1, 1, -1, 0],
    transition: {
      scale: { duration: 0.2 },
      rotate: { duration: 0.5, repeat: Number.POSITIVE_INFINITY, repeatType: "reverse" as const },
    },
  },
  tap: {
    scale: 0.95,
    transition: { duration: 0.1 },
  },
}

const bounceAnimation = {
  hover: {
    y: [-2, -8, -2],
    transition: {
      y: { duration: 0.6, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" },
    },
  },
}

const wiggleAnimation = {
  hover: {
    rotate: [0, -3, 3, -3, 0],
    scale: 1.1,
    transition: {
      rotate: { duration: 0.4, repeat: Number.POSITIVE_INFINITY },
      scale: { duration: 0.2 },
    },
  },
}

export default function IdentiFace() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)
  const [clusters, setClusters] = useState<Cluster[]>([])
  const [activeTab, setActiveTab] = useState("people")
  const [apiStatus, setApiStatus] = useState<{
    status: "checking" | "online" | "offline"
    endpoint: string
    lastChecked?: Date
    error?: string
  }>({
    status: "checking",
    endpoint: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  })
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingStage, setProcessingStage] = useState("")
  const [showSetupGuide, setShowSetupGuide] = useState(false)
  const [showDiagnostics, setShowDiagnostics] = useState(false)
  const [stats, setStats] = useState({
    totalPhotos: 0,
    totalFaces: 0,
    totalClusters: 0,
  })
  const [selectedCluster, setSelectedCluster] = useState<Cluster | null>(null)
  const [searchResults, setSearchResults] = useState<Face[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [similarityThreshold, setSimilarityThreshold] = useState(60)
  const [maxResults, setMaxResults] = useState(20)

  useEffect(() => {
    setMounted(true)
    checkApiStatus()
  }, [])

  useEffect(() => {
    if (apiStatus.status === "online") {
      loadData()
    } else if (apiStatus.status === "offline") {
      setIsLoading(false)
    }
  }, [apiStatus.status])

  const checkApiStatus = async () => {
    setApiStatus((prev) => ({ ...prev, status: "checking" }))

    try {
      const statusInfo = await apiService.getApiStatus()
      console.log("API Status:", statusInfo)

      setApiStatus({
        status: statusInfo.isOnline ? "online" : "offline",
        endpoint: statusInfo.endpoint,
        lastChecked: statusInfo.lastChecked,
        error: statusInfo.error,
      })
    } catch (error) {
      console.error("API Status Check Error:", error)
      setApiStatus((prev) => ({
        ...prev,
        status: "offline",
        lastChecked: new Date(),
        error: error instanceof Error ? error.message : "Connection failed",
      }))
    }
  }

  const loadData = async () => {
    setIsLoading(true)
    try {
      // Load stats
      const statsData = await apiService.getStats()
      setStats({
        totalPhotos: statsData.totalPhotos || 0,
        totalFaces: statsData.totalFaces || 0,
        totalClusters: statsData.totalClusters || 0,
      })

      // Load clusters
      const clustersData = await apiService.getClusters()
      setClusters(clustersData || [])
    } catch (error) {
      console.error("Failed to load data:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleImageUpload = async (file: File) => {
    setIsProcessing(true)
    setProcessingStage("Uploading image...")

    try {
      // Simulate processing stages
      const stages = [
        "Uploading image...",
        "Detecting faces...",
        "Extracting features...",
        "Searching database...",
        "Analyzing matches...",
        "Complete!",
      ]

      for (let i = 0; i < stages.length - 1; i++) {
        setProcessingStage(stages[i])
        await new Promise((resolve) => setTimeout(resolve, 800))
      }

      // Actual API call
      if (activeTab === "upload") {
        await apiService.uploadGroupPhoto(file)
        await loadData() // Refresh data
      } else if (activeTab === "search") {
        const results = await apiService.searchSimilarFaces(file, similarityThreshold / 100)
        setSearchResults(results.slice(0, maxResults))
      }

      setProcessingStage("Complete!")
      await new Promise((resolve) => setTimeout(resolve, 800))
    } catch (error) {
      console.error("Upload error:", error)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleClusterSelect = (cluster: Cluster) => {
    setSelectedCluster(cluster)
  }

  const handleBackToGallery = () => {
    setSelectedCluster(null)
  }

  // Handle stat card clicks
  const handleStatClick = (statType: string) => {
    setActiveTab("people")
    setSelectedCluster(null)

    // You can add specific filtering logic here based on statType
    if (statType === "photos") {
      // Could show a photos view
      console.log("Show all photos")
    } else if (statType === "faces") {
      // Could show all faces
      console.log("Show all faces")
    } else if (statType === "people") {
      // Show people gallery
      console.log("Show all people")
    }
  }

  // Toggle theme function
  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark")
  }

  if (!mounted) return null

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-md sticky top-0 z-10">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo with jigly animation */}
            <motion.div
              className="flex items-center space-x-3 cursor-pointer"
              variants={jiglyAnimation}
              whileHover="hover"
              whileTap="tap"
              onClick={() => setActiveTab("people")}
            >
              <motion.div
                className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center"
                variants={wiggleAnimation}
                whileHover="hover"
              >
                <Users className="w-7 h-7 text-white" />
              </motion.div>
              <div>
                <h1 className="text-2xl font-bold text-white">IdentiFace</h1>
                <div className="text-sm text-gray-400 flex items-center space-x-2">
                  <span>AI-Powered Face Recognition</span>
                  <div className="flex items-center space-x-1">
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1, repeat: Number.POSITIVE_INFINITY }}
                    >
                      {apiStatus.status === "online" ? (
                        <div className="w-3 h-3 bg-green-400 rounded-full" />
                      ) : apiStatus.status === "checking" ? (
                        <div className="w-3 h-3 bg-yellow-400 rounded-full" />
                      ) : (
                        <WifiOff className="w-3 h-3 text-red-400" />
                      )}
                    </motion.div>
                    <span
                      className={
                        apiStatus.status === "online"
                          ? "text-green-400"
                          : apiStatus.status === "checking"
                            ? "text-yellow-400"
                            : "text-red-400"
                      }
                    >
                      {apiStatus.status === "online"
                        ? "Online"
                        : apiStatus.status === "checking"
                          ? "Checking..."
                          : "Offline"}
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Navigation with bouncy buttons */}
            <div className="flex items-center space-x-2">
              <motion.div variants={bounceAnimation} whileHover="hover">
                <Button
                  variant={activeTab === "upload" ? "default" : "ghost"}
                  onClick={() => setActiveTab("upload")}
                  className="flex items-center space-x-2 text-white hover:bg-gray-800"
                >
                  <Upload className="w-4 h-4" />
                  <span>Upload</span>
                </Button>
              </motion.div>

              <motion.div variants={bounceAnimation} whileHover="hover">
                <Button
                  variant={activeTab === "people" ? "default" : "ghost"}
                  onClick={() => {
                    setActiveTab("people")
                    setSelectedCluster(null)
                  }}
                  className="flex items-center space-x-2 text-white hover:bg-gray-800"
                >
                  <Users className="w-4 h-4" />
                  <span>People</span>
                </Button>
              </motion.div>

              <motion.div variants={bounceAnimation} whileHover="hover">
                <Button
                  variant={activeTab === "search" ? "default" : "ghost"}
                  onClick={() => setActiveTab("search")}
                  className="flex items-center space-x-2 text-white hover:bg-gray-800"
                >
                  <Search className="w-4 h-4" />
                  <span>Search</span>
                </Button>
              </motion.div>

              <motion.div variants={wiggleAnimation} whileHover="hover">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowDiagnostics(true)}
                  className="text-white hover:bg-gray-800"
                >
                  <Settings className="w-5 h-5" />
                </Button>
              </motion.div>

              <motion.div variants={jiglyAnimation} whileHover="hover" whileTap="tap">
                <Button variant="ghost" size="icon" onClick={toggleTheme} className="text-white hover:bg-gray-800">
                  <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                  <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
                </Button>
              </motion.div>
            </div>
          </div>
        </div>
      </header>

      {/* Connection Alert */}
      {apiStatus.status === "offline" && (
        <div className="border-b border-red-800/50">
          <div className="container mx-auto px-6 py-3">
            <Alert className="border-red-800 bg-red-900/20">
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center space-x-2 mb-1">
                    <motion.div
                      className="w-4 h-4 rounded-full bg-red-500"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1, repeat: Number.POSITIVE_INFINITY }}
                    />
                    <span className="font-semibold text-red-400">Backend Connection Issue</span>
                  </div>
                  <AlertDescription className="text-red-300 mb-2">
                    Running in demo mode: Backend at{" "}
                    <code className="bg-gray-800 px-1 rounded">{apiStatus.endpoint}</code> is not responding.
                    {apiStatus.error && <div className="mt-1 text-sm">Error: {apiStatus.error}</div>}
                  </AlertDescription>
                  <div className="text-gray-300 text-sm">
                    The app is running in demo mode with sample data.
                    {apiStatus.lastChecked && (
                      <div className="mt-1">Last checked: {apiStatus.lastChecked.toLocaleTimeString()}</div>
                    )}
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <motion.div variants={jiglyAnimation} whileHover="hover" whileTap="tap">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowSetupGuide(true)}
                      disabled={apiStatus.status === "checking"}
                      className="border-blue-600 text-blue-400 hover:bg-blue-900/30 flex items-center space-x-2 bg-transparent"
                    >
                      <HelpCircle className="w-4 h-4" />
                      <span>Setup Guide</span>
                    </Button>
                  </motion.div>
                  <motion.div variants={jiglyAnimation} whileHover="hover" whileTap="tap">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={checkApiStatus}
                      disabled={apiStatus.status === "checking"}
                      className="border-red-600 text-red-400 hover:bg-red-900/30 flex items-center space-x-2 bg-transparent"
                    >
                      <RefreshCw className={`w-4 h-4 ${apiStatus.status === "checking" ? "animate-spin" : ""}`} />
                      <span>{apiStatus.status === "checking" ? "Checking..." : "Retry"}</span>
                    </Button>
                  </motion.div>
                </div>
              </div>
            </Alert>
          </div>
        </div>
      )}

      {/* Stats Section with clickable cards */}
      <div className="border-b border-gray-800">
        <div className="container mx-auto px-6 py-8">
          <div className="grid grid-cols-3 gap-8">
            {/* Total Photos - Clickable */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="cursor-pointer"
              onClick={() => handleStatClick("photos")}
            >
              <div className="flex items-center space-x-4">
                <motion.div
                  className="w-16 h-16 bg-blue-600 rounded-xl flex items-center justify-center cursor-pointer"
                  variants={bounceAnimation}
                  whileHover="hover"
                  whileTap={{ scale: 0.95 }}
                >
                  <ImageIcon className="w-8 h-8 text-white" />
                </motion.div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">Total Photos</div>
                  <div className="text-4xl font-bold text-white">{stats.totalPhotos}</div>
                  <div className="w-32 h-1 bg-gray-800 rounded-full mt-2">
                    <motion.div
                      className="h-full bg-blue-500 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: "100%" }}
                      transition={{ duration: 1.5, delay: 0.5 }}
                    />
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Faces Detected - Clickable */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="cursor-pointer"
              onClick={() => handleStatClick("faces")}
            >
              <div className="flex items-center space-x-4">
                <motion.div
                  className="w-16 h-16 bg-purple-600 rounded-xl flex items-center justify-center cursor-pointer"
                  variants={wiggleAnimation}
                  whileHover="hover"
                  whileTap={{ scale: 0.95 }}
                >
                  <Eye className="w-8 h-8 text-white" />
                </motion.div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">Faces Detected</div>
                  <div className="text-4xl font-bold text-white">{stats.totalFaces}</div>
                  <div className="w-32 h-1 bg-gray-800 rounded-full mt-2">
                    <motion.div
                      className="h-full bg-purple-500 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: "100%" }}
                      transition={{ duration: 1.5, delay: 0.7 }}
                    />
                  </div>
                </div>
              </div>
            </motion.div>

            {/* People Identified - Clickable */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="cursor-pointer"
              onClick={() => handleStatClick("people")}
            >
              <div className="flex items-center space-x-4">
                <motion.div
                  className="w-16 h-16 bg-green-600 rounded-xl flex items-center justify-center cursor-pointer"
                  variants={jiglyAnimation}
                  whileHover="hover"
                  whileTap="tap"
                >
                  <Users className="w-8 h-8 text-white" />
                </motion.div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">People Identified</div>
                  <div className="text-4xl font-bold text-white">{stats.totalClusters}</div>
                  <div className="w-32 h-1 bg-gray-800 rounded-full mt-2">
                    <motion.div
                      className="h-full bg-green-500 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: "100%" }}
                      transition={{ duration: 1.5, delay: 0.9 }}
                    />
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-12">
        <AnimatePresence mode="wait">
          {activeTab === "upload" && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="text-center space-y-8"
            >
              <div>
                <h2 className="text-4xl font-bold text-white mb-4">Upload & Process Faces</h2>
                <div className="text-xl text-gray-400 max-w-3xl mx-auto">
                  Upload photos to detect and cluster faces. Our AI will automatically identify and group similar faces
                  across all your photos.
                </div>
              </div>
              <UploadZone
                onImageUpload={handleImageUpload}
                isLoading={isLoading}
                apiStatus={{ status: apiStatus.status }}
              />
            </motion.div>
          )}

          {activeTab === "people" && !selectedCluster && (
            <motion.div
              key="people"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-8"
            >
              <div className="text-center">
                <h2 className="text-4xl font-bold text-white mb-4">People Gallery</h2>
                <div className="text-xl text-gray-400 max-w-3xl mx-auto mb-6">
                  Browse all the people identified in your photos. Each cluster represents a unique person found across
                  your collection.
                </div>
                {apiStatus.status === "offline" && (
                  <Badge variant="secondary" className="bg-gray-800 text-gray-300">
                    Demo Mode - Start backend for real data
                  </Badge>
                )}
              </div>

              <PeopleGallery
                clusters={clusters}
                onClusterSelect={handleClusterSelect}
                isLoading={isLoading}
                similarityThreshold={similarityThreshold}
              />
            </motion.div>
          )}

          {activeTab === "people" && selectedCluster && (
            <motion.div
              key="cluster-view"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <ClusterView cluster={selectedCluster} onBack={handleBackToGallery} />
            </motion.div>
          )}

          {activeTab === "search" && (
            <motion.div
              key="search"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-8"
            >
              <div className="text-center">
                <h2 className="text-4xl font-bold text-white mb-4">Face Search</h2>
                <div className="text-xl text-gray-400 max-w-3xl mx-auto mb-6">
                  Upload a photo to find similar faces in your collection. Our AI will instantly identify and match
                  faces across all your photos.
                </div>
              </div>

              {/* Search Filters */}
              <div className="max-w-md mx-auto">
                <SearchFilters
                  similarityThreshold={similarityThreshold}
                  onSimilarityChange={setSimilarityThreshold}
                  maxResults={maxResults}
                  onMaxResultsChange={setMaxResults}
                  totalAvailable={stats.totalFaces}
                />
              </div>

              <UploadZone
                onImageUpload={handleImageUpload}
                isLoading={isLoading}
                apiStatus={{ status: apiStatus.status }}
              />

              <SearchResults
                results={searchResults}
                isLoading={isLoading}
                similarityThreshold={similarityThreshold}
                maxResults={maxResults}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Modals */}
      <ProcessingModal isOpen={isProcessing} stage={processingStage} onClose={() => setIsProcessing(false)} />
      <BackendSetupGuide isOpen={showSetupGuide} onClose={() => setShowSetupGuide(false)} />
      <ConnectionDiagnostics isOpen={showDiagnostics} onClose={() => setShowDiagnostics(false)} />
    </div>
  )
}
