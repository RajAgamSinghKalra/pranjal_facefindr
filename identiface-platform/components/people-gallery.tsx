"use client"

import { useState } from "react"
import { Users, Eye, RefreshCw } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { motion, AnimatePresence } from "framer-motion"
import { cn, normalizeImageUrl } from "@/lib/utils"

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

interface PeopleGalleryProps {
  clusters: Cluster[]
  onClusterSelect: (cluster: Cluster) => void
  isLoading: boolean
  similarityThreshold: number
}

export function PeopleGallery({ clusters, onClusterSelect, isLoading, similarityThreshold }: PeopleGalleryProps) {
  const [searchTerm, setSearchTerm] = useState("")
  const [hoveredCluster, setHoveredCluster] = useState<number | null>(null)
  const [imageErrors, setImageErrors] = useState<Set<string>>(new Set())
  const [imageLoadStates, setImageLoadStates] = useState<Map<string, "loading" | "loaded" | "error">>(new Map())

  const filteredClusters = clusters.filter(
    (cluster) => cluster.size >= Math.floor(similarityThreshold / 20), // Simple filtering logic
  )

  const searchedClusters = filteredClusters.filter(
    (cluster) => searchTerm === "" || cluster.id.toString().includes(searchTerm),
  )

  const handleImageError = (clusterId: number) => {
    console.error(`Image failed to load for cluster ${clusterId}`)
    setImageErrors((prev) => new Set([...prev, clusterId.toString()]))
    setImageLoadStates((prev) => new Map(prev.set(clusterId.toString(), "error")))
  }

  const handleImageLoad = (clusterId: number) => {
    console.log(`Image loaded successfully for cluster ${clusterId}`)
    setImageErrors((prev) => {
      const newSet = new Set(prev)
      newSet.delete(clusterId.toString())
      return newSet
    })
    setImageLoadStates((prev) => new Map(prev.set(clusterId.toString(), "loaded")))
  }

  const handleImageLoadStart = (clusterId: number) => {
    setImageLoadStates((prev) => new Map(prev.set(clusterId.toString(), "loading")))
  }

  const retryImageLoad = (clusterId: number) => {
    setImageErrors((prev) => {
      const newSet = new Set(prev)
      newSet.delete(clusterId.toString())
      return newSet
    })
    setImageLoadStates((prev) => new Map(prev.set(clusterId.toString(), "loading")))

    // Force re-render by updating the key
    const img = document.querySelector(`img[data-cluster-id="${clusterId}"]`) as HTMLImageElement
    if (img) {
      const originalSrc = img.src
      img.src = ""
      setTimeout(() => {
        img.src = originalSrc
      }, 100)
    }
  }

  const getImageLoadState = (clusterId: number) => {
    return imageLoadStates.get(clusterId.toString()) || "loading"
  }

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        {Array.from({ length: 12 }).map((_, i) => (
          <Card key={i} className="aspect-square bg-gray-800 border-gray-700">
            <CardContent className="p-4 h-full">
              <div className="w-full h-full bg-gray-600 animate-pulse rounded-lg" />
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h2 className="text-3xl font-bold mb-2 text-white">People Gallery</h2>
          <p className="text-gray-400">
            {searchedClusters.length} people found • Similarity threshold: {similarityThreshold}%
          </p>
          <div className="text-sm text-gray-500 mt-1">
            Images loaded: {Array.from(imageLoadStates.values()).filter((state) => state === "loaded").length} /{" "}
            {searchedClusters.length}
          </div>
        </div>
        <div className="w-64">
          <Input
            placeholder="Search people..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-gray-800 border-gray-600 text-white"
          />
        </div>
      </motion.div>

      <AnimatePresence>
        {searchedClusters.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="text-center py-12"
          >
            <Users className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2 text-white">No people found</h3>
            <p className="text-gray-400">Try adjusting the similarity threshold or process more photos</p>
          </motion.div>
        ) : (
          <motion.div layout className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {searchedClusters.map((cluster, index) => (
              <motion.div
                key={cluster.id}
                layout
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ delay: index * 0.05 }}
                whileHover={{ y: -4 }}
                onHoverStart={() => setHoveredCluster(cluster.id)}
                onHoverEnd={() => setHoveredCluster(null)}
              >
                <Card
                  className={cn(
                    "aspect-square cursor-pointer transition-all duration-300 hover:shadow-lg bg-gray-800 border-gray-700",
                    hoveredCluster === cluster.id && "ring-2 ring-primary",
                  )}
                  onClick={() => onClusterSelect(cluster)}
                >
                  <CardContent className="p-3 h-full flex flex-col">
                    <div className="flex-1 relative overflow-hidden rounded-lg bg-gray-600">
                      {imageErrors.has(cluster.id.toString()) || !cluster.representative_face ? (
                        // Fallback placeholder with retry option
                        <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-blue-500 to-purple-600">
                          <div className="text-white text-center">
                            <Users className="w-8 h-8 mx-auto mb-2" />
                            <div className="text-xs mb-2">Person {cluster.id}</div>
                            {cluster.representative_face && (
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  retryImageLoad(cluster.id)
                                }}
                                className="text-xs p-1 h-auto text-white hover:bg-white/20"
                              >
                                <RefreshCw className="w-3 h-3 mr-1" />
                                Retry
                              </Button>
                            )}
                          </div>
                        </div>
                      ) : (
                        <>
                          {getImageLoadState(cluster.id) === "loading" && (
                            <div className="absolute inset-0 flex items-center justify-center bg-gray-600">
                              <RefreshCw className="w-6 h-6 text-gray-400 animate-spin" />
                            </div>
                          )}
                          <img
                            data-cluster-id={cluster.id}
                            src={normalizeImageUrl(cluster.representative_face.path) || "/placeholder.svg"}
                            alt={`Person ${cluster.id}`}
                            className={cn(
                              "w-full h-full object-cover transition-all duration-300",
                              getImageLoadState(cluster.id) === "loaded" ? "opacity-100 hover:scale-110" : "opacity-0",
                            )}
                            onLoadStart={() => handleImageLoadStart(cluster.id)}
                            onLoad={() => handleImageLoad(cluster.id)}
                            onError={() => handleImageError(cluster.id)}
                            crossOrigin="anonymous"
                          />
                        </>
                      )}

                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{
                          opacity: hoveredCluster === cluster.id ? 1 : 0,
                        }}
                        className="absolute inset-0 bg-black/50 flex items-center justify-center"
                      >
                        <Eye className="w-6 h-6 text-white" />
                      </motion.div>
                    </div>
                    <div className="mt-2 flex items-center justify-between">
                      <Badge variant="secondary" className="text-xs bg-gray-700 text-gray-300">
                        {cluster.size} photos
                      </Badge>
                      <div className="flex items-center space-x-1">
                        <span className="text-xs text-gray-400">ID: {cluster.id}</span>
                        {getImageLoadState(cluster.id) === "loaded" && (
                          <div className="w-2 h-2 bg-green-400 rounded-full" />
                        )}
                        {getImageLoadState(cluster.id) === "error" && (
                          <div className="w-2 h-2 bg-red-400 rounded-full" />
                        )}
                        {getImageLoadState(cluster.id) === "loading" && (
                          <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Debug Info */}
      {process.env.NODE_ENV === "development" && (
        <div className="mt-8 p-4 bg-gray-800 rounded-lg text-xs text-gray-400">
          <h4 className="font-semibold mb-2">Debug Info:</h4>
          <div>Total clusters: {clusters.length}</div>
          <div>Filtered clusters: {searchedClusters.length}</div>
          <div>Images with errors: {imageErrors.size}</div>
          <div>Sample image URL: {clusters[0]?.representative_face?.path}</div>
        </div>
      )}
    </div>
  )
}
