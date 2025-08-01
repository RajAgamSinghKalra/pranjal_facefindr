"use client"

import { useState, useEffect } from "react"
import { Search, ExternalLink, Info, Eye, X } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { motion, AnimatePresence } from "framer-motion"
import { cn, normalizeImageUrl, API_BASE_URL } from "@/lib/utils"

interface Face {
  id: string
  path: string
  similarity: number
  cluster_id: number
  original_image: string
  original_image_path?: string
  bounding_box: number[]
  landmarks: number[][]
}

interface SearchResultsProps {
  results: Face[]
  isLoading: boolean
  similarityThreshold: number
  maxResults: number
}

export function SearchResults({ results, isLoading, similarityThreshold, maxResults }: SearchResultsProps) {
  const [selectedFace, setSelectedFace] = useState<Face | null>(null)
  const [imageErrors, setImageErrors] = useState<Set<string>>(new Set())
  const [showOriginalPhoto, setShowOriginalPhoto] = useState(false)
  const [originalPhotoError, setOriginalPhotoError] = useState(false)

  // Reset modal state when dialog closes
  useEffect(() => {
    if (!showOriginalPhoto) {
      setOriginalPhotoError(false)
    }
  }, [showOriginalPhoto])

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.9) return "bg-green-500"
    if (similarity >= 0.7) return "bg-yellow-500"
    return "bg-red-500"
  }

  const getSimilarityLabel = (similarity: number) => {
    if (similarity >= 0.9) return "Excellent Match"
    if (similarity >= 0.7) return "Good Match"
    return "Possible Match"
  }

  const handleImageError = (faceId: string) => {
    setImageErrors((prev) => new Set([...prev, faceId]))
  }

  const handleImageLoad = (faceId: string) => {
    setImageErrors((prev) => {
      const newSet = new Set(prev)
      newSet.delete(faceId)
      return newSet
    })
  }

  const handleViewOriginalPhoto = (face: Face) => {
    setSelectedFace(face)
    setOriginalPhotoError(false) // Reset error state
    setShowOriginalPhoto(true)
  }

  const getOriginalPhotoUrl = (face: Face) => {
    const filename =
      face.original_image_path?.split("/").pop() || face.original_image
    return `${API_BASE_URL}/api/original-photo/${filename}`
  }

  // Filter and limit results based on similarity threshold and max results
  const filteredResults = results.filter((face) => face.similarity >= similarityThreshold / 100).slice(0, maxResults)

  if (isLoading) {
    return (
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2 text-white">
            <Search className="w-5 h-5" />
            <span>Searching...</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="aspect-square bg-gray-600 animate-pulse rounded-lg" />
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <>
      <AnimatePresence>
        {filteredResults.length > 0 && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}>
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="flex items-center justify-between text-white">
                  <div className="flex items-center space-x-2">
                    <Search className="w-5 h-5" />
                    <span>Search Results</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline" className="border-gray-600 text-gray-300">
                      {filteredResults.length} matches found
                    </Badge>
                    <Badge variant="secondary" className="bg-gray-700 text-gray-300">
                      ≥{similarityThreshold}% similarity
                    </Badge>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  {filteredResults.map((face, index) => (
                    <motion.div
                      key={face.id}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.1 }}
                      whileHover={{ y: -2 }}
                      className="relative group"
                    >
                      <Card
                        className={cn(
                          "aspect-square cursor-pointer transition-all duration-300 hover:shadow-lg bg-gray-700 border-gray-600",
                          selectedFace?.id === face.id && "ring-2 ring-primary",
                        )}
                        onClick={() => setSelectedFace(face)}
                      >
                        <CardContent className="p-2 h-full flex flex-col">
                          <div className="flex-1 relative overflow-hidden rounded-lg bg-gray-600">
                            {imageErrors.has(face.id) ? (
                              // Fallback placeholder
                              <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-blue-500 to-purple-600">
                                <div className="text-white text-center">
                                  <Eye className="w-8 h-8 mx-auto mb-2" />
                                  <div className="text-xs">Face {face.id.slice(-4)}</div>
                                </div>
                              </div>
                            ) : (
                              <img
                                src={normalizeImageUrl(face.path) || "/placeholder.svg"}
                                alt={`Similar face ${face.id}`}
                                className="w-full h-full object-cover"
                                onError={() => handleImageError(face.id)}
                                onLoad={() => handleImageLoad(face.id)}
                                crossOrigin="anonymous"
                              />
                            )}

                            <div className="absolute top-2 right-2">
                              <Badge className={cn("text-white text-xs", getSimilarityColor(face.similarity))}>
                                {Math.round(face.similarity * 100)}%
                              </Badge>
                            </div>

                            <motion.div
                              initial={{ opacity: 0 }}
                              whileHover={{ opacity: 1 }}
                              className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                              <Info className="w-5 h-5 text-white" />
                            </motion.div>
                          </div>

                          <div className="mt-2 space-y-1">
                            <div className="text-xs text-center">
                              <span className="font-medium text-white">{getSimilarityLabel(face.similarity)}</span>
                            </div>
                            <div className="text-xs text-gray-400 text-center">Cluster {face.cluster_id}</div>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  ))}
                </div>

                {filteredResults.length === 0 && results.length > 0 && (
                  <div className="text-center py-8">
                    <Search className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-white mb-2">No matches found</h3>
                    <p className="text-gray-400">Try lowering the similarity threshold below {similarityThreshold}%</p>
                  </div>
                )}

                {selectedFace && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-6 p-4 bg-gray-700 rounded-lg"
                  >
                    <h4 className="font-semibold mb-2 text-white">Face Details</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-400">Similarity:</span>
                        <span className="ml-2 font-medium text-white">
                          {Math.round(selectedFace.similarity * 100)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400">Cluster ID:</span>
                        <span className="ml-2 font-medium text-white">{selectedFace.cluster_id}</span>
                      </div>
                      <div className="col-span-2">
                        <span className="text-gray-400">Original Photo:</span>
                        <span className="ml-2 font-medium text-white truncate">{selectedFace.original_image}</span>
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      className="mt-3 bg-transparent text-white border-gray-600 hover:bg-gray-600"
                      onClick={() => handleViewOriginalPhoto(selectedFace)}
                    >
                      <ExternalLink className="w-4 h-4 mr-2" />
                      View Original Photo
                    </Button>
                  </motion.div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Original Photo Dialog */}
      <Dialog open={showOriginalPhoto} onOpenChange={setShowOriginalPhoto}>
        <DialogContent className="max-w-4xl bg-gray-900 border-gray-700">
          <DialogHeader>
            <DialogTitle className="text-white flex items-center justify-between">
              <span>Original Photo: {selectedFace?.original_image}</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowOriginalPhoto(false)}
                className="text-gray-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </Button>
            </DialogTitle>
          </DialogHeader>
          <div className="mt-4">
            {selectedFace && (
              <div className="relative">
                {originalPhotoError ? (
                  <div className="w-full h-96 flex items-center justify-center bg-gray-800 rounded-lg">
                    <div className="text-center text-gray-400">
                      <ExternalLink className="w-12 h-12 mx-auto mb-4" />
                      <p>Original photo not found</p>
                      <p className="text-sm mt-2">File: {selectedFace.original_image}</p>
                      <Button
                        variant="outline"
                        size="sm"
                        className="mt-4 bg-transparent border-gray-600 text-gray-300"
                        onClick={() => setOriginalPhotoError(false)}
                      >
                        Try Again
                      </Button>
                    </div>
                  </div>
                ) : (
                  <img
                    key={selectedFace.id} // Force re-render when face changes
                    src={getOriginalPhotoUrl(selectedFace) || "/placeholder.svg"}
                    alt={`Original photo: ${selectedFace.original_image}`}
                    className="w-full max-h-[70vh] object-contain rounded-lg"
                    onError={() => setOriginalPhotoError(true)}
                    crossOrigin="anonymous"
                  />
                )}
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}
