"use client"

import { ArrowLeft, Download, Share, Grid, List } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { motion } from "framer-motion"

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

interface ClusterViewProps {
  cluster: Cluster | null
  onBack: () => void
}

export function ClusterView({ cluster, onBack }: ClusterViewProps) {
  if (!cluster) {
    return (
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center py-12">
        <h3 className="text-xl font-semibold mb-2">No cluster selected</h3>
        <p className="text-muted-foreground mb-4">Select a person from the gallery to view their photos</p>
        <Button onClick={onBack}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Gallery
        </Button>
      </motion.div>
    )
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Button variant="ghost" onClick={onBack}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>
          <div>
            <h2 className="text-3xl font-bold">Person {cluster.id}</h2>
            <p className="text-muted-foreground">{cluster.size} photos found</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <Share className="w-4 h-4 mr-2" />
            Share
          </Button>
          <Button variant="outline" size="sm">
            <Download className="w-4 h-4 mr-2" />
            Download All
          </Button>
        </div>
      </div>

      {/* Representative Face */}
      <Card>
        <CardHeader>
          <CardTitle>Representative Face</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-6">
            <div className="w-32 h-32 rounded-lg overflow-hidden bg-muted">
              <img
                src={`/placeholder.svg?height=128&width=128&query=representative face person ${cluster.id}`}
                alt={`Person ${cluster.id}`}
                className="w-full h-full object-cover"
              />
            </div>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <Badge variant="secondary">Cluster ID: {cluster.id}</Badge>
                <Badge variant="outline">{cluster.size} instances</Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                This is the most representative face from all detected instances of this person.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* All Faces */}
      <Card>
        <CardHeader>
          <CardTitle>All Photos</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="grid" className="w-full">
            <TabsList className="mb-4">
              <TabsTrigger value="grid" className="flex items-center space-x-2">
                <Grid className="w-4 h-4" />
                <span>Grid View</span>
              </TabsTrigger>
              <TabsTrigger value="list" className="flex items-center space-x-2">
                <List className="w-4 h-4" />
                <span>List View</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="grid">
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {cluster.faces.map((face, index) => (
                  <motion.div
                    key={face.id}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                    whileHover={{ y: -2 }}
                  >
                    <Card className="aspect-square cursor-pointer hover:shadow-lg transition-all duration-300">
                      <CardContent className="p-2 h-full">
                        <div className="h-full rounded-lg overflow-hidden bg-muted">
                          <img
                            src={`/placeholder.svg?height=150&width=150&query=face instance ${index} person ${cluster.id}`}
                            alt={`Face instance ${index}`}
                            className="w-full h-full object-cover hover:scale-110 transition-transform duration-300"
                          />
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="list">
              <div className="space-y-3">
                {cluster.faces.map((face, index) => (
                  <motion.div
                    key={face.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    <Card className="hover:shadow-md transition-shadow">
                      <CardContent className="p-4">
                        <div className="flex items-center space-x-4">
                          <div className="w-16 h-16 rounded-lg overflow-hidden bg-muted flex-shrink-0">
                            <img
                              src={`/placeholder.svg?height=64&width=64&query=face instance ${index} person ${cluster.id}`}
                              alt={`Face instance ${index}`}
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <div className="flex-1 min-w-0">
                            <h4 className="font-medium truncate">{face.original_image}</h4>
                            <p className="text-sm text-muted-foreground">
                              Bounding box: {face.bounding_box.map(Math.round).join(", ")}
                            </p>
                          </div>
                          <Button variant="ghost" size="sm">
                            View Original
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </motion.div>
  )
}
