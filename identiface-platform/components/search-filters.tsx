"use client"
import { Slider } from "@/components/ui/slider"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Settings, Users, Target } from "lucide-react"
import { motion } from "framer-motion"

interface SearchFiltersProps {
  similarityThreshold: number
  onSimilarityChange: (value: number) => void
  maxResults: number
  onMaxResultsChange: (value: number) => void
  totalAvailable?: number
}

export function SearchFilters({
  similarityThreshold,
  onSimilarityChange,
  maxResults,
  onMaxResultsChange,
  totalAvailable = 100,
}: SearchFiltersProps) {
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center space-x-2 text-white text-sm">
            <Settings className="w-4 h-4" />
            <span>Search Filters</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Similarity Threshold */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Target className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-medium text-white">Similarity Threshold</span>
              </div>
              <Badge variant="outline" className="border-blue-600 text-blue-400">
                {similarityThreshold}%
              </Badge>
            </div>
            <Slider
              value={[similarityThreshold]}
              onValueChange={(value) => onSimilarityChange(value[0])}
              max={100}
              min={0}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-400">
              <span>0% (Show All)</span>
              <span>100% (Exact Match)</span>
            </div>
          </div>

          {/* Max Results */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Users className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium text-white">Max Results</span>
              </div>
              <Badge variant="outline" className="border-purple-600 text-purple-400">
                {maxResults}
              </Badge>
            </div>
            <Slider
              value={[maxResults]}
              onValueChange={(value) => onMaxResultsChange(value[0])}
              max={Math.min(totalAvailable, 50)}
              min={1}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-400">
              <span>1 result</span>
              <span>{Math.min(totalAvailable, 50)} results</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
