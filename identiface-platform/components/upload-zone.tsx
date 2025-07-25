"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Upload } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface UploadZoneProps {
  onImageUpload: (file: File) => void
  isLoading: boolean
  apiStatus?: { status: "online" | "offline" }
}

// Jigly animation for upload icon
const uploadIconAnimation = {
  hover: {
    scale: 1.2,
    rotate: [0, -5, 5, -5, 0],
    y: [-5, -15, -5],
    transition: {
      scale: { duration: 0.2 },
      rotate: { duration: 0.6, repeat: Number.POSITIVE_INFINITY },
      y: { duration: 0.8, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" },
    },
  },
}

const badgeAnimation = {
  hover: {
    scale: 1.1,
    rotate: [0, -2, 2, -2, 0],
    transition: {
      scale: { duration: 0.2 },
      rotate: { duration: 0.4, repeat: Number.POSITIVE_INFINITY },
    },
  },
}

export function UploadZone({ onImageUpload, isLoading, apiStatus }: UploadZoneProps) {
  const [isDragActive, setIsDragActive] = useState(false)

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0]
      if (file) {
        onImageUpload(file)
      }
    },
    [onImageUpload],
  )

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpeg", ".jpg", ".png", ".webp"],
    },
    multiple: false,
    disabled: isLoading,
    onDragEnter: () => setIsDragActive(true),
    onDragLeave: () => setIsDragActive(false),
  })

  return (
    <div className="max-w-2xl mx-auto">
      <motion.div
        {...getRootProps()}
        className={cn(
          "relative border-2 border-dashed border-gray-600 rounded-2xl p-16 text-center cursor-pointer transition-all duration-300",
          isDragActive && "border-blue-500 bg-blue-500/10 scale-105",
          isLoading && "pointer-events-none opacity-50",
        )}
        whileHover={{ scale: isDragActive ? 1.05 : 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <input {...getInputProps()} />

        <motion.div animate={isDragActive ? { scale: 1.1 } : { scale: 1 }} className="space-y-6">
          <motion.div
            className="mx-auto w-24 h-24 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center cursor-pointer"
            variants={uploadIconAnimation}
            whileHover="hover"
          >
            <Upload className="w-12 h-12 text-white" />
          </motion.div>

          <div>
            <motion.h3 className="text-2xl font-bold text-white mb-2" whileHover={{ scale: 1.05 }}>
              Drop your photo here
            </motion.h3>
            <motion.p className="text-gray-400 text-lg" whileHover={{ scale: 1.02 }}>
              or click to browse your files
            </motion.p>
          </div>

          <div className="flex items-center justify-center space-x-3">
            <motion.div variants={badgeAnimation} whileHover="hover">
              <Badge variant="secondary" className="bg-gray-800 text-gray-300 border-gray-600">
                JPG
              </Badge>
            </motion.div>
            <motion.div variants={badgeAnimation} whileHover="hover">
              <Badge variant="secondary" className="bg-gray-800 text-gray-300 border-gray-600">
                PNG
              </Badge>
            </motion.div>
            <motion.div variants={badgeAnimation} whileHover="hover">
              <Badge variant="secondary" className="bg-gray-800 text-gray-300 border-gray-600">
                WebP
              </Badge>
            </motion.div>
          </div>

          <motion.p
            className={`text-sm ${apiStatus?.status === "offline" ? "text-red-400" : "text-gray-400"}`}
            animate={{ opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY }}
          >
            {apiStatus?.status === "offline"
              ? "Upload disabled - Backend not available"
              : "Ready to upload and process images"}
          </motion.p>
        </motion.div>
      </motion.div>
    </div>
  )
}
