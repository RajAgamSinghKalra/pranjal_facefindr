"use client"

import { useEffect, useState } from "react"
import { CheckCircle } from "lucide-react"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { Progress } from "@/components/ui/progress"
import { motion, AnimatePresence } from "framer-motion"

interface ProcessingModalProps {
  isOpen: boolean
  stage: string
  onClose: () => void
}

// Jigly animation for the processing icon
const processingIconAnimation = {
  processing: {
    rotate: 360,
    scale: [1, 1.1, 1],
    transition: {
      rotate: { duration: 2, repeat: Number.POSITIVE_INFINITY, ease: "linear" },
      scale: { duration: 1, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" },
    },
  },
  complete: {
    scale: [1, 1.3, 1],
    rotate: [0, 10, -10, 0],
    transition: {
      scale: { duration: 0.6 },
      rotate: { duration: 0.4 },
    },
  },
}

export function ProcessingModal({ isOpen, stage, onClose }: ProcessingModalProps) {
  const [progress, setProgress] = useState(0)
  const [isComplete, setIsComplete] = useState(false)

  useEffect(() => {
    if (isOpen && stage) {
      // Calculate progress based on stage
      const stages = [
        "Uploading image...",
        "Detecting faces...",
        "Extracting features...",
        "Searching database...",
        "Analyzing matches...",
        "Complete!",
      ]

      const currentIndex = stages.indexOf(stage)
      const newProgress = ((currentIndex + 1) / stages.length) * 100

      setProgress(newProgress)
      setIsComplete(stage === "Complete!")

      if (stage === "Complete!") {
        setTimeout(() => {
          onClose()
          setProgress(0)
          setIsComplete(false)
        }, 2000)
      }
    }
  }, [stage, isOpen, onClose])

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="bg-gray-900 border-gray-700 text-white max-w-md">
        <div className="space-y-6 py-4">
          {/* Header with animated icon */}
          <div className="flex items-center space-x-3">
            <motion.div
              className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center"
              variants={processingIconAnimation}
              animate={isComplete ? "complete" : "processing"}
            >
              {isComplete ? (
                <CheckCircle className="w-6 h-6 text-white" />
              ) : (
                <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full" />
              )}
            </motion.div>
            <div>
              <motion.h3
                className="font-semibold text-lg"
                animate={{ scale: isComplete ? [1, 1.05, 1] : 1 }}
                transition={{ duration: 0.3 }}
              >
                Processing your image...
              </motion.h3>
              <motion.p
                className="text-gray-400 text-sm"
                key={stage} // Re-animate when stage changes
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3 }}
              >
                {isComplete ? "Complete!" : stage}
              </motion.p>
            </div>
            <motion.div
              className="ml-auto"
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 0.5, repeat: Number.POSITIVE_INFINITY }}
            >
              <span className="text-2xl font-bold">{Math.round(progress)}%</span>
            </motion.div>
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Processing</span>
              <motion.span
                className="text-gray-400"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 1.5, repeat: Number.POSITIVE_INFINITY }}
              >
                ETA: {isComplete ? "0s" : "2s"}
              </motion.span>
            </div>
            <Progress value={progress} className="h-2 bg-gray-800" />
          </div>

          {/* Completion Message */}
          <AnimatePresence>
            {isComplete && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.9 }}
                animate={{
                  opacity: 1,
                  y: 0,
                  scale: 1,
                  rotate: [0, -2, 2, 0],
                }}
                exit={{ opacity: 0, y: -10 }}
                transition={{
                  opacity: { duration: 0.3 },
                  y: { duration: 0.3 },
                  scale: { duration: 0.3 },
                  rotate: { duration: 0.5 },
                }}
                className="flex items-center justify-center space-x-2 text-green-400"
              >
                <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ duration: 0.6 }}>
                  <CheckCircle className="w-5 h-5" />
                </motion.div>
                <span className="font-medium">Search completed!</span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </DialogContent>
    </Dialog>
  )
}
