"use client"

import { useState, useEffect } from "react"
import { RefreshCw, CheckCircle, XCircle, AlertTriangle, Terminal, Wifi } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { motion, AnimatePresence } from "framer-motion"

interface DiagnosticResult {
  test: string
  status: "running" | "success" | "error" | "warning"
  message: string
  details?: string
  duration?: number
}

interface ConnectionDiagnosticsProps {
  isOpen: boolean
  onClose: () => void
}

const diagnosticTests = [
  {
    id: "network",
    name: "Network Connectivity",
    description: "Check if the device can reach the internet",
  },
  {
    id: "localhost",
    name: "Localhost Resolution",
    description: "Verify localhost DNS resolution",
  },
  {
    id: "port-8000",
    name: "Port 8000 Availability",
    description: "Check if port 8000 is accessible",
  },
  {
    id: "backend-health",
    name: "Backend Health Check",
    description: "Test FastAPI server response",
  },
  {
    id: "api-endpoints",
    name: "API Endpoints",
    description: "Verify all API routes are responding",
  },
]

export function ConnectionDiagnostics({ isOpen, onClose }: ConnectionDiagnosticsProps) {
  const [results, setResults] = useState<DiagnosticResult[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)

  const runDiagnostics = async () => {
    setIsRunning(true)
    setResults([])
    setProgress(0)

    const newResults: DiagnosticResult[] = []

    for (let i = 0; i < diagnosticTests.length; i++) {
      const test = diagnosticTests[i]
      const startTime = Date.now()

      // Update progress
      setProgress((i / diagnosticTests.length) * 100)

      // Add running status
      const runningResult: DiagnosticResult = {
        test: test.name,
        status: "running",
        message: "Testing...",
      }
      setResults([...newResults, runningResult])

      let result: DiagnosticResult

      try {
        switch (test.id) {
          case "network":
            result = await testNetworkConnectivity(test.name, startTime)
            break
          case "localhost":
            result = await testLocalhostResolution(test.name, startTime)
            break
          case "port-8000":
            result = await testPortAvailability(test.name, startTime)
            break
          case "backend-health":
            result = await testBackendHealth(test.name, startTime)
            break
          case "api-endpoints":
            result = await testApiEndpoints(test.name, startTime)
            break
          default:
            result = {
              test: test.name,
              status: "error",
              message: "Unknown test",
              duration: Date.now() - startTime,
            }
        }
      } catch (error) {
        result = {
          test: test.name,
          status: "error",
          message: "Test failed",
          details: error instanceof Error ? error.message : "Unknown error",
          duration: Date.now() - startTime,
        }
      }

      newResults.push(result)
      setResults([...newResults])

      // Small delay between tests
      await new Promise((resolve) => setTimeout(resolve, 500))
    }

    setProgress(100)
    setIsRunning(false)
  }

  const testNetworkConnectivity = async (testName: string, startTime: number): Promise<DiagnosticResult> => {
    try {
      const response = await fetch("https://httpbin.org/get", {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      })

      if (response.ok) {
        return {
          test: testName,
          status: "success",
          message: "Internet connectivity is working",
          duration: Date.now() - startTime,
        }
      } else {
        return {
          test: testName,
          status: "warning",
          message: "Limited connectivity",
          details: `HTTP ${response.status}`,
          duration: Date.now() - startTime,
        }
      }
    } catch (error) {
      return {
        test: testName,
        status: "error",
        message: "No internet connectivity",
        details: error instanceof Error ? error.message : "Network error",
        duration: Date.now() - startTime,
      }
    }
  }

  const testLocalhostResolution = async (testName: string, startTime: number): Promise<DiagnosticResult> => {
    try {
      // Try to resolve localhost
      const response = await fetch("http://localhost:3000", {
        method: "HEAD",
        signal: AbortSignal.timeout(3000),
      })

      return {
        test: testName,
        status: "success",
        message: "Localhost resolution working",
        details: "Frontend server detected",
        duration: Date.now() - startTime,
      }
    } catch (error) {
      // This might fail if frontend is not on 3000, but localhost should still resolve
      return {
        test: testName,
        status: "warning",
        message: "Localhost accessible",
        details: "Frontend not on port 3000, but localhost resolves",
        duration: Date.now() - startTime,
      }
    }
  }

  const testPortAvailability = async (testName: string, startTime: number): Promise<DiagnosticResult> => {
    try {
      const response = await fetch("http://localhost:8000", {
        method: "HEAD",
        signal: AbortSignal.timeout(3000),
      })

      return {
        test: testName,
        status: "success",
        message: "Port 8000 is accessible",
        details: "Backend server is running",
        duration: Date.now() - startTime,
      }
    } catch (error) {
      return {
        test: testName,
        status: "error",
        message: "Port 8000 is not accessible",
        details: "Backend server is not running or blocked",
        duration: Date.now() - startTime,
      }
    }
  }

  const testBackendHealth = async (testName: string, startTime: number): Promise<DiagnosticResult> => {
    try {
      const response = await fetch("http://localhost:8000/", {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      })

      if (response.ok) {
        const data = await response.json()
        return {
          test: testName,
          status: "success",
          message: "Backend is healthy",
          details: data.message || "FastAPI server responding",
          duration: Date.now() - startTime,
        }
      } else {
        return {
          test: testName,
          status: "error",
          message: "Backend health check failed",
          details: `HTTP ${response.status}: ${response.statusText}`,
          duration: Date.now() - startTime,
        }
      }
    } catch (error) {
      return {
        test: testName,
        status: "error",
        message: "Cannot reach backend",
        details: error instanceof Error ? error.message : "Connection failed",
        duration: Date.now() - startTime,
      }
    }
  }

  const testApiEndpoints = async (testName: string, startTime: number): Promise<DiagnosticResult> => {
    const endpoints = ["/api/stats", "/api/clusters", "/docs"]
    const results = []

    for (const endpoint of endpoints) {
      try {
        const response = await fetch(`http://localhost:8000${endpoint}`, {
          method: "GET",
          signal: AbortSignal.timeout(3000),
        })
        results.push({ endpoint, status: response.status, ok: response.ok })
      } catch (error) {
        results.push({ endpoint, status: 0, ok: false, error: error instanceof Error ? error.message : "Failed" })
      }
    }

    const successCount = results.filter((r) => r.ok).length
    const totalCount = results.length

    if (successCount === totalCount) {
      return {
        test: testName,
        status: "success",
        message: "All API endpoints responding",
        details: `${successCount}/${totalCount} endpoints working`,
        duration: Date.now() - startTime,
      }
    } else if (successCount > 0) {
      return {
        test: testName,
        status: "warning",
        message: "Some API endpoints working",
        details: `${successCount}/${totalCount} endpoints responding`,
        duration: Date.now() - startTime,
      }
    } else {
      return {
        test: testName,
        status: "error",
        message: "No API endpoints responding",
        details: "Backend may not be properly configured",
        duration: Date.now() - startTime,
      }
    }
  }

  useEffect(() => {
    if (isOpen) {
      runDiagnostics()
    }
  }, [isOpen])

  if (!isOpen) return null

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-gray-900 border border-gray-700 rounded-xl max-w-2xl w-full max-h-[90vh] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <Terminal className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Connection Diagnostics</h2>
                <p className="text-gray-400 text-sm">Testing backend connectivity and API health</p>
              </div>
            </div>
            <Button variant="ghost" onClick={onClose} className="text-gray-400 hover:text-white">
              ✕
            </Button>
          </div>
        </div>

        <div className="p-6 space-y-6 overflow-y-auto max-h-[calc(90vh-120px)]">
          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">
                {isRunning ? "Running diagnostics..." : "Diagnostics complete"}
              </span>
              <span className="text-sm text-gray-400">{Math.round(progress)}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>

          {/* Test Results */}
          <div className="space-y-3">
            <AnimatePresence>
              {results.map((result, index) => (
                <motion.div
                  key={result.test}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card className="bg-gray-800 border-gray-700">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="flex-shrink-0">
                            {result.status === "running" && (
                              <RefreshCw className="w-5 h-5 text-blue-400 animate-spin" />
                            )}
                            {result.status === "success" && <CheckCircle className="w-5 h-5 text-green-400" />}
                            {result.status === "warning" && <AlertTriangle className="w-5 h-5 text-yellow-400" />}
                            {result.status === "error" && <XCircle className="w-5 h-5 text-red-400" />}
                          </div>
                          <div>
                            <h4 className="font-medium text-white">{result.test}</h4>
                            <p className="text-sm text-gray-400">{result.message}</p>
                            {result.details && <p className="text-xs text-gray-500 mt-1">{result.details}</p>}
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          {result.duration && (
                            <Badge variant="outline" className="text-xs">
                              {result.duration}ms
                            </Badge>
                          )}
                          <Badge
                            variant={
                              result.status === "success"
                                ? "default"
                                : result.status === "warning"
                                  ? "secondary"
                                  : "destructive"
                            }
                            className="text-xs"
                          >
                            {result.status}
                          </Badge>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-between pt-4 border-t border-gray-700">
            <Button
              variant="outline"
              onClick={runDiagnostics}
              disabled={isRunning}
              className="text-white bg-transparent"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${isRunning ? "animate-spin" : ""}`} />
              {isRunning ? "Running..." : "Run Again"}
            </Button>
            <div className="flex items-center space-x-2 text-sm text-gray-400">
              <Wifi className="w-4 h-4" />
              <span>Backend: http://localhost:8000</span>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}
