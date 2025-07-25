"use client"

import { useState } from "react"
import { CheckCircle, Copy, ExternalLink, Terminal, AlertCircle, Play, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { motion } from "framer-motion"

interface BackendSetupGuideProps {
  isOpen: boolean
  onClose: () => void
}

const copyToClipboard = (text: string) => {
  navigator.clipboard.writeText(text)
}

const setupSteps = [
  {
    id: "prerequisites",
    title: "Prerequisites",
    items: ["Python 3.8 or higher", "Node.js 18 or higher", "PostgreSQL 12 or higher", "Git (for cloning repository)"],
  },
  {
    id: "backend",
    title: "Backend Setup",
    commands: [
      {
        description: "Navigate to backend directory",
        command: "cd Identiface/backend",
      },
      {
        description: "Create Python virtual environment",
        command: "python3 -m venv venv",
      },
      {
        description: "Activate virtual environment",
        command: "source venv/bin/activate  # On Windows: venv\\Scripts\\activate",
      },
      {
        description: "Install Python dependencies",
        command: "pip install -r requirements.txt",
      },
      {
        description: "Start the FastAPI server",
        command: "uvicorn main:app --host 0.0.0.0 --port 8000 --reload",
      },
    ],
  },
  {
    id: "database",
    title: "Database Setup",
    commands: [
      {
        description: "Create PostgreSQL database",
        command: "createdb face_recognition_db",
      },
      {
        description: "Run database setup scripts",
        command: "psql -d face_recognition_db -f ../01_setup_extensions_and_tables.sql",
      },
      {
        description: "Create indexes",
        command: "psql -d face_recognition_db -f ../02_create_indexes.sql",
      },
      {
        description: "Create functions and triggers",
        command: "psql -d face_recognition_db -f ../03_create_functions_and_triggers.sql",
      },
    ],
  },
]

const troubleshootingSteps = [
  {
    issue: "Port 8000 already in use",
    solution: "Kill existing process: `lsof -ti:8000 | xargs kill -9` or use different port: `--port 8001`",
  },
  {
    issue: "Python dependencies fail to install",
    solution: "Update pip: `pip install --upgrade pip` and try installing dependencies individually",
  },
  {
    issue: "PostgreSQL connection failed",
    solution:
      "Check if PostgreSQL is running: `brew services start postgresql` (macOS) or `sudo systemctl start postgresql` (Linux)",
  },
  {
    issue: "Permission denied errors",
    solution: "Make sure you have write permissions in the project directory and virtual environment",
  },
]

export function BackendSetupGuide({ isOpen, onClose }: BackendSetupGuideProps) {
  const [copiedCommand, setCopiedCommand] = useState<string | null>(null)

  const handleCopy = (command: string) => {
    copyToClipboard(command)
    setCopiedCommand(command)
    setTimeout(() => setCopiedCommand(null), 2000)
  }

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
        className="bg-gray-900 border border-gray-700 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-white mb-2">Backend Setup Guide</h2>
              <p className="text-gray-400">Get your IdentiFace backend server running in minutes</p>
            </div>
            <Button variant="ghost" onClick={onClose} className="text-gray-400 hover:text-white">
              ✕
            </Button>
          </div>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
          <Tabs defaultValue="setup" className="w-full">
            <TabsList className="grid w-full grid-cols-3 bg-gray-800">
              <TabsTrigger value="setup" className="text-white">
                Setup Steps
              </TabsTrigger>
              <TabsTrigger value="troubleshooting" className="text-white">
                Troubleshooting
              </TabsTrigger>
              <TabsTrigger value="quick-start" className="text-white">
                Quick Start
              </TabsTrigger>
            </TabsList>

            <TabsContent value="setup" className="space-y-6 mt-6">
              {setupSteps.map((step, stepIndex) => (
                <motion.div
                  key={step.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: stepIndex * 0.1 }}
                >
                  <Card className="bg-gray-800 border-gray-700">
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2 text-white">
                        <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-sm font-bold">
                          {stepIndex + 1}
                        </div>
                        <span>{step.title}</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {step.items && (
                        <div className="grid grid-cols-2 gap-3">
                          {step.items.map((item, index) => (
                            <div key={index} className="flex items-center space-x-2">
                              <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0" />
                              <span className="text-gray-300 text-sm">{item}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      {step.commands && (
                        <div className="space-y-3">
                          {step.commands.map((cmd, index) => (
                            <div key={index} className="space-y-2">
                              <p className="text-gray-400 text-sm">{cmd.description}</p>
                              <div className="relative">
                                <div className="bg-gray-900 border border-gray-600 rounded-lg p-3 font-mono text-sm text-green-400 flex items-center justify-between">
                                  <code className="flex-1">{cmd.command}</code>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleCopy(cmd.command)}
                                    className="ml-2 text-gray-400 hover:text-white"
                                  >
                                    {copiedCommand === cmd.command ? (
                                      <CheckCircle className="w-4 h-4 text-green-400" />
                                    ) : (
                                      <Copy className="w-4 h-4" />
                                    )}
                                  </Button>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </TabsContent>

            <TabsContent value="troubleshooting" className="space-y-4 mt-6">
              <Alert className="border-yellow-600 bg-yellow-900/20">
                <AlertCircle className="h-4 w-4 text-yellow-400" />
                <AlertDescription className="text-yellow-300">
                  Common issues and their solutions. If you're still having trouble, check the console logs for more
                  details.
                </AlertDescription>
              </Alert>

              {troubleshootingSteps.map((item, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card className="bg-gray-800 border-gray-700">
                    <CardContent className="p-4">
                      <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <AlertCircle className="w-4 h-4 text-red-400" />
                          <h4 className="font-semibold text-white">{item.issue}</h4>
                        </div>
                        <div className="bg-gray-900 border border-gray-600 rounded p-3">
                          <code className="text-green-400 text-sm">{item.solution}</code>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleCopy(item.solution)}
                            className="ml-2 text-gray-400 hover:text-white"
                          >
                            <Copy className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </TabsContent>

            <TabsContent value="quick-start" className="space-y-6 mt-6">
              <Alert className="border-green-600 bg-green-900/20">
                <Play className="h-4 w-4 text-green-400" />
                <AlertDescription className="text-green-300">
                  Use the automated setup script for the fastest way to get started.
                </AlertDescription>
              </Alert>

              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white flex items-center space-x-2">
                    <Terminal className="w-5 h-5" />
                    <span>One-Command Setup</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-gray-400">
                    Run this single command from the project root to set up everything automatically:
                  </p>
                  <div className="bg-gray-900 border border-gray-600 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <code className="text-green-400 font-mono">chmod +x start-dev.sh && ./start-dev.sh</code>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleCopy("chmod +x start-dev.sh && ./start-dev.sh")}
                        className="text-gray-400 hover:text-white"
                      >
                        <Copy className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-semibold text-white">This script will:</h4>
                    <ul className="space-y-1 text-gray-300 text-sm">
                      <li className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-400" />
                        <span>Check for required dependencies</span>
                      </li>
                      <li className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-400" />
                        <span>Set up Python virtual environment</span>
                      </li>
                      <li className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-400" />
                        <span>Install all dependencies</span>
                      </li>
                      <li className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-400" />
                        <span>Start both frontend and backend servers</span>
                      </li>
                    </ul>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle className="text-white flex items-center space-x-2">
                    <Download className="w-5 h-5" />
                    <span>Manual Download</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-gray-400">If you don't have the project files yet:</p>
                  <div className="bg-gray-900 border border-gray-600 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <code className="text-green-400 font-mono">
                        git clone &lt;repository-url&gt; && cd Identiface
                      </code>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleCopy("git clone <repository-url> && cd Identiface")}
                        className="text-gray-400 hover:text-white"
                      >
                        <Copy className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          <div className="mt-8 p-4 bg-blue-900/20 border border-blue-600 rounded-lg">
            <div className="flex items-start space-x-3">
              <ExternalLink className="w-5 h-5 text-blue-400 mt-0.5" />
              <div>
                <h4 className="font-semibold text-blue-300 mb-1">Need Help?</h4>
                <p className="text-blue-200 text-sm">
                  Check the console logs for detailed error messages, or visit our documentation for more advanced setup
                  options.
                </p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}
