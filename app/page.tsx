"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import ImageUploader from "@/components/image-uploader"
import ResultsDisplay from "@/components/results-display"
import AboutProject from "@/components/about-project"

export default function Home() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<any>(null)
  const [activeTab, setActiveTab] = useState("upload")

  const handleUpload = async (file: File) => {
    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()
      setResults(data)
      setActiveTab("results")
    } catch (err: any) {
      console.error("Error uploading image:", err)
      setError(err.message || "Failed to process image. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-4 md:p-24">
      <div className="z-10 max-w-5xl w-full">
        <h1 className="text-4xl font-bold text-center mb-6">Dental Cavity Detection</h1>
        <p className="text-center text-gray-500 dark:text-gray-400 mb-8">
          Upload dental images to detect cavities using machine learning
        </p>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="upload">Upload</TabsTrigger>
            <TabsTrigger value="results" disabled={!results}>
              Results
            </TabsTrigger>
            <TabsTrigger value="about">About</TabsTrigger>
          </TabsList>

          <TabsContent value="upload">
            <Card>
              <CardHeader>
                <CardTitle>Upload Dental Image</CardTitle>
                <CardDescription>Upload a clear image of teeth to analyze for cavities</CardDescription>
              </CardHeader>
              <CardContent>
                <ImageUploader onUpload={handleUpload} isLoading={isLoading} />

                {isLoading && (
                  <div className="mt-4">
                    <p className="text-sm text-gray-500 mb-2">Processing image...</p>
                    <Progress value={45} className="h-2" />
                  </div>
                )}

                {error && (
                  <Alert variant="destructive" className="mt-4">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}
              </CardContent>
              <CardFooter className="flex justify-between">
                <p className="text-sm text-gray-500">Supported formats: JPG, PNG</p>
              </CardFooter>
            </Card>
          </TabsContent>

          <TabsContent value="results">{results && <ResultsDisplay results={results} />}</TabsContent>

          <TabsContent value="about">
            <AboutProject />
          </TabsContent>
        </Tabs>
      </div>
    </main>
  )
}

