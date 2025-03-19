"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import Image from "next/image"

interface ResultsDisplayProps {
  results: {
    class: string
    confidence: number
    original_image: string
    gradcam_image: string
    result_image: string
  }
}

export default function ResultsDisplay({ results }: ResultsDisplayProps) {
  const [activeImageTab, setActiveImageTab] = useState("result")

  // Format confidence as percentage
  const confidencePercent = (results.confidence * 100).toFixed(2)

  // Determine confidence level for styling
  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 0.8) return "high"
    if (confidence >= 0.5) return "medium"
    return "low"
  }

  const confidenceLevel = getConfidenceLevel(results.confidence)
  const confidenceBadgeVariant =
    confidenceLevel === "high" ? "default" : confidenceLevel === "medium" ? "secondary" : "outline"

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Detection Results</span>
          <Badge variant={confidenceBadgeVariant}>{confidencePercent}% Confidence</Badge>
        </CardTitle>
        <CardDescription>Analysis of the dental image</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2">Diagnosis</h3>
          <div className="p-4 bg-primary/10 rounded-md">
            <p className="text-xl font-bold">{results.class}</p>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-2">Visualization</h3>
          <Tabs value={activeImageTab} onValueChange={setActiveImageTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="result">Combined</TabsTrigger>
              <TabsTrigger value="original">Original</TabsTrigger>
              <TabsTrigger value="gradcam">Heatmap</TabsTrigger>
            </TabsList>

            <TabsContent value="result" className="mt-4">
              <div className="relative w-full h-[300px] md:h-[400px]">
                <Image
                  src={`data:image/png;base64,${results.result_image}`}
                  alt="Result visualization"
                  fill
                  className="object-contain rounded-md"
                />
              </div>
              <p className="text-sm text-gray-500 mt-2">
                Combined visualization showing original image and cavity detection heatmap
              </p>
            </TabsContent>

            <TabsContent value="original" className="mt-4">
              <div className="relative w-full h-[300px] md:h-[400px]">
                <Image
                  src={`data:image/png;base64,${results.original_image}`}
                  alt="Original dental image"
                  fill
                  className="object-contain rounded-md"
                />
              </div>
              <p className="text-sm text-gray-500 mt-2">Original uploaded dental image</p>
            </TabsContent>

            <TabsContent value="gradcam" className="mt-4">
              <div className="relative w-full h-[300px] md:h-[400px]">
                <Image
                  src={`data:image/png;base64,${results.gradcam_image}`}
                  alt="Grad-CAM heatmap"
                  fill
                  className="object-contain rounded-md"
                />
              </div>
              <p className="text-sm text-gray-500 mt-2">
                Grad-CAM heatmap showing areas of interest for cavity detection
              </p>
            </TabsContent>
          </Tabs>
        </div>
      </CardContent>
      <CardFooter>
        <p className="text-sm text-gray-500">
          This is an AI-assisted diagnosis and should be confirmed by a dental professional.
        </p>
      </CardFooter>
    </Card>
  )
}

