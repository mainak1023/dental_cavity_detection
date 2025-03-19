import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function AboutProject() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>About This Project</CardTitle>
        <CardDescription>Learn more about the dental cavity detection system</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="technology">Technology</TabsTrigger>
            <TabsTrigger value="usage">Usage</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4 mt-4">
            <h3 className="text-lg font-semibold">Dental Cavity Detection</h3>
            <p>
              This project uses deep learning to automatically detect dental cavities and other dental conditions from
              images. The system can help dentists with preliminary screening and assist in diagnosis.
            </p>
            <p>
              The AI model has been trained on a dataset of dental images to recognize patterns associated with cavities
              and other dental conditions. It uses convolutional neural networks (CNNs) to analyze the visual features
              of dental images and make predictions.
            </p>
            <h4 className="text-md font-semibold mt-4">Key Features</h4>
            <ul className="list-disc pl-5 space-y-1">
              <li>Automatic detection of dental cavities</li>
              <li>Visual heatmaps highlighting areas of concern</li>
              <li>High accuracy classification</li>
              <li>User-friendly interface</li>
              <li>Fast processing time</li>
            </ul>
          </TabsContent>

          <TabsContent value="technology" className="space-y-4 mt-4">
            <h3 className="text-lg font-semibold">Technology Stack</h3>
            <div className="space-y-4">
              <div>
                <h4 className="text-md font-semibold">Frontend</h4>
                <ul className="list-disc pl-5 space-y-1">
                  <li>Next.js - React framework</li>
                  <li>Tailwind CSS - Styling</li>
                  <li>shadcn/ui - UI components</li>
                </ul>
              </div>

              <div>
                <h4 className="text-md font-semibold">Backend</h4>
                <ul className="list-disc pl-5 space-y-1">
                  <li>FastAPI - Python web framework</li>
                  <li>TensorFlow - Deep learning framework</li>
                  <li>OpenCV - Image processing</li>
                </ul>
              </div>

              <div>
                <h4 className="text-md font-semibold">Machine Learning</h4>
                <ul className="list-disc pl-5 space-y-1">
                  <li>MobileNetV2 - Base architecture</li>
                  <li>Transfer learning and fine-tuning</li>
                  <li>Grad-CAM - Visualization technique</li>
                </ul>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="usage" className="space-y-4 mt-4">
            <h3 className="text-lg font-semibold">How to Use</h3>
            <ol className="list-decimal pl-5 space-y-2">
              <li>
                <strong>Upload an Image:</strong> Click the upload button or drag and drop a dental image onto the
                upload area.
              </li>
              <li>
                <strong>Wait for Processing:</strong> The system will analyze the image using the AI model.
              </li>
              <li>
                <strong>View Results:</strong> The system will display the detected condition and a confidence score.
              </li>
              <li>
                <strong>Examine Visualizations:</strong> View the heatmap to see which areas of the image contributed to
                the diagnosis.
              </li>
            </ol>

            <div className="mt-4">
              <h4 className="text-md font-semibold">Best Practices for Images</h4>
              <ul className="list-disc pl-5 space-y-1">
                <li>Use clear, well-lit images</li>
                <li>Focus on the area of concern</li>
                <li>Avoid blurry or dark images</li>
                <li>Use images taken from a direct angle</li>
              </ul>
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-md mt-4">
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                <strong>Important Note:</strong> This tool is designed to assist dental professionals and should not
                replace professional diagnosis. Always consult with a qualified dentist for proper diagnosis and
                treatment.
              </p>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

