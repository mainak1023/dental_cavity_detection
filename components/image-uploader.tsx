"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Upload, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import Image from "next/image"

interface ImageUploaderProps {
  onUpload: (file: File) => void
  isLoading: boolean
}

export default function ImageUploader({ onUpload, isLoading }: ImageUploaderProps) {
  const [preview, setPreview] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      processFile(file)
    }
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()

    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0])
    }
  }

  const processFile = (file: File) => {
    // Check if file is an image
    if (!file.type.match("image.*")) {
      alert("Please upload an image file")
      return
    }

    // Create preview
    const reader = new FileReader()
    reader.onload = () => {
      setPreview(reader.result as string)
    }
    reader.readAsDataURL(file)

    // Pass file to parent component
    onUpload(file)
  }

  const handleButtonClick = () => {
    inputRef.current?.click()
  }

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-6 text-center ${
        dragActive ? "border-primary bg-primary/5" : "border-gray-300 dark:border-gray-700"
      }`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input ref={inputRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />

      {preview ? (
        <div className="flex flex-col items-center">
          <div className="relative w-full max-w-md h-64 mb-4">
            <Image src={preview || "/placeholder.svg"} alt="Preview" fill className="object-contain rounded-md" />
          </div>
          <Button onClick={handleButtonClick} disabled={isLoading} variant="outline">
            Choose Another Image
          </Button>
        </div>
      ) : (
        <div className="flex flex-col items-center">
          <div className="w-16 h-16 mb-4 rounded-full bg-primary/10 flex items-center justify-center">
            <ImageIcon className="h-8 w-8 text-primary" />
          </div>
          <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
            <span className="font-semibold">Click to upload</span> or drag and drop
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400">PNG, JPG (MAX. 10MB)</p>
          <Button onClick={handleButtonClick} disabled={isLoading} className="mt-4">
            <Upload className="mr-2 h-4 w-4" />
            Upload Image
          </Button>
        </div>
      )}
    </div>
  )
}

