import { useState } from 'react'
import axios from 'axios'

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [caption, setCaption] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (!file) return
    setSelectedImage(file)
    setPreview(URL.createObjectURL(file))
    setCaption('')
    setError('')
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (!file) return
    setSelectedImage(file)
    setPreview(URL.createObjectURL(file))
    setCaption('')
    setError('')
  }

  const handleCaption = async () => {
    if (!selectedImage) return
    setLoading(true)
    setError('')

    const formData = new FormData()
    formData.append('image', selectedImage)

    try {
      const response = await axios.post('http://localhost:5001/caption', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setCaption(response.data.caption)
    } catch (err) {
      setError('Failed to generate caption. Is the Flask server running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center py-16 px-4">

      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold text-white mb-3">VisionScript</h1>
        <p className="text-gray-400 text-lg">AI-powered image captioning with ResNet + GPT-2</p>
      </div>

      {/* Upload Area */}
      <div className="w-full max-w-2xl">
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="border-2 border-dashed border-gray-700 rounded-2xl p-10 text-center cursor-pointer hover:border-blue-500 transition-colors"
          onClick={() => document.getElementById('fileInput').click()}
        >
          {preview ? (
            <img
              src={preview}
              alt="Preview"
              className="max-h-80 mx-auto rounded-xl object-contain"
            />
          ) : (
            <div className="text-gray-500">
              <p className="text-xl mb-2">Drop an image here</p>
              <p className="text-sm">or click to browse</p>
            </div>
          )}
        </div>

        <input
          id="fileInput"
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="hidden"
        />

        {/* Generate Button */}
        <button
          onClick={handleCaption}
          disabled={!selectedImage || loading}
          className="w-full mt-6 py-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-xl text-white font-semibold text-lg transition-colors"
        >
          {loading ? 'Generating caption...' : 'Generate Caption'}
        </button>

        {/* Caption Output */}
        {caption && (
          <div className="mt-6 p-6 bg-gray-900 rounded-xl border border-gray-800">
            <p className="text-gray-400 text-sm mb-2 uppercase tracking-wider">Generated Caption</p>
            <p className="text-white text-xl font-medium">{caption}</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-6 p-4 bg-red-900/30 border border-red-700 rounded-xl">
            <p className="text-red-400">{error}</p>
          </div>
        )}
      </div>
    </div>
  )
}