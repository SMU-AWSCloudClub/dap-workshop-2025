import React, { useState } from 'react'
import axios from 'axios'
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const SENTIMENT_API_BASE = import.meta.env.VITE_SENTIMENT_API_ENDPOINT;
const SENTIMENT_API_ENDPOINT = `http://${SENTIMENT_API_BASE}:8000/predict`;

function App() {
  const [text, setText] = useState('')
  const [sentimentScore, setSentimentScore] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault()
      analyzeSentiment()
    }
  }

  const analyzeSentiment = async () => {
    setLoading(true)
    try {
      const response = await axios.post(SENTIMENT_API_ENDPOINT, { text })
      setSentimentScore(response.data.prediction) // Expect response in range 0-4
    } catch (error) {
      console.error('Error analyzing sentiment:', error)
      setSentimentScore(null)
    }
    setLoading(false)
  }

  const getSentimentLabel = (score: number) => {
    switch (score) {
      case 0: return 'Very Negative'
      case 1: return 'Negative'
      case 2: return 'Neutral'
      case 3: return 'Positive'
      case 4: return 'Very Positive'
      default: return 'Neutral'
    }
  }

  const getSentimentColor = (score: number | null) => {
    if (score === null) return 'text-gray-600'
    switch (score) {
      case 0: return 'text-red-700'
      case 1: return 'text-orange-600'
      case 2: return 'text-gray-700'
      case 3: return 'text-sky-500'
      case 4: return 'text-green-700'
      default: return 'text-gray-600'
    }
  }

  const markerLeft = sentimentScore !== null ? sentimentScore * 25 : 0;

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl font-bold">Sentiment Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter text to analyze..."
              className="w-full h-32 resize-none"
            />
            <Button 
              onClick={analyzeSentiment} 
              disabled={!text || loading}
              className="w-full"
            >
              {loading ? 'Analyzing...' : 'Analyze Sentiment'}
            </Button>
            {sentimentScore !== null && (
              <>
                <div className={`text-center font-semibold ${getSentimentColor(sentimentScore)}`}>
                  Sentiment: {getSentimentLabel(sentimentScore)}
                </div>
                {/* Shortened gradient scale and labels container */}
                <div className="relative mt-6 w-4/5 mx-auto">
                  {/* Gradient scale bar */}
                  <div
                    className="h-2 rounded-full"
                    style={{
                      background: 'linear-gradient(to right, #b91c1c, #ea580c, #4b5563, #0ea5e9, #16a34a)'
                    }}
                  ></div>
                  {/* Arrow marker positioned above the bar */}
                  <div
                    className="absolute"
                    style={{ top: '-14px', left: `calc(${markerLeft}% - 8px)` }}
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                      className={getSentimentColor(sentimentScore)}
                    >
                      {/* Downward-pointing arrow */}
                      <polygon points="4,6 20,6 12,14" fill="currentColor" />
                    </svg>
                  </div>
                </div>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default App