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

  const analyzeSentiment = async () => {
    setLoading(true)
    try {
      const response = await axios.post(SENTIMENT_API_ENDPOINT, { text })
      setSentimentScore(response.data.sentiment) // Expect response in range 0-4
    } catch (error) {
      console.error('Error analyzing sentiment:', error)
      setSentimentScore(null)
    }
    setLoading(false)
  }

  const getSentimentColor = (score: number | null) => {
    if (score === null) return 'text-gray-600'
    switch (score) {
      case 0: return 'text-red-600'   // Very Negative
      case 1: return 'text-orange-500' // Negative
      case 2: return 'text-gray-600'   // Neutral
      case 3: return 'text-blue-500'   // Positive
      case 4: return 'text-green-600'  // Very Positive
      default: return 'text-gray-600'
    }
  }

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
              <div className={`text-center font-semibold ${getSentimentColor(sentimentScore)}`}>
                Sentiment: {sentimentScore === 0 ? 'Very Negative' :
                  sentimentScore === 1 ? 'Negative' :
                  sentimentScore === 2 ? 'Neutral' :
                  sentimentScore === 3 ? 'Positive' : 'Very Positive'}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default App