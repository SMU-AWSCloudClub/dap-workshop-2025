import React, { useState } from 'react'
import axios from 'axios'
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const SENTIMENT_API_ENDPOINT = import.meta.env.VITE_SENTIMENT_API_ENDPOINT

function App() {
  const [text, setText] = useState('')
  const [sentiment, setSentiment] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const analyzeSentiment = async () => {
    setLoading(true)
    try {
      const response = await axios.post(SENTIMENT_API_ENDPOINT, { text })
      setSentiment(response.data.sentiment)
    } catch (error) {
      console.error('Error analyzing sentiment:', error)
      setSentiment('Error occurred')
    }
    setLoading(false)
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
            {sentiment && (
              <div className={`text-center font-semibold ${
                sentiment === 'positive' ? 'text-green-600' : 
                sentiment === 'negative' ? 'text-red-600' : 'text-gray-600'
              }`}>
                Sentiment: {sentiment}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default App

