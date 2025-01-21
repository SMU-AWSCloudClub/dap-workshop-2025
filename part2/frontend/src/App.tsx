import React, { useState } from 'react'
import axios from 'axios'
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const LLM_API_ENDPOINT = import.meta.env.VITE_LLM_API_ENDPOINT

function App() {
  const [text, setText] = useState('')
  const [response, setResponse] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const queryLLM = async () => {
    setLoading(true)
    setResponse(null)
    try {
      const payload = {
        inputs: text,
        parameters: {
          max_new_tokens: 64,
          top_p: 0.9,
          temperature: 0.6,
          details: true,
        }
      }
      const res = await axios.post(LLM_API_ENDPOINT, payload)
      setResponse(res.data.generated_text)
    } catch (error) {
      console.error('Error querying LLM:', error)
      setResponse('Error occurred while querying the model.')
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl font-bold">LLM Query</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter your query..."
              className="w-full h-32 resize-none"
            />
            <Button
              onClick={queryLLM}
              disabled={!text || loading}
              className="w-full"
            >
              {loading ? 'Processing...' : 'Query LLM'}
            </Button>
            {response && (
              <div className="mt-4 p-4 bg-gray-50 rounded border">
                <strong>Response:</strong>
                <p>{response}</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default App