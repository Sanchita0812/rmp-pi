'use client'
import { Box, Button, Stack, TextField } from '@mui/material'
import { useState } from 'react'

export default function Home() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: `Hi! I'm the Rate My Professor support assistant. How can I help you today?`,
    },
  ])
  const [message, setMessage] = useState('')

  const sendMessage = async () => {
    if (message.trim() === '') return // Prevent sending empty messages

    // Update UI optimistically
    const newMessages = [
      ...messages,
      { role: 'user', content: message },
      { role: 'assistant', content: '' },
    ]
    setMessages(newMessages)
    setMessage('')

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify([...newMessages]),
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let result = ''

      reader.read().then(function processText({ done, value }) {
        if (done) {
          return result
        }

        const text = decoder.decode(value || new Uint8Array(), { stream: true })
        setMessages((prevMessages) => {
          let lastMessage = prevMessages[prevMessages.length - 1]
          let otherMessages = prevMessages.slice(0, prevMessages.length - 1)
          return [
            ...otherMessages,
            { ...lastMessage, content: lastMessage.content + text },
          ]
        })

        return reader.read().then(processText)
      })
    } catch (error) {
      console.error('Error fetching chat response:', error)
      // Handle error by showing a message or resetting the state
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          role: 'assistant',
          content: 'Sorry, there was an error processing your request.',
        },
      ])
    }
  }

  return (
    <Box
      width="100vw"
      height="100vh"
      display="flex"
      flexDirection="column"
      justifyContent="center"
      alignItems="center"
    >
      <Stack
        direction={'column'}
        width="500px"
        height="700px"
        border="1px solid black"
        p={2}
        spacing={3}
      >
        <Stack
          direction={'column'}
          spacing={2}
          flexGrow={1}
          overflow="auto"
          maxHeight="100%"
        >
          {messages.map((message, index) => (
            <Box
              key={index}
              display="flex"
              justifyContent={
                message.role === 'assistant' ? 'flex-start' : 'flex-end'
              }
            >
              <Box
                bgcolor={
                  message.role === 'assistant'
                    ? 'primary.main'
                    : 'secondary.main'
                }
                color="white"
                borderRadius={16}
                p={3}
              >
                {message.content}
              </Box>
            </Box>
          ))}
        </Stack>
        <Stack direction={'row'} spacing={2}>
          <TextField
            label="Message"
            fullWidth
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter') sendMessage() // Optional: Send message on Enter key
            }}
          />
          <Button variant="contained" onClick={sendMessage}>
            Send
          </Button>
        </Stack>
      </Stack>
    </Box>
  )
}
