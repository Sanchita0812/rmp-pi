import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import { Groq } from 'groq-sdk'  // Adjust import according to the actual Groq SDK

const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
Use them to answer the question if needed.
`

export async function POST(req) {
    const data = await req.json();
    
    if (!data || data.length === 0) {
        return new NextResponse('No data provided', { status: 400 });
    }

    // Initialize Pinecone
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    });
    const index = pc.index('rag').namespace('ns1');

    // Initialize Groq client
    const groqClient = new Groq({
        apiKey: process.env.GROQ_API_KEY,
    });

    // Extract the last user message
    const text = data[data.length - 1].content;

    if (!text || text.trim() === '') {
        return new NextResponse('Message content is empty', { status: 400 });
    }

    // Create embedding using Groq API
    let embedding;
    try {
        const embeddingResponse = await groqClient.embeddings.create({
            input: [text],  // Ensure `input` is an array
            model: 'text-embedding-3-small',
            encoding_format: 'float',
        });

        embedding = embeddingResponse.data[0].embedding;
    } catch (error) {
        return new NextResponse(`Embedding creation failed: ${error.message}`, { status: 500 });
    }

    // Query Pinecone index
    let results;
    try {
        results = await index.query({
            topK: 5,
            includeMetadata: true,
            vector: embedding,
        });
    } catch (error) {
        return new NextResponse(`Pinecone query failed: ${error.message}`, { status: 500 });
    }

    let resultString = '';
    results.matches.forEach((match) => {
        resultString += `
        Returned Results:
        Professor: ${match.id}
        Review: ${match.metadata.review}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n`;
    });

    const lastMessage = data[data.length - 1];
    const lastMessageContent = lastMessage.content + resultString;
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

    // Create completion using Groq API
    let completion;
    try {
        completion = await groqClient.completions.create({
            messages: [
                { role: 'system', content: systemPrompt },
                ...lastDataWithoutLastMessage,
                { role: 'user', content: lastMessageContent },
            ],
            model: 'llama3-70b-8192',  // Ensure correct model identifier
            stream: true,
        });
    } catch (error) {
        return new NextResponse(`Completion creation failed: ${error.message}`, { status: 500 });
    }

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder();
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content;
                    if (content) {
                        const text = encoder.encode(content);
                        controller.enqueue(text);
                    }
                }
            } catch (err) {
                controller.error(err);
            } finally {
                controller.close();
            }
        },
    });

    return new NextResponse(stream);
}
