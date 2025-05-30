import { toast } from "@/hooks/use-toast";

// Chat history storage
const chatHistories = new Map<string, any>();

// Proxy Google Generative AI (Gemini) to backend
export async function generateAIResponse(prompt: string, chatId?: string): Promise<string> {
  try {
    const response = await fetch('/api/ai/googleai/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    const data = await response.json();
    if (data.content) return data.content;
    return 'Sorry, there was an error generating a response.';
  } catch (error) {
    return 'Sorry, there was an error generating a response.';
  }
}

export async function analyzeHealthSymptoms(symptoms: string): Promise<any> {
  try {
    const prompt = `As a medical assistant, analyze these symptoms and provide a structured response with the following sections:
1. Possible conditions (list the top 3 most likely conditions based on the symptoms. Try to write the most common name for the condition if it is available)
2. Suggested medications or treatments for each condition
4. When to seek professional medical help
5. Disclaimer

User symptoms: ${symptoms}

IMPORTANT: Begin each section with EXACTLY these headings:
- "Possible conditions"
- "Suggested medications"
- "Urgency level"
- "When to seek professional medical help"
- "DISCLAIMER"

Format your response for easy reading with numbered lists. Include a disclaimer that this is not a substitute for professional medical advice. DO NOT use asterisk formatting like ** or * around any text or quotes "" around the text or any weird formatting`;

    const response = await generateAIResponse(prompt);
    console.log("Raw symptom analysis response:", response);
    
    return {
      analysisText: response,
      success: true
    };
  } catch (error) {
    console.error("Error analyzing health symptoms:", error);
    return {
      analysisText: "Sorry, there was an error analyzing your symptoms. Please try again later.",
      success: false
    };
  }
}

// Start or continue a chat session with memory
export async function startOrContinueChat(chatId: string): Promise<void> {
  if (!chatHistories.has(chatId)) {
    const messages = [];
    chatHistories.set(chatId, messages);
  }
}

export async function getHealthChatResponse(userMessage: string, chatId: string, language: string = "english"): Promise<string> {
  try {
    await startOrContinueChat(chatId);
    const messages = chatHistories.get(chatId);

    const systemPrompt = `You are Chiremba, a friendly and knowledgeable health assistant. \nRespond to the following health-related question or comment in a helpful, informative, and compassionate way.\nIf the query suggests a serious medical condition, advise the user to seek professional medical help.\n\nUse appropriate formatting with line breaks to improve readability.\nDO NOT use asterisks (**) in your response to highlight text or for formatting.\n\nIMPORTANT: All responses must be in ${language.toUpperCase()} language.`;

    if (messages.length === 0) {
      messages.push({ role: "system", content: systemPrompt });
    }

    messages.push({ role: "user", content: userMessage });

    const prompt = messages.map((message: any) => `${message.role}: ${message.content}`).join('\n');
    const response = await generateAIResponse(prompt);
    const text = response;
    messages.push({ role: "assistant", content: text });

    return text;
  } catch (error) {
    console.error("Error getting health chat response:", error);
    return "I'm sorry, I'm having trouble responding right now. Please try again in a moment.";
  }
}

// Function to clear chat history for a specific chat
export function clearChatHistory(chatId: string): void {
  if (chatHistories.has(chatId)) {
    chatHistories.delete(chatId);
  }
}

// Placeholder function for enhanced text-to-speech (to be implemented with ElevenLabs)
export async function getEnhancedTextToSpeech(text: string): Promise<void> {
  // This function will be replaced by the ElevenLabs implementation
  console.log("Enhanced TTS would process:", text);
}

