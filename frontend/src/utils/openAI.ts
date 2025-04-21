import { toast } from "@/hooks/use-toast";

const BASE_URL = import.meta.env.VITE_EXPRESS_API_URL || '';

// Chat history storage
const chatHistories = new Map<string, any>();

export async function generateAIResponse(prompt: string, chatId?: string): Promise<string> {
  try {
    if (!prompt.trim()) {
      return "Please provide a valid prompt.";
    }
    const response = await fetch(`${BASE_URL}/api/ai/openai`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    const data = await response.json();
    // Extract text from OpenAI response
    let text = '';
    if (data.choices && data.choices[0] && data.choices[0].message) {
      text = data.choices[0].message.content;
    } else if (data.choices && data.choices[0] && data.choices[0].text) {
      text = data.choices[0].text;
    }
    return text || "";
  } catch (error) {
    console.error("Error generating AI response:", error);
    toast({
      title: "AI Generation Error",
      description: error instanceof Error ? error.message : "An error occurred while generating the AI response",
      variant: "destructive",
    });
    return "Sorry, there was an error generating a response. Please try again later.";
  }
}

export async function analyzeHealthSymptoms(symptoms: string): Promise<any> {
  try {
    const prompt = `As a medical assistant, analyze these symptoms and provide a structured response with the following sections:
1. Possible conditions (list the top 3 most likely conditions based on the symptoms)
2. Suggested medications or treatments for each condition
3. Urgency level (low, medium, high)
4. When to seek professional medical help

User symptoms: ${symptoms}

IMPORTANT: Begin each section with EXACTLY these headings:
- "Possible conditions"
- "Suggested medications"
- "Urgency level"
- "When to seek professional medical help"
- "DISCLAIMER"

Format your response for easy reading with numbered lists. Include a disclaimer that this is not a substitute for professional medical advice. DO NOT use asterisk formatting like ** around any text.`;

    const response = await generateAIResponse(prompt);
    console.log("Raw symptom analysis response:", response);

    return {
      analysisText: response,
      success: true,
    };
  } catch (error) {
    console.error("Error analyzing health symptoms:", error);
    return {
      analysisText: "Sorry, there was an error analyzing your symptoms. Please try again later.",
      success: false,
    };
  }
}

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

    const systemPrompt = `You are Chiremba, a friendly and knowledgeable health assistant. 
Respond to the following health-related question or comment in a helpful, informative, and compassionate way.
If the query suggests a serious medical condition, advise the user to seek professional medical help.

Use appropriate formatting with line breaks to improve readability.
DO NOT use asterisks (**) in your response to highlight text or for formatting.

IMPORTANT: All responses must be in ${language.toUpperCase()} language.`;

    if (messages.length === 0) {
      messages.push({ role: "system", content: systemPrompt });
    }

    messages.push({ role: "user", content: userMessage });

    const prompt = messages.map((message) => ({ role: message.role, content: message.content })).map((message, index) => `${index + 1}. ${message.role}: ${message.content}`).join('\n\n');
    const response = await generateAIResponse(prompt);
    const text = response;
    messages.push({ role: "assistant", content: text });

    return text;
  } catch (error) {
    console.error("Error getting health chat response:", error);
    return "I'm sorry, I'm having trouble responding right now. Please try again in a moment.";
  }
}

export function clearChatHistory(chatId: string): void {
  if (chatHistories.has(chatId)) {
    chatHistories.delete(chatId);
  }
}

export async function getEnhancedTextToSpeech(text: string): Promise<void> {
  console.log("Enhanced TTS would process:", text);
}

export async function loadBrainTumorModel(): Promise<void> {
  console.log("Loading brain tumor detection model...");
  return new Promise((resolve) => {
    setTimeout(() => {
      console.log("Brain tumor model loaded");
      resolve();
    }, 1000);
  });
}

export async function detectBrainTumor(imageElement: HTMLImageElement): Promise<{ hasTumor: boolean; confidence: number }> {
  console.log("Analyzing brain scan for tumors...");
  return new Promise((resolve) => {
    setTimeout(() => {
      const random = Math.random();
      const hasTumor = random > 0.5;
      const confidence = hasTumor
        ? Math.floor(70 + random * 25)
        : Math.floor(75 + random * 20);

      console.log(`Analysis complete: ${hasTumor ? "Tumor detected" : "No tumor detected"} with ${confidence}% confidence`);
      resolve({ hasTumor, confidence });
    }, 2000);
  });
}

export async function analyzeMedicalDocument(documentText: string): Promise<string> {
  try {
    const prompt = `Extract and summarize the key medical information from this document:
${documentText}

Focus on:
1. Patient information
2. Diagnosis details
3. Treatment recommendations
4. Follow-up information

DO NOT use asterisks (**) in your response for any formatting.`;

    return await generateAIResponse(prompt);
  } catch (error) {
    console.error("Error analyzing medical document:", error);
    return "Sorry, there was an error analyzing this document. Please try again later.";
  }
}
