import { toast } from "@/hooks/use-toast";

// Calls backend Gemini endpoint
export async function generateAIResponse(prompt: string, chatId?: string): Promise<string> {
  try {
    if (!prompt.trim()) {
      return "Please provide a valid prompt.";
    }
    const response = await fetch('/api/ai/google', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    const data = await response.json();
    // Gemini API returns a nested structure; extract text
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || data.text || "";
    return text;
  } catch (error) {
    console.error("Error generating AI response:", error);
    toast({
      title: "AI Generation Error",
      description: error instanceof Error ? error.message : "An error occurred while generating the AI response",
      variant: "destructive"
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
  // No-op for now, as we're using the backend endpoint for Gemini calls
}

export async function getHealthChatResponse(userMessage: string, chatId: string, language: string = "english"): Promise<string> {
  try {
    // Ensure a chat session exists (not applicable with backend endpoint)
    // await startOrContinueChat(chatId);
    
    // Get the chat session (not applicable with backend endpoint)
    // const chat = chatHistories.get(chatId);
    // if (!chat) {
    //   throw new Error("Chat session not found");
    // }
    
    // Use the stored chat to maintain context (not applicable with backend endpoint)
    // const systemPrompt = `You are Chiremba, a friendly and knowledgeable health assistant. 
    // Respond to the following health-related question or comment in a helpful, informative, and compassionate way.
    // If the query suggests a serious medical condition, advise the user to seek professional medical help.
    
    // Use appropriate formatting with line breaks to improve readability.
    // DO NOT use asterisks (**) in your response to highlight text or for formatting.
    
    // IMPORTANT: All responses must be in ${language.toUpperCase()} language.`;
    
    // Add system prompt if this is a new conversation (not applicable with backend endpoint)
    // if (!chat.history || chat.history.length === 0) {
    //   await chat.sendMessage(systemPrompt);
    // }
    
    // Send the user message and get response
    const prompt = `You are Chiremba, a friendly and knowledgeable health assistant. 
    Respond to the following health-related question or comment in a helpful, informative, and compassionate way.
    If the query suggests a serious medical condition, advise the user to seek professional medical help.
    
    Use appropriate formatting with line breaks to improve readability.
    DO NOT use asterisks (**) in your response to highlight text or for formatting.
    
    IMPORTANT: All responses must be in ${language.toUpperCase()} language.

    User message: ${userMessage}`;
    const result = await generateAIResponse(prompt);
    return result;
  } catch (error) {
    console.error("Error getting health chat response:", error);
    return "I'm sorry, I'm having trouble responding right now. Please try again in a moment.";
  }
}

// Function to clear chat history for a specific chat
export function clearChatHistory(chatId: string): void {
  // No-op for now, as we're using the backend endpoint for Gemini calls
}

// Placeholder function for enhanced text-to-speech (to be implemented with ElevenLabs)
export async function getEnhancedTextToSpeech(text: string): Promise<void> {
  // This function will be replaced by the ElevenLabs implementation
  console.log("Enhanced TTS would process:", text);
}

// Function to load the brain tumor model (mockup for now)
export async function loadBrainTumorModel(): Promise<void> {
  console.log("Loading brain tumor detection model...");
  // This would actually load a TensorFlow.js model in a real implementation
  return new Promise((resolve) => {
    setTimeout(() => {
      console.log("Brain tumor model loaded");
      resolve();
    }, 1000);
  });
}

// Function to detect brain tumors (mockup for now)
export async function detectBrainTumor(imageElement: HTMLImageElement): Promise<{ hasTumor: boolean; confidence: number }> {
  console.log("Analyzing brain scan for tumors...");
  // This would use a real ML model in a production implementation
  // For now, we'll return a random result for demonstration purposes
  
  return new Promise((resolve) => {
    setTimeout(() => {
      // Randomly decide if a tumor is detected (for demo purposes)
      const random = Math.random();
      const hasTumor = random > 0.5;
      const confidence = hasTumor ? 
        Math.floor(70 + random * 25) : // 70-95% confidence if tumor detected
        Math.floor(75 + random * 20);  // 75-95% confidence if no tumor
      
      console.log(`Analysis complete: ${hasTumor ? 'Tumor detected' : 'No tumor detected'} with ${confidence}% confidence`);
      resolve({ hasTumor, confidence });
    }, 2000);
  });
}

// New function to analyze medical documents or images (will be implemented in future)
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
