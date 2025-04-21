import { toast } from "@/hooks/use-toast";

let GOOGLE_API_KEY = '';

// Function to fetch config from backend
async function getConfig() {
  const response = await fetch('/api/config');
  return response.json();
}

// Immediately fetch config on module load
getConfig().then(config => {
  GOOGLE_API_KEY = config.GOOGLE_API_KEY || '';
});

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

export async function getHealthChatResponse(userMessage: string, chatId: string, language: string = "english"): Promise<string> {
  try {
    // Ensure a chat session exists (no-op for now)
    // await startOrContinueChat(chatId);
    
    // Get the chat session (no-op for now)
    // const chat = chatHistories.get(chatId);
    // if (!chat) {
    //   throw new Error("Chat session not found");
    // }
    
    // Use the stored chat to maintain context (no-op for now)
    // const systemPrompt = `You are Chiremba, a friendly and knowledgeable health assistant. \nRespond to the following health-related question or comment in a helpful, informative, and compassionate way.\nIf the query suggests a serious medical condition, advise the user to seek professional medical help.\n\nUse appropriate formatting with line breaks to improve readability.\nDO NOT use asterisks (**) in your response to highlight text or for formatting.\n\nIMPORTANT: All responses must be in ${language.toUpperCase()} language.\n\nUser message: ${userMessage}`;
    const prompt = `You are Chiremba, a friendly and knowledgeable health assistant. \nRespond to the following health-related question or comment in a helpful, informative, and compassionate way.\nIf the query suggests a serious medical condition, advise the user to seek professional medical help.\n\nUse appropriate formatting with line breaks to improve readability.\nDO NOT use asterisks (**) in your response to highlight text or for formatting.\n\nIMPORTANT: All responses must be in ${language.toUpperCase()} language.\n\nUser message: ${userMessage}`;
    const result = await generateAIResponse(prompt);
    return result;
  } catch (error) {
    console.error("Error getting health chat response:", error);
    return "I'm sorry, I'm having trouble responding right now. Please try again in a moment.";
  }
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

export async function analyzeHealthSymptoms(symptoms: string): Promise<any> {
  try {
    const prompt = `As a medical assistant, analyze these symptoms and provide a structured response with the following sections:\n1. Possible conditions (list the top 3 most likely conditions based on the symptoms)\n2. Suggested medications or treatments for each condition\n3. Urgency level (low, medium, high)\n4. When to seek professional medical help\n\nUser symptoms: ${symptoms}\n\nIMPORTANT: Begin each section with EXACTLY these headings:\n- "Possible conditions"\n- "Suggested medications"\n- "Urgency level"\n\nFormat your response for easy reading with numbered lists. Include a disclaimer that this is not a substitute for professional medical advice. DO NOT use asterisk formatting like ** around any text.`;

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

// Function to clear chat history for a specific chat
export function clearChatHistory(chatId: string): void {
  // No-op for now, as we're proxying to the backend
}

// Start or continue a chat session with memory
export async function startOrContinueChat(chatId: string): Promise<void> {
  // No-op for now, as we're proxying to the backend
}

// New function to analyze medical documents or images (will be implemented in future)
export async function analyzeMedicalDocument(documentText: string): Promise<string> {
  try {
    const prompt = `Extract and summarize the key medical information from this document:\n    ${documentText}\n    \n    Focus on:\n    1. Patient information\n    2. Diagnosis details\n    3. Treatment recommendations\n    4. Follow-up information\n    \n    DO NOT use asterisks (**) in your response for any formatting.`;

    return await generateAIResponse(prompt);
  } catch (error) {
    console.error("Error analyzing medical document:", error);
    return "Sorry, there was an error analyzing this document. Please try again later.";
  }
}

// Placeholder function for enhanced text-to-speech (to be implemented with ElevenLabs)
export async function getEnhancedTextToSpeech(text: string): Promise<void> {
  // This function will be replaced by the ElevenLabs implementation
  console.log("Enhanced TTS would process:", text);
}
