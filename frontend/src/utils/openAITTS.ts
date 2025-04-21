// OpenAI API integration for high-quality text-to-speech
import OpenAI from 'openai';

// Function to fetch config from backend
async function getConfig() {
  const response = await fetch('/api/config');
  return response.json();
}

let OPENAI_API_KEY = '';

// Immediately fetch config on module load
getConfig().then(config => {
  OPENAI_API_KEY = config.OPENAI_API_KEY || '';
});

// Initialize the OpenAI client
const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
  dangerouslyAllowBrowser: true // Required for client-side usage
});

const BASE_URL = import.meta.env.VITE_EXPRESS_API_URL || '';

/**
 * Converts text to speech using OpenAI's TTS API
 * @param text Text to convert to speech
 * @returns Audio URL that can be played
 */
export const textToSpeech = async (text: string): Promise<string> => {
  try {
    const response = await fetch(`${BASE_URL}/api/ai/openai/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) throw new Error('Backend TTS failed');
    const audioBlob = await response.blob();
    return URL.createObjectURL(audioBlob);
  } catch (error) {
    return '';
  }
};
