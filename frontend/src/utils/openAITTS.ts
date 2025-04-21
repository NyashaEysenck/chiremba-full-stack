// OpenAI API integration for high-quality text-to-speech
import OpenAI from 'openai';

const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY || '';

// Initialize the OpenAI client
const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
  dangerouslyAllowBrowser: true // Required for client-side usage
});

/**
 * Converts text to speech using OpenAI's TTS API
 * @param text Text to convert to speech
 * @returns Audio URL that can be played
 */
export const textToSpeech = async (text: string): Promise<string> => {
  try {
    // Check if API key is available
    if (!OPENAI_API_KEY) {
      console.warn('OpenAI API key not found. Using browser TTS instead.');
      return '';
    }

    console.log('Generating speech with OpenAI...');
    
    // Generate speech using OpenAI SDK
    const response = await openai.audio.speech.create({
      model: "tts-1",
      voice: "shimmer",
      input: text
    });
    
    // Convert the response to an ArrayBuffer
    const arrayBuffer = await response.arrayBuffer();
    
    // Create a Blob from the ArrayBuffer
    const blob = new Blob([arrayBuffer], { type: 'audio/mpeg' });
    
    // Create a URL for the Blob
    const audioUrl = URL.createObjectURL(blob);
    
    console.log('Speech generated successfully');
    return audioUrl;
  } catch (error) {
    console.error('Error converting text to speech:', error);
    return '';
  }
};
