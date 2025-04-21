// OpenAI TTS utility now calls backend endpoint
const BASE_URL = import.meta.env.VITE_EXPRESS_API_URL || '';

/**
 * Converts text to speech using OpenAI's TTS API via backend
 * @param text Text to convert to speech
 * @returns Audio URL that can be played
 */
export const textToSpeech = async (text: string): Promise<string> => {
  try {
    if (!text) {
      console.warn('No text provided for TTS.');
      return '';
    }
    const response = await fetch(`${BASE_URL}/api/tts/openai`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      throw new Error('Failed to generate speech with OpenAI TTS');
    }
    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    return audioUrl;
  } catch (error) {
    console.error('Error converting text to speech (OpenAI):', error);
    return '';
  }
};
