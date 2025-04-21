// Google Cloud TTS utility now calls backend endpoint
// WARNING: Do NOT expose your API key in production!

const GOOGLE_CLOUD_TTS_ENDPOINT = '/api/tts/google';

/**
 * Converts text to speech using Google Cloud TTS via backend
 * @param text Text to convert to speech
 * @returns Audio URL that can be played
 */
export const textToSpeech = async (text: string): Promise<string> => {
  try {
    if (!text) {
      console.warn('No text provided for TTS.');
      return '';
    }
    const response = await fetch(GOOGLE_CLOUD_TTS_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      throw new Error('Failed to generate speech with Google Cloud TTS');
    }
    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    return audioUrl;
  } catch (error) {
    console.error('Error converting text to speech (Google Cloud):', error);
    return '';
  }
};
