// Google Cloud TTS direct REST API integration (for dev/testing only)
// WARNING: Do NOT expose your API key in production!

// Function to fetch config from backend
async function getConfig() {
  const response = await fetch('/api/config');
  return response.json();
}

let GOOGLE_CLOUD_TTS_API_KEY = '';
let GOOGLE_CLOUD_TTS_ENDPOINT = '';

const BASE_URL = import.meta.env.VITE_EXPRESS_API_URL || '';

// Immediately fetch config on module load
getConfig().then(config => {
  GOOGLE_CLOUD_TTS_API_KEY = config.GOOGLE_CLOUD_TTS_API_KEY || '';
  GOOGLE_CLOUD_TTS_ENDPOINT = `https://texttospeech.googleapis.com/v1/text:synthesize?key=${GOOGLE_CLOUD_TTS_API_KEY}`;
});

/**
 * Converts text to speech using Google Cloud TTS API
 * @param text Text to convert to speech
 * @returns Audio URL that can be played
 */
export const textToSpeech = async (text: string): Promise<string> => {
  try {
    const response = await fetch(`${BASE_URL}/api/ai/google/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) throw new Error('Backend Google TTS failed');
    const audioBlob = await response.blob();
    return URL.createObjectURL(audioBlob);
  } catch (error) {
    return '';
  }
};

/**
 * Converts a base64 string to a Blob
 */
function base64ToBlob(base64: string, contentType: string) {
  const byteCharacters = atob(base64);
  const byteArrays = [];
  for (let offset = 0; offset < byteCharacters.length; offset += 512) {
    const slice = byteCharacters.slice(offset, offset + 512);
    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }
  return new Blob(byteArrays, { type: contentType });
}
