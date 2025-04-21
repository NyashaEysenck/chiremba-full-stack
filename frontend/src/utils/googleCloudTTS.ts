// Google Cloud TTS direct REST API integration (for dev/testing only)
// WARNING: Do NOT expose your API key in production!

const GOOGLE_CLOUD_TTS_API_KEY = import.meta.env.VITE_GOOGLE_CLOUD_TTS_API_KEY || '';
const GOOGLE_CLOUD_TTS_ENDPOINT = `https://texttospeech.googleapis.com/v1/text:synthesize?key=${GOOGLE_CLOUD_TTS_API_KEY}`;

/**
 * Converts text to speech using Google Cloud TTS API
 * @param text Text to convert to speech
 * @returns Audio URL that can be played
 */
export const textToSpeech = async (text: string): Promise<string> => {
  try {
    if (!GOOGLE_CLOUD_TTS_API_KEY) {
      console.warn('Google Cloud TTS API key not found.');
      return '';
    }
    if (!text) {
      console.warn('No text provided for TTS.');
      return '';
    }

    console.log('Generating speech with Google Cloud TTS...');

    const requestBody = {
      input: { text },
      voice: {
        languageCode: 'en-US', // You can change this
        name: 'en-US-Wavenet-D', // You can change this
      },
      audioConfig: {
        audioEncoding: 'MP3',
      },
    };

    const response = await fetch(GOOGLE_CLOUD_TTS_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error('Failed to generate speech with Google Cloud TTS');
    }

    const data = await response.json();
    if (!data.audioContent) {
      throw new Error('No audio content received from Google Cloud TTS');
    }

    // audioContent is a base64-encoded string
    const audioBlob = base64ToBlob(data.audioContent, 'audio/mp3');
    const audioUrl = URL.createObjectURL(audioBlob);
    console.log('Speech generated successfully (Google Cloud)');
    return audioUrl;
  } catch (error) {
    console.error('Error converting text to speech (Google Cloud):', error);
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
