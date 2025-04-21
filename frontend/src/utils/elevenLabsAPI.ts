// ElevenLabs API integration for high-quality text-to-speech
let ELEVEN_LABS_API_KEY = '';

const BASE_URL = import.meta.env.VITE_EXPRESS_API_URL || '';

// Function to fetch config from backend
async function getConfig() {
  const response = await fetch(`${BASE_URL}/api/config`);
  return response.json();
}

// Immediately fetch config on module load
getConfig().then(config => {
  ELEVEN_LABS_API_KEY = config.ELEVEN_LABS_API_KEY || '';
});

const ELEVEN_LABS_VOICE_ID = 'tnSpp4vdxKPjI9w0GnoV'; // Sarah voice

/**
 * Converts text to speech using ElevenLabs API
 * @param text Text to convert to speech
 * @returns Audio URL that can be played
 */
export const textToSpeech = async (text: string): Promise<string> => {
  try {
    const response = await fetch(`${BASE_URL}/api/ai/elevenlabs/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) throw new Error('Backend ElevenLabs TTS failed');
    const audioBlob = await response.blob();
    return URL.createObjectURL(audioBlob);
  } catch (error) {
    return '';
  }
};

/**
 * Play audio from URL with volume control
 * @param audioUrl URL of audio to play
 * @returns Audio element that's playing
 */
export const playAudio = (audioUrl: string): HTMLAudioElement | null => {
  if (!audioUrl) return null;
  
  const audio = new Audio(audioUrl);
  audio.volume = 1.0;
  audio.play().catch((error) => {
    console.error('Error playing audio:', error);
  });
  
  return audio;
};
