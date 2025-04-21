
// ElevenLabs API integration for high-quality text-to-speech
let ELEVEN_LABS_API_KEY = '';

// Function to fetch config from backend
async function getConfig() {
  const response = await fetch('/api/config');
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
    // Check if API key is available
    if (!ELEVEN_LABS_API_KEY) {
      throw new Error('Eleven Labs API key is not set.');
    }

    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${ELEVEN_LABS_VOICE_ID}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'xi-api-key': ELEVEN_LABS_API_KEY,
        },
        body: JSON.stringify({
          text,
          model_id: 'eleven_multilingual_v2',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75,
            style: 0.5,
            use_speaker_boost: true,
          },
        }),
      }
    );

    if (!response.ok) {
      const errorData = await response.json();
      console.error('ElevenLabs API error:', errorData);
      return '';
    }

    // Get audio blob and create URL
    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    return audioUrl;
  } catch (error) {
    console.error('Error generating speech with ElevenLabs:', error);
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
