// ElevenLabs TTS utility now calls backend endpoint
const BASE_URL = import.meta.env.VITE_EXPRESS_API_URL || '';
const ELEVEN_LABS_VOICE_ID = 'tnSpp4vdxKPjI9w0GnoV'; // Sarah voice

/**
 * Converts text to speech using ElevenLabs API via backend
 * @param text Text to convert to speech
 * @returns Audio URL that can be played
 */
export const textToSpeech = async (text: string): Promise<string> => {
  try {
    if (!text) {
      console.warn('No text provided for TTS.');
      return '';
    }
    const response = await fetch(`${BASE_URL}/api/tts/elevenlabs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      throw new Error('Failed to generate speech with ElevenLabs TTS');
    }
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
