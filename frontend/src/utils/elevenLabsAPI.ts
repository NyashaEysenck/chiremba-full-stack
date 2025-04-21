// ElevenLabs API integration for high-quality text-to-speech

const ELEVEN_LABS_VOICE_ID = 'tnSpp4vdxKPjI9w0GnoV'; // Sarah voice

/**
 * Converts text to speech using ElevenLabs API
 * @param text Text to convert to speech
 * @returns Audio URL that can be played
 */
export const textToSpeech = async (text: string): Promise<string> => {
  try {
    const response = await fetch('/api/ai/elevenlabs/tts', {
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
