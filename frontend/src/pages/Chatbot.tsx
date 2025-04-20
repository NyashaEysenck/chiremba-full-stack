import { useState, useRef, useEffect } from 'react';
import { useToast } from '@/hooks/use-toast';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { 
  SendHorizonal, 
  Mic, 
  MicOff, 
  Loader2, 
  Clock, 
  Bot, 
  User,
  Globe,
  Pause,
  Play,
  Volume2,
  VolumeX,
  Settings
} from 'lucide-react';
import { getHealthChatResponse, startOrContinueChat, clearChatHistory } from '@/utils/googleAI';
import { Switch } from "@/components/ui/switch"; 
import { textToSpeech } from '@/utils/elevenLabsAPI';
import { v4 as uuidv4 } from 'uuid';
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

type Message = {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
};

const initialMessages: Message[] = [
  {
    role: 'assistant',
    content: 'Hello, I am Chiremba, your virtual health assistant. How can I help you today?',
    timestamp: new Date(),
  },
];

// Shona initial message
const initialMessagesShona: Message[] = [
  {
    role: 'assistant',
    content: 'Mhoro, ndini Chiremba, mushandi wenyu weutano wekudzivirira. Ndingakubatsirei sei nhasi?',
    timestamp: new Date(),
  },
];

const Chatbot = () => {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isShona, setIsShona] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [autoplayEnabled, setAutoplayEnabled] = useState(false);
  const [useEnhancedVoice, setUseEnhancedVoice] = useState(true);
  const [currentAudio, setCurrentAudio] = useState<HTMLAudioElement | null>(null);
  const [currentMessageId, setCurrentMessageId] = useState<number | null>(null);
  const [chatId, setChatId] = useState<string>(uuidv4());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();
  
  // Scroll to top when page loads
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);
  
  // Initialize the chat session
  useEffect(() => {
    startOrContinueChat(chatId);
    
    return () => {
      // Cleanup when component unmounts
      if (currentAudio) {
        currentAudio.pause();
        URL.revokeObjectURL(currentAudio.src);
      }
    };
  }, [chatId]);
  
  // Update messages when language changes
  useEffect(() => {
    setMessages(isShona ? initialMessagesShona : initialMessages);
    // Generate a new chat ID to restart the conversation
    setChatId(uuidv4());
  }, [isShona]);
  
  // Speech recognition setup
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  
  useEffect(() => {
    // Check if browser supports SpeechRecognition
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognitionAPI) {
        recognitionRef.current = new SpeechRecognitionAPI();
        recognitionRef.current.continuous = true;
        recognitionRef.current.interimResults = true;
        
        recognitionRef.current.onresult = (event: SpeechRecognitionEvent) => {
          const transcript = Array.from(event.results)
            .map(result => result[0])
            .map(result => result.transcript)
            .join('');
          
          setInput(transcript);
        };
        
        // Explicitly type the event as SpeechRecognitionErrorEvent
        recognitionRef.current.onerror = (event: SpeechRecognitionErrorEvent) => {
          console.error('Speech recognition error', event.error);
          setIsListening(false);
          toast({
            title: 'Voice Input Error',
            description: 'There was a problem with voice recognition. Please try again.',
            variant: 'destructive',
          });
        };
      }
    }
    
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      
      // Stop any playing audio when component unmounts
      if (currentAudio) {
        currentAudio.pause();
        URL.revokeObjectURL(currentAudio.src);
        setCurrentAudio(null);
      }
    };
  }, [toast]);
  
  const toggleListening = () => {
    if (!recognitionRef.current) {
      toast({
        title: 'Voice Input Not Supported',
        description: 'Your browser does not support voice input. Please type your message instead.',
        variant: 'destructive',
      });
      return;
    }
    
    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      recognitionRef.current.start();
      setIsListening(true);
      toast({
        title: 'Voice Input Active',
        description: 'Speak now. Your words will appear in the input field.',
      });
    }
  };
  
  // Auto scroll to bottom on new messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Toggle mute state
  const toggleMute = () => {
    setIsMuted(!isMuted);
    if (isSpeaking && currentAudio) {
      stopSpeech();
    }
  };

  // Toggle enhanced voice option
  const toggleEnhancedVoice = () => {
    setUseEnhancedVoice(!useEnhancedVoice);
    
    // Show toast to notify the user
    toast({
      title: useEnhancedVoice ? 'Using Standard Voice' : 'Using Enhanced Voice',
      description: useEnhancedVoice 
        ? 'Switched to standard browser voice.' 
        : 'Switched to premium OpenAI voice.',
    });
    
    // Stop current speech if any
    if (isSpeaking) {
      stopSpeech();
    }
  };

  // Enhanced text to speech function with OpenAI
  const speakText = async (text: string, messageIndex: number) => {
    if (isMuted) return;
    
    // Stop any currently playing audio
    stopSpeech();
    
    setCurrentMessageId(messageIndex);
    
    try {
      setIsSpeaking(true);
      
      let audioUrl = '';
      
      if (useEnhancedVoice) {
        // Use OpenAI TTS
        audioUrl = await textToSpeech(text);
      }
      
      if (audioUrl) {
        // Play the enhanced voice
        const audio = new Audio(audioUrl);
        setCurrentAudio(audio);
        
        audio.onended = () => {
          setIsSpeaking(false);
          setCurrentMessageId(null);
          setCurrentAudio(null);
          URL.revokeObjectURL(audioUrl);
        };
        
        audio.onerror = () => {
          console.error('Audio playback error');
          setIsSpeaking(false);
          setCurrentMessageId(null);
          setCurrentAudio(null);
          URL.revokeObjectURL(audioUrl);
        };
        
        audio.play();
      } else {
        // Fall back to browser TTS if enhanced voice fails or is disabled
        useBrowserTTS(text);
      }
    } catch (error) {
      console.error('Error speaking text:', error);
      toast({
        title: 'Voice Error',
        description: 'There was a problem with the text-to-speech service. Falling back to browser TTS.',
        variant: 'destructive',
      });
      
      // Fall back to browser TTS
      useBrowserTTS(text);
    }
  };
  
  // Browser TTS fallback
  const useBrowserTTS = (text: string) => {
    const speech = new SpeechSynthesisUtterance(text);
    speech.volume = 1;
    speech.rate = 1;
    speech.pitch = 1;
    
    // Set language based on isShona flag
    speech.lang = isShona ? 'en-US' : 'en-US';
    
    // Use a better voice if available
    const voices = window.speechSynthesis.getVoices();
    const preferredVoices = ['Google UK English Female', 'Microsoft Zira', 'Samantha'];
    
    for (const name of preferredVoices) {
      const voice = voices.find(v => v.name === name);
      if (voice) {
        speech.voice = voice;
        break;
      }
    }
    
    // Start speaking
    window.speechSynthesis.speak(speech);
    
    // Event handlers
    speech.onend = () => {
      setIsSpeaking(false);
      setCurrentMessageId(null);
    };
    
    speech.onerror = () => {
      console.error("Browser TTS error");
      setIsSpeaking(false);
      setCurrentMessageId(null);
    };
  };
  
  // Pause speech
  const pauseSpeech = () => {
    if (currentAudio) {
      currentAudio.pause();
      setIsPaused(true);
    } else {
      window.speechSynthesis.pause();
      setIsPaused(true);
    }
  };
  
  // Resume speech
  const resumeSpeech = () => {
    if (currentAudio) {
      currentAudio.play().catch(err => console.error("Error resuming audio:", err));
      setIsPaused(false);
    } else {
      window.speechSynthesis.resume();
      setIsPaused(false);
    }
  };
  
  // Safely stop speech without causing errors
  const stopSpeech = () => {
    if (currentAudio) {
      // For OpenAI audio
      currentAudio.pause();
      setCurrentAudio(null);
    } else {
      // For native speech synthesis
      if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
      }
    }
    setIsSpeaking(false);
    setIsPaused(false);
    setCurrentMessageId(null);
  };
  
  // Reset chat conversation
  const resetChat = () => {
    // Clear the existing conversation
    stopSpeech();
    clearChatHistory(chatId);
    
    // Create a new chat ID
    const newChatId = uuidv4();
    setChatId(newChatId);
    
    // Initialize new conversation
    startOrContinueChat(newChatId);
    
    // Reset messages
    setMessages(isShona ? initialMessagesShona : initialMessages);
    
    toast({
      title: 'Conversation Reset',
      description: 'Starting a new conversation with Chiremba.',
    });
  };
  
  const handleSendMessage = async () => {
    if (!input.trim()) return;
    
    // Stop listening if active
    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
    
    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    
    try {
      // Modify prompt to include language preference
      const language = isShona ? "shona" : "english";
      
      // Use enhanced Google AI API to get response with conversation context
      const aiResponse = await getHealthChatResponse(input, chatId, language);
      
      const botMessage: Message = {
        role: 'assistant',
        content: aiResponse,
        timestamp: new Date(),
      };
      
      // Update messages with the bot's response
      setMessages(prev => [...prev, botMessage]);
      
      // Speak the response if not muted and autoplay is enabled
      if (!isMuted && autoplayEnabled) {
        // Give a small delay to ensure UI is updated
        setTimeout(() => {
          speakText(aiResponse, messages.length + 1);
        }, 100);
      }
      
    } catch (error) {
      console.error("Error getting AI response:", error);
      
      // Fallback response
      const fallbackMessage: Message = {
        role: 'assistant',
        content: isShona 
          ? "Ndinonzwa kuti ndiri kusangana nematambudziko ekubatanidza neruzivo rwangu. Ndapota edza zvakare munguva shoma."
          : "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again in a moment.",
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, fallbackMessage]);
      
      toast({
        title: 'Connection Error',
        description: 'There was a problem getting a response. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  
  // Toggle language
  const toggleLanguage = () => {
    setIsShona(!isShona);
    
    // Show toast notification about language change
    toast({
      title: isShona ? 'Switching to English' : 'Kushandura kuenda kuShona',
      description: isShona 
        ? 'The assistant will now respond in English' 
        : 'Chiremba achapindura nechiShona',
    });
  };
  
  // Format message content to HTML with proper styling - removing asterisks instead of converting to bold
  const formatMessageContent = (content: string) => {
    return content
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove asterisks instead of converting to bold
      .replace(/\n\n/g, '<br /><br />')
      .replace(/\n/g, '<br />');
  };
  
  return (
    <div className="min-h-screen flex flex-col pattern-bg">
      <Navbar />
      
      <div className="flex-grow flex justify-center px-4 pt-24 pb-20">
        <div className="w-full max-w-3xl">
          <div className="glassmorphism rounded-2xl overflow-hidden flex flex-col h-[75vh] shadow-lg">
            {/* Chat header */}
            <div className="bg-white bg-opacity-90 backdrop-blur-sm p-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="h-10 w-10 rounded-full bg-gradient-to-r from-terracotta to-ochre flex items-center justify-center shadow-md">
                    <Bot className="h-6 w-6 text-white" />
                  </div>
                  <div className="ml-3">
                    <h3 className="text-lg font-semibold text-gray-900">Chiremba Health Assistant</h3>
                    <p className="text-sm text-gray-600">Virtual Consultation</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <button 
                    onClick={toggleMute} 
                    className="p-1 rounded-full hover:bg-gray-100"
                    title={isMuted ? "Enable voice" : "Mute voice"}
                  >
                    {isMuted ? (
                      <VolumeX className="h-4 w-4 text-gray-500" />
                    ) : (
                      <Volume2 className="h-4 w-4 text-terracotta" />
                    )}
                  </button>
                  
                  <Popover>
                    <PopoverTrigger asChild>
                      <button className="p-1 rounded-full hover:bg-gray-100" title="Voice settings">
                        <Settings className="h-4 w-4 text-gray-500" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="w-80">
                      <div className="space-y-4">
                        <h4 className="font-medium">Voice Settings</h4>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-700">Autoplay Responses</span>
                          <Switch
                            checked={autoplayEnabled}
                            onCheckedChange={() => setAutoplayEnabled(!autoplayEnabled)}
                            className="data-[state=checked]:bg-terracotta"
                          />
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-700">Use Enhanced Voice</span>
                          <Switch
                            checked={useEnhancedVoice}
                            onCheckedChange={toggleEnhancedVoice}
                            className="data-[state=checked]:bg-terracotta"
                          />
                        </div>
                        <p className="text-xs text-gray-500">
                          Enhanced voice uses OpenAI for more natural speech. Autoplay must be enabled for automatic responses.
                          {!import.meta.env.VITE_OPENAI_API_KEY && 
                            " Add your API key in the .env.local file to enable this feature."}
                        </p>
                        <div className="flex items-center justify-between">
                          <button 
                            onClick={resetChat} 
                            className="text-xs bg-red-50 hover:bg-red-100 text-red-600 px-2 py-1 rounded"
                          >
                            Reset Conversation
                          </button>
                        </div>
                      </div>
                    </PopoverContent>
                  </Popover>
                  
                  <Globe className="h-4 w-4 text-gray-500" />
                  <span className="text-sm text-gray-600 mr-2">
                    {isShona ? 'Shona' : 'English'}
                  </span>
                  <Switch
                    checked={isShona}
                    onCheckedChange={toggleLanguage}
                    className="data-[state=checked]:bg-terracotta"
                  />
                </div>
              </div>
            </div>
            
            {/* Chat messages with improved styling */}
            <div className="flex-grow overflow-y-auto p-4 bg-gray-50 bg-opacity-70">
              <div className="space-y-4">
                {messages.map((message, index) => (
                  <div 
                    key={index}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div 
                      className={`max-w-[80%] rounded-2xl p-4 ${
                        message.role === 'user' 
                          ? 'bg-gradient-to-r from-terracotta to-ochre text-white shadow-md' 
                          : 'bg-white border border-gray-200 shadow-sm'
                      }`}
                    >
                      <div className="flex items-center mb-2">
                        <div 
                          className={`h-6 w-6 rounded-full flex items-center justify-center mr-2 ${
                            message.role === 'user' 
                              ? 'bg-white bg-opacity-20' 
                              : 'bg-terracotta bg-opacity-10'
                          }`}
                        >
                          {message.role === 'user' 
                            ? <User className={`h-3 w-3 text-white`} /> 
                            : <Bot className={`h-3 w-3 text-terracotta`} />
                          }
                        </div>
                        <span 
                          className={`text-xs font-medium ${
                            message.role === 'user' ? 'text-white text-opacity-90' : 'text-gray-600'
                          }`}
                        >
                          {message.role === 'user' ? 'You' : 'Chiremba'}
                        </span>
                        <span 
                          className={`ml-auto text-xs ${
                            message.role === 'user' ? 'text-white text-opacity-75' : 'text-gray-400'
                          }`}
                        >
                          {formatTime(message.timestamp)}
                        </span>
                      </div>
                      <div 
                        className={`prose max-w-none ${
                          message.role === 'user' 
                            ? 'text-white prose-headings:text-white prose-strong:text-white text-sm leading-relaxed' 
                            : 'text-gray-800 text-sm leading-relaxed'
                        }`}
                        dangerouslySetInnerHTML={{ 
                          __html: formatMessageContent(message.content)
                        }}
                      />
                      {message.role === 'assistant' && (
                        <div className="mt-2 flex items-center text-xs text-gray-500">
                          {currentMessageId === index && isSpeaking ? (
                            <div className="flex items-center space-x-2">
                              {isPaused ? (
                                <button 
                                  onClick={resumeSpeech}
                                  className="flex items-center text-terracotta hover:text-terracotta/80"
                                >
                                  <Play className="h-3 w-3 mr-1" />
                                  <span>Resume</span>
                                </button>
                              ) : (
                                <button 
                                  onClick={pauseSpeech}
                                  className="flex items-center text-terracotta hover:text-terracotta/80"
                                >
                                  <Pause className="h-3 w-3 mr-1" />
                                  <span>Pause</span>
                                </button>
                              )}
                              <button 
                                onClick={stopSpeech}
                                className="flex items-center text-gray-500 hover:text-gray-700 ml-2"
                              >
                                <span>Stop</span>
                              </button>
                              <span className="flex items-center ml-2">
                                <span className="relative flex h-2 w-2 mr-1">
                                  <span className={`absolute inline-flex h-full w-full rounded-full bg-terracotta opacity-75 ${isPaused ? '' : 'animate-ping'}`}></span>
                                  <span className="relative inline-flex rounded-full h-2 w-2 bg-terracotta"></span>
                                </span>
                                {isPaused ? 'Paused' : 'Speaking...'}
                              </span>
                            </div>
                          ) : (
                            <button 
                              onClick={() => !isMuted && speakText(message.content, index)}
                              className={`flex items-center ${isMuted ? 'text-gray-400 cursor-not-allowed' : 'hover:text-terracotta'}`}
                              disabled={isMuted}
                            >
                              <Play className="h-3 w-3 mr-1" />
                              <span>Listen</span>
                              {isMuted && <span className="ml-1">(Voice disabled)</span>}
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="max-w-[80%] rounded-2xl p-4 bg-white border border-gray-200 shadow-sm">
                      <div className="flex items-center mb-1">
                        <div className="h-6 w-6 rounded-full bg-terracotta bg-opacity-10 flex items-center justify-center mr-2">
                          <Bot className="h-3 w-3 text-terracotta" />
                        </div>
                        <span className="text-xs font-medium text-gray-600">Chiremba</span>
                        <Clock className="ml-auto h-3 w-3 text-gray-400" />
                      </div>
                      <div className="flex items-center">
                        <Loader2 className="h-4 w-4 text-terracotta animate-spin" />
                        <span className="ml-2 text-gray-600">Thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
            </div>
            
            {/* Chat input */}
            <div className="p-4 bg-white bg-opacity-90 backdrop-blur-sm">
              <div className="flex items-end gap-2">
                <button
                  onClick={toggleListening}
                  className={`p-2 rounded-full ${
                    isListening 
                      ? 'bg-red-500 text-white animate-pulse shadow-md' 
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  } transition-colors`}
                  title={isListening ? 'Stop voice input' : 'Start voice input'}
                >
                  {isListening ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
                </button>
                <div className="flex-grow">
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={isShona ? "Nyora mibvunzo yako pano..." : "Type your health concern or question..."}
                    className="w-full resize-none rounded-lg border border-gray-300 focus:border-terracotta focus:ring-1 focus:ring-terracotta p-3 max-h-32"
                    rows={1}
                    disabled={isLoading}
                  />
                </div>
                <button
                  onClick={handleSendMessage}
                  disabled={!input.trim() || isLoading}
                  className={`p-3 rounded-full ${
                    input.trim() && !isLoading
                      ? 'bg-gradient-to-r from-terracotta to-ochre text-white shadow-md'
                      : 'bg-gray-200 text-gray-400'
                  } transition-colors`}
                  title="Send message"
                >
                  <SendHorizonal className="h-5 w-5" />
                </button>
              </div>
              <div className="mt-2 text-xs text-gray-500 text-center px-4">
                <p>
                  <strong>Note:</strong> This is an AI health assistant. 
                  For medical emergencies, please contact emergency services immediately.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
};

export default Chatbot;
