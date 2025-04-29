import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/components/AuthContext';
import { useToast } from '@/hooks/use-toast';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import ReactCrop, { Crop, PixelCrop } from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';
import { 
  Upload, 
  X, 
  Image as ImageIcon, 
  Loader2, 
  AlertCircle, 
  Check,
  Info,
  Camera,
  Crop as CropIcon,
  RotateCw
} from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch"; 
import { analyzeImage as analyzeImageAPI } from '@/utils/imageAnalysisService';
import { CONDITION_DETAILS } from '@/utils/imageAnalysisService';
import { requestLungCancerGradCAM } from '@/utils/gradcamService';
import { downloadReport, printReport } from '@/utils/reportGenerator';
import { generateAIResponse } from '@/utils/googleAI';

const getAIExplanation = async (condition: string, confidence: number, modelType: string) => {
  const prompt = `Provide a concise medical analysis for ${condition} (${confidence}% confidence, ${modelType}) with these sections:

1. Condition Overview:
   What is ${condition}? Describe its key characteristics and medical significance.

2. Patient Impact:
   How this condition typically affects patients and its common symptoms.

3. Risk Assessment:
   Severity level based on the findings and factors that might influence the condition.

4. Recommendations:
   Immediate steps to take, when to seek emergency care, and follow-up care suggestions.

5. Additional Considerations:
   Related conditions, preventive measures, and lifestyle modifications if applicable.

Format your response with clear section headings. Keep each section concise while including all critical information. Start immediately with the analysis without any introductory text.`;

  try {
    const explanation = await generateAIResponse(prompt);
    return explanation;
  } catch (error) {
    console.error('Error getting AI explanation:', error);
    return 'AI explanation currently unavailable. Please consult with a healthcare professional for detailed information about your condition.';
  }
};
const ImageDiagnosis = () => {
  const { isAuthenticated } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("brain-tumor");
  const [cameraActive, setCameraActive] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isCameraLoading, setIsCameraLoading] = useState(false);
  
  // Cropping state
  const [crop, setCrop] = useState<Crop>();
  const [completedCrop, setCompletedCrop] = useState<PixelCrop>();
  const [isCropping, setIsCropping] = useState(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  
  // Grad-CAM state
  const [gradcamImage, setGradcamImage] = useState<string | null>(null);
  const [isGradcamLoading, setIsGradcamLoading] = useState(false);
  
  // Available models for detection
  const models = [
    { id: "brain-tumor", name: "Brain Tumor Detection" },
    { id: "skin-infection", name: "Skin Infection Detection" },
    { id: "pneumonia", name: "Pneumonia Detection" },
    { id: "lung-cancer", name: "Lung Cancer Detection" }
  ];
  
  // Clean up function for camera stream
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);
  
  // Function to enumerate available cameras
  const enumerateCameras = async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        throw new Error("Media Devices API not supported in this browser");
      }
      
      // Request permission first to get labeled devices
      try {
        const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
        // Stop the temporary stream immediately
        tempStream.getTracks().forEach(track => track.stop());
      } catch (err) {
        console.log("Permission request for camera labels failed:", err);
        // Continue anyway, we'll just have unlabeled devices
      }
      
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      
      console.log("Available video devices:", videoDevices);
      setAvailableCameras(videoDevices);
      
      // Set default camera if available and not already set
      if (videoDevices.length > 0 && !selectedCamera) {
        setSelectedCamera(videoDevices[0].deviceId);
      }
      
      return videoDevices;
    } catch (error) {
      console.error('Error enumerating cameras:', error);
      setCameraError(error instanceof Error ? error.message : "Failed to detect cameras");
      return [];
    }
  };

  // Initialize camera enumeration on component mount
  useEffect(() => {
    enumerateCameras();
    
    // Clean up any camera stream when component unmounts
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);
  
  // Function to stop camera
  const stopCamera = () => {
    console.log("Stopping camera...");
    if (stream) {
      console.log("Stopping all tracks in stream");
      stream.getTracks().forEach(track => {
        console.log(`Stopping track: ${track.kind}, enabled: ${track.enabled}`);
        track.stop();
      });
      setStream(null);
    } else {
      console.log("No stream to stop");
    }
    
    // Clear video source
    if (videoRef.current && videoRef.current.srcObject) {
      console.log("Clearing video source object");
      videoRef.current.srcObject = null;
    }
    
    setCameraActive(false);
    setIsCameraLoading(false);
    setCameraError(null);
  };
  
  // Function to start camera
  const startCamera = async () => {
    try {
      console.log("Starting camera...");
      setCameraError(null);
      setIsCameraLoading(true);
      
      // First set camera active to show the modal
      setCameraActive(true);
      
      // Then enumerate cameras to ensure we have the latest list
      const cameras = await enumerateCameras();
      console.log("Cameras enumerated:", cameras.length);
      
      if (cameras.length === 0) {
        throw new Error("No cameras detected on your device");
      }
      
      // If no camera is selected yet, use the first one
      const deviceId = selectedCamera || cameras[0].deviceId;
      console.log("Using camera with deviceId:", deviceId);
      
      // Request camera permissions with the selected device
      const constraints = { 
        video: { 
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      };
      
      console.log("Requesting media with constraints:", JSON.stringify(constraints));
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      console.log("Media stream obtained, tracks:", mediaStream.getTracks().length);
      
      // Store the stream in state
      setStream(mediaStream);
      
      // Set up the video element
      if (videoRef.current) {
        console.log("Setting video source object");
        videoRef.current.srcObject = mediaStream;
        
        // Set up event handler for when metadata is loaded
        videoRef.current.onloadedmetadata = () => {
          console.log("Video metadata loaded, attempting to play");
          if (videoRef.current) {
            videoRef.current.play()
              .then(() => {
                console.log("Camera started successfully");
                setIsCameraLoading(false);
              })
              .catch(err => {
                console.error("Error playing video:", err);
                setCameraError("Could not start camera: " + err.message);
                setIsCameraLoading(false);
              });
          }
        };
        
        // Reset any previous analysis and file selections
        setSelectedFile(null);
        setPreview(null);
        setAnalysisResult(null);
      } else {
        console.error("Video ref is not available");
        setCameraError("Video element not found");
        setIsCameraLoading(false);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      setCameraError(error instanceof Error ? error.message : "Unknown camera error");
      setIsCameraLoading(false);
      
      toast({
        title: "Camera Error",
        description: "Could not access your camera. Please check permissions.",
        variant: "destructive"
      });
    }
  };
  
  // Function to switch cameras
  const switchCamera = async (newCameraId: string) => {
    try {
      console.log("Switching to camera:", newCameraId);
      setIsCameraLoading(true);
      
      // Stop current stream if it exists
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      
      // Update selected camera
      setSelectedCamera(newCameraId);
      
      // Get new stream with selected camera
      const newStream = await navigator.mediaDevices.getUserMedia({
        video: { 
          deviceId: { exact: newCameraId },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });
      
      // Update stream state
      setStream(newStream);
      
      // Update video element
      if (videoRef.current) {
        videoRef.current.srcObject = newStream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play()
            .then(() => {
              console.log("Camera switched successfully");
              setIsCameraLoading(false);
            })
            .catch(err => {
              console.error("Error playing video after switch:", err);
              setCameraError("Could not start new camera: " + err.message);
              setIsCameraLoading(false);
            });
        };
      } else {
        throw new Error("Video element not available");
      }
    } catch (error) {
      console.error("Error switching camera:", error);
      setCameraError(error instanceof Error ? error.message : "Failed to switch camera");
      setIsCameraLoading(false);
    }
  };
  
  // Helper function to process files (used by both drag & drop and file input)
  const processFile = (file: File) => {
    // Create preview with image resizing
    const reader = new FileReader();
    reader.onloadend = () => {
      const img = new Image();
      img.onload = () => {
        // Create a canvas to resize the image
        const canvas = document.createElement('canvas');
        // Define max dimensions
        const maxWidth = 1200;
        const maxHeight = 1200;
        
        let width = img.width;
        let height = img.height;
        
        // Calculate new dimensions while maintaining aspect ratio
        if (width > height) {
          if (width > maxWidth) {
            height = Math.round(height * (maxWidth / width));
            width = maxWidth;
          }
        } else {
          if (height > maxHeight) {
            width = Math.round(width * (maxHeight / height));
            height = maxHeight;
          }
        }
        
        // Set canvas dimensions and draw resized image
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx?.drawImage(img, 0, 0, width, height);
        
        // Get resized image as data URL and blob
        canvas.toBlob((blob) => {
          if (blob) {
            const resizedFile = new File([blob], file.name, { type: file.type });
            
            // Instead of setting selected file directly, go to cropping mode
            const imageUrl = URL.createObjectURL(blob);
            setImageSrc(imageUrl);
            setIsCropping(true);
          }
        }, file.type);
      };
      img.src = reader.result as string;
    };
    reader.readAsDataURL(file);
  };
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      
      // Validate file type
      if (!file.type.startsWith('image/')) {
        toast({
          title: "Invalid File",
          description: "Please select an image file.",
          variant: "destructive",
        });
        return;
      }
      
      // Instead of setting selected file directly, go to cropping mode
      const imageUrl = URL.createObjectURL(file);
      setImageSrc(imageUrl);
      setIsCropping(true);
    }
  };
  
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };
  
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      
      // Validate file type
      if (!file.type.startsWith('image/')) {
        toast({
          title: "Invalid File",
          description: "Please select an image file.",
          variant: "destructive",
        });
        return;
      }
      
      // Instead of setting selected file directly, go to cropping mode
      const imageUrl = URL.createObjectURL(file);
      setImageSrc(imageUrl);
      setIsCropping(true);
    }
  };
  
  // Function to handle crop completion
  const cropImage = () => {
    if (!completedCrop || !imgRef.current) return;
    
    const image = imgRef.current;
    const scaleX = image.naturalWidth / image.width;
    const scaleY = image.naturalHeight / image.height;
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Calculate actual pixel values based on the crop percentage or pixels
    const pixelCrop = {
      x: (completedCrop.unit as string) === '%' 
        ? (completedCrop.x * image.naturalWidth) / 100
        : completedCrop.x * scaleX,
      y: (completedCrop.unit as string) === '%' 
        ? (completedCrop.y * image.naturalHeight) / 100
        : completedCrop.y * scaleY,
      width: (completedCrop.unit as string) === '%' 
        ? (completedCrop.width * image.naturalWidth) / 100
        : completedCrop.width * scaleX,
      height: (completedCrop.unit as string) === '%' 
        ? (completedCrop.height * image.naturalHeight) / 100
        : completedCrop.height * scaleY,
    };
    
    // Set canvas dimensions to match the original image size if crop dimensions match
    const isFullCrop = Math.abs(pixelCrop.width - image.naturalWidth) < 1 && 
                      Math.abs(pixelCrop.height - image.naturalHeight) < 1;
    
    if (isFullCrop) {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      ctx?.drawImage(image, 0, 0);
    } else {
      // Set canvas dimensions to crop size
      canvas.width = pixelCrop.width;
      canvas.height = pixelCrop.height;
      
      // Draw the cropped image
      ctx.drawImage(
        image,
        pixelCrop.x,
        pixelCrop.y,
        pixelCrop.width,
        pixelCrop.height,
        0,
        0,
        pixelCrop.width,
        pixelCrop.height
      );
    }
    
    // Convert to blob and create a File object
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], "cropped-image.jpg", { type: "image/jpeg" });
        setSelectedFile(file);
        setPreview(canvas.toDataURL('image/jpeg', 0.85));
        setIsCropping(false);
        setImageSrc(null);
      }
    }, 'image/jpeg', 0.85);
  };
  const cancelCropping = () => {
    setIsCropping(false);
    setImageSrc(null);
  };
  
  // Function to select the entire image for initial crop
  const centerAspectCrop = (mediaWidth: number, mediaHeight: number): PixelCrop => {
    return {
      unit: 'px',
      x: 0,
      y: 0,
      width: mediaWidth,
      height: mediaHeight
    };
  };
  
  
  const onImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setCrop(centerAspectCrop(img.naturalWidth, img.naturalHeight));
    setCompletedCrop(centerAspectCrop(img.naturalWidth, img.naturalHeight));
  };
  
  
  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw current video frame onto canvas
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0);
        
        // Convert to image file and preview
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
            
            // Instead of setting selected file directly, go to cropping mode
            const imageUrl = URL.createObjectURL(blob);
            setImageSrc(imageUrl);
            setIsCropping(true);
            
            // Stop the camera after capturing
            stopCamera();
          }
        }, 'image/jpeg', 0.85);
      }
    }
  };
  
  const clearSelectedFile = () => {
    setSelectedFile(null);
    setPreview(null);
    setAnalysisResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  const analyzeImage = async () => {
    if (!selectedFile) return;
    
    setIsAnalyzing(true);
    
    try {
      console.log(`Analyzing image with model: ${selectedModel}`);
      console.log(`Image file: ${selectedFile.name}, size: ${selectedFile.size} bytes`);
      
      // Call our backend API service with the image
      const result = await analyzeImageAPI(selectedFile, selectedModel);
      // Get AI explanation for the result
      const explanation = await getAIExplanation(
        result.condition,
        result.confidence,
        models.find(m => m.id === selectedModel)?.name || 'AI Detection'
      );
      
      // Add the explanation to the result
      result.aiExplanation = explanation;
      
      setAnalysisResult(result);
      console.log("Analysis completed successfully:", result);
    } catch (error) {
      console.error('Error analyzing image:', error);
      // Provide more detailed error feedback to the user
      let errorMessage = "Failed to analyze the image. Please try again.";
      
      if (error instanceof Error) {
        errorMessage = error.message;
        
        // Check for specific error types
        if (errorMessage.includes("fetch")) {
          errorMessage = `Failed to connect to the analysis server. Please make sure the backend server is running at ${import.meta.env.VITE_FASTAPI_URL || 'the configured URL'}.`;
        } else if (errorMessage.includes("NetworkError")) {
          errorMessage = "Network error: Please check your internet connection and ensure the backend server is running.";
        }
      }
      
      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  // Handler to request Grad-CAM for lung cancer
  const handleShowGradcam = async () => {
    if (!selectedFile) return;
    setIsGradcamLoading(true);
    setGradcamImage(null);
    try {
      const gradcam = await requestLungCancerGradCAM(selectedFile);
      setGradcamImage(`data:image/png;base64,${gradcam}`);
    } catch (err) {
      toast({
        title: 'Grad-CAM Error',
        description: err instanceof Error ? err.message : 'Failed to generate Grad-CAM',
        variant: 'destructive',
      });
    } finally {
      setIsGradcamLoading(false);
    }
  };
  
  // Redirect if not authenticated
  if (!isAuthenticated) {
    toast({
      title: "Access Denied",
      description: "You must be logged in to access this feature.",
      variant: "destructive",
    });
    navigate("/login");
    return null;
  }
  
  // Whenever a new prediction is made or image changes, clear Grad-CAM
  useEffect(() => {
    setGradcamImage(null);
  }, [selectedFile, analysisResult, selectedModel]);

  return (
    <div className="min-h-screen flex flex-col pattern-bg">
      <Navbar />
      
      {/* Camera Modal */}
      {cameraActive && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center">
          <div className="w-full max-w-4xl bg-white rounded-lg overflow-hidden shadow-xl">
            {/* Video container */}
            <div className="relative bg-black aspect-video">
              <video 
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              {/* Subtle camera guidelines overlay */}
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-64 h-64 border border-white/30 rounded-full"></div>
                </div>
              </div>
              {/* Minimal header bar */}
              <div className="absolute top-0 left-0 right-0 p-3 flex justify-between items-center z-10">
                <div className="flex items-center bg-black/50 px-3 py-1.5 rounded-full">
                  <span className="text-sm text-white font-medium">Medical Image Capture</span>
                  {/* Camera selection dropdown - only if multiple cameras */}
                  {availableCameras.length > 1 && (
                    <select
                      value={selectedCamera}
                      onChange={(e) => switchCamera(e.target.value)}
                      className="ml-3 bg-transparent text-white border-0 text-sm focus:outline-none focus:ring-0"
                    >
                      {availableCameras.map((camera, index) => (
                        <option key={camera.deviceId} value={camera.deviceId} className="bg-gray-800 text-white">
                          {camera.label || `Camera ${index + 1}`}
                        </option>
                      ))}
                    </select>
                  )}
                </div>
                <button
                  onClick={() => {
                    console.log("Close camera button clicked");
                    stopCamera();
                  }}
                  className="p-1.5 bg-black/50 hover:bg-black/70 text-white rounded-full transition-colors"
                  aria-label="Close camera"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              {/* Minimal camera controls */}
              <div className="absolute bottom-4 left-0 right-0 flex justify-center">
                <button
                  onClick={() => {
                    console.log("Capture photo button clicked");
                    capturePhoto();
                  }}
                  className="p-3 bg-white/20 backdrop-blur-sm hover:bg-white/30 rounded-full transition-all focus:outline-none"
                  aria-label="Take Photo"
                >
                  <div className="h-12 w-12 rounded-full border-2 border-white flex items-center justify-center">
                    <div className="h-10 w-10 bg-terracotta rounded-full"></div>
                  </div>
                </button>
              </div>
              {/* Subtle hint text */}
              <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/70 to-transparent pointer-events-none">
                <p className="text-xs text-center text-white/70">
                  Center the affected area and click to capture
                </p>
              </div>
              {/* Camera loading indicator */}
              {isCameraLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/70 z-20">
                  <div className="text-center p-6">
                    <Loader2 className="h-10 w-10 text-terracotta mx-auto animate-spin mb-4" />
                    <p className="mt-3 text-white font-medium">Initializing camera...</p>
                  </div>
                </div>
              )}
              {/* Camera error message */}
              {cameraError && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/70 z-30">
                  <div className="text-center p-6 bg-white rounded-lg max-w-md">
                    <AlertCircle className="h-10 w-10 mx-auto mb-2 text-red-500" />
                    <h3 className="text-lg font-medium">Camera Error</h3>
                    <p className="mt-2 text-sm text-gray-600">{cameraError}</p>
                    <button 
                      onClick={() => {
                        console.log("Error close button clicked");
                        stopCamera();
                      }}
                      className="mt-4 px-4 py-2 bg-gray-100 text-gray-800 rounded-md font-medium hover:bg-gray-200 transition-colors"
                    >
                      Close
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      
      {/* Cropping interface */}
      {isCropping && imageSrc && (
        <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center">
          <div className="w-full max-w-4xl bg-white rounded-lg overflow-hidden shadow-xl">
            <div className="p-4 border-b border-gray-200 flex justify-between items-center">
              <h3 className="text-lg font-medium text-gray-900">Adjust Image</h3>
              <button 
                onClick={cancelCropping}
                className="p-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full transition-colors"
                aria-label="Cancel"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            
            <div className="p-4 flex flex-col items-center">
              <div className="mb-4 max-h-[60vh] overflow-auto bg-gray-100 p-2 rounded-lg">
                <ReactCrop
                  crop={crop}
                  onChange={(c) => setCrop(c)}
                  onComplete={(c) => setCompletedCrop(c)}
                  className="max-w-full"
                  ruleOfThirds
                >
                  <img
                    ref={imgRef}
                    src={imageSrc}
                    alt="Crop preview"
                    onLoad={onImageLoad}
                    className="max-w-full"
                  />
                </ReactCrop>
              </div>
              
              <div className="w-full max-w-md mb-4">
                <div className="text-sm text-gray-600 mb-1">Drag the corners to adjust crop area</div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-2 rounded border border-gray-200">
                    <div className="text-xs text-gray-500">Width</div>
                    <div className="text-sm font-medium">
                      {completedCrop ? Math.round(completedCrop.width) + ((completedCrop.unit as string) === 'px' ? 'px' : '%') : '-'}
                    </div>
                  </div>
                  <div className="bg-gray-50 p-2 rounded border border-gray-200">
                    <div className="text-xs text-gray-500">Height</div>
                    <div className="text-sm font-medium">
                      {completedCrop ? Math.round(completedCrop.height) + ((completedCrop.unit as string) === 'px' ? 'px' : '%') : '-'}
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="flex space-x-4">
                <button
                  onClick={cancelCropping}
                  className="px-4 py-2 bg-gray-100 text-gray-800 rounded-md font-medium hover:bg-gray-200 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={cropImage}
                  className="px-4 py-2 bg-terracotta text-white rounded-md font-medium hover:bg-terracotta/90 transition-colors flex items-center"
                >
                  <CropIcon className="h-4 w-4 mr-2" />
                  Apply Crop
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
      
      <div className="flex-grow flex justify-center px-4 pt-24 pb-20">
        <div className="w-full max-w-4xl">
          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="mb-6">
              <h1 className="text-2xl font-bold text-gray-900">Image Diagnosis</h1>
              <p className="mt-2 text-gray-600">
                Upload medical images or take a photo for AI analysis and condition detection.
              </p>
            </div>
            
            <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-100">
              <div className="flex">
                <Info className="h-5 w-5 text-blue-700 mr-2 flex-shrink-0" />
                <div>
                  <h3 className="text-sm font-medium text-blue-800">Image Analysis Tool</h3>
                  <p className="mt-1 text-sm text-blue-700">
                    Select the appropriate detection model for your image. Each model has been trained to detect specific conditions.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="mb-6">
              <label htmlFor="model-select" className="block text-sm font-medium text-gray-700 mb-1">
                Select Detection Model
              </label>
              <Select
                value={selectedModel}
                onValueChange={(value) => {
                  setSelectedModel(value);
                  setAnalysisResult(null);
                }}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select a detection model" />
                </SelectTrigger>
                <SelectContent>
                  {models.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            {/* Add model-specific guidance */}
            <div className="mb-6">
              <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
                <div className="flex items-start">
                  <Info className="h-5 w-5 text-blue-700 mt-1 mr-2 flex-shrink-0" />
                  <div>
                    <h3 className="text-sm font-medium text-blue-800">Image Selection Guidelines</h3>
                    
                    {selectedModel === 'brain-tumor' && (
                      <div className="mt-1 text-sm text-blue-700">
                        <p className="mb-2"><strong>For Brain Tumor Detection:</strong></p>
                        <ul className="list-disc pl-5 space-y-1">
                          <li>Use axial (top-down) T1 or T2-weighted MRI slices</li>
                          <li>Select slices where the tumor is clearly visible</li>
                          <li>The MRI should be properly cropped to show the brain area</li>
                          <li>Higher resolution images yield better results</li>
                          <li>Avoid using CT scans or other non-MRI images</li>
                        </ul>
                      </div>
                    )}
                    
                    {selectedModel === 'pneumonia' && (
                      <div className="mt-1 text-sm text-blue-700">
                        <p className="mb-2"><strong>For Pneumonia Detection:</strong></p>
                        <ul className="list-disc pl-5 space-y-1">
                          <li>Use clear, frontal chest X-rays (PA view preferred)</li>
                          <li>The entire lung field should be visible</li>
                          <li>Avoid photos of X-rays - use digital X-ray files if possible</li>
                          <li>Proper contrast is important for accurate detection</li>
                        </ul>
                      </div>
                    )}
                    
                    {selectedModel === 'skin-infection' && (
                      <div className="mt-1 text-sm text-blue-700">
                        <p className="mb-2"><strong>For Skin Infection Detection:</strong></p>
                        <ul className="list-disc pl-5 space-y-1">
                          <li>Take clear, well-lit photos of the affected area</li>
                          <li>Include the entire affected region if possible</li>
                          <li>Multiple angles may help with classification</li>
                          <li>Include surrounding unaffected skin for context</li>
                        </ul>
                      </div>
                    )}
                    
                    {selectedModel === 'lung-cancer' && (
                      <div className="mt-1 text-sm text-blue-700">
                        <p className="mb-2"><strong>For Lung Cancer Detection:</strong></p>
                        <ul className="list-disc pl-5 space-y-1">
                          <li>Use clear, high-resolution CT scan images of the lungs</li>
                          <li>Ensure the image shows a complete cross-section of the chest</li>
                          <li>The model works best with axial (horizontal) CT slices</li>
                          <li>Images should be properly oriented (top of body at top of image)</li>
                          <li>For best results, use images that focus on areas of concern</li>
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Canvas for capturing images (hidden) */}
            <canvas ref={canvasRef} className="hidden" />
            
            {!preview ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div 
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-terracotta transition-colors"
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    className="hidden"
                    accept="image/*"
                  />
                  <div className="flex flex-col items-center">
                    <Upload className="h-12 w-12 text-gray-400 mb-3" />
                    <h3 className="text-lg font-medium text-gray-900 mb-1">Upload an Image</h3>
                    <p className="text-sm text-gray-600 mb-2">
                      Drag and drop an image here, or click to select
                    </p>
                    <p className="text-xs text-gray-500">
                      Supported formats: JPEG, PNG, GIF (max 10MB)
                    </p>
                  </div>
                </div>
                
                <div 
                  onClick={() => {
                    console.log("Camera button clicked");
                    startCamera();
                  }}
                  className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-terracotta transition-colors"
                >
                  <div className="flex flex-col items-center">
                    <Camera className="h-12 w-12 text-gray-400 mb-3" />
                    <h3 className="text-lg font-medium text-gray-900 mb-1">Take a Medical Photo</h3>
                    <p className="text-sm text-gray-600 mb-2">
                      Use your camera to capture a medical image
                    </p>
                    <p className="text-xs text-gray-500">
                      Ideal for skin conditions, visible symptoms, or medical scans
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-8">
                {/* Main image display area: show both original and Grad-CAM side by side if available and lung-cancer selected */}
                {selectedModel === 'lung-cancer' && analysisResult ? (
                  <>
                    <div className="relative flex flex-row gap-8 justify-center items-center bg-gray-50 rounded-lg border border-gray-200" style={{ height: '400px' }}>
                      {/* Original Image */}
                      <div className="flex flex-col items-center justify-center h-full w-1/2">
                        <img 
                          src={preview || ''} 
                          alt="Original" 
                          className="object-contain max-h-[350px] max-w-full rounded shadow border border-gray-200 bg-white p-2" 
                          style={{ width: '100%', height: '350px' }}
                        />
                        <span className="text-xs text-gray-500 mt-2">Original Image</span>
                      </div>
                      {/* Grad-CAM Image (only show if available) */}
                      {gradcamImage && (
                        <div className="flex flex-col items-center justify-center h-full w-1/2">
                          <img 
                            src={gradcamImage} 
                            alt="AI Focus (Grad-CAM)" 
                            className="object-contain max-h-[350px] max-w-full rounded shadow border border-blue-300 bg-white p-2" 
                            style={{ width: '100%', height: '350px' }}
                          />
                          <span className="text-xs text-blue-600 mt-2">AI Focus (Grad-CAM)</span>
                        </div>
                      )}
                      <button
                        onClick={clearSelectedFile}
                        className="absolute top-2 right-2 p-1 bg-gray-800 bg-opacity-70 text-white rounded-full z-10"
                      >
                        <X className="h-5 w-5" />
                      </button>
                    </div>
                    {/* Move Grad-CAM button here, just below the images */}
                    {/* <div className="flex justify-center mt-4">
                      <button
                        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-60"
                        onClick={handleShowGradcam}
                        disabled={isGradcamLoading || !selectedFile}
                        type="button"
                      >
                        {isGradcamLoading ? 'Analyzing what the AI saw...' : gradcamImage ? 'Regenerate Grad-CAM' : 'See what the AI saw'}
                      </button>
                    </div> */}
                  </>
                ) : (
                  <div className="relative">
                    {/* Default: Show just the uploaded/cropped image */}
                    <div className="rounded-lg border border-gray-200 flex items-center justify-center bg-gray-50" style={{ height: "400px" }}>
                      <img 
                        src={preview} 
                        alt="Selected" 
                        className="max-w-full max-h-full object-contain p-2"
                      />
                    </div>
                    <button
                      onClick={clearSelectedFile}
                      className="absolute top-2 right-2 p-1 bg-gray-800 bg-opacity-70 text-white rounded-full"
                    >
                      <X className="h-5 w-5" />
                    </button>
                    <div className="mt-2 text-sm text-gray-600">
                      {selectedFile?.name}
                      {selectedFile && <span className="text-gray-400 ml-1">({Math.round(selectedFile.size / 1024)} KB)</span>}
                    </div>
                  </div>
                )}
                
                {!analysisResult && !isAnalyzing && (
                  <div className="flex justify-center">
                    <button
                      onClick={analyzeImage}
                      className="py-3 px-6 bg-gradient-to-r from-terracotta to-ochre text-white rounded-lg hover:from-terracotta/90 hover:to-ochre/90 transition-colors"
                    >
                      <ImageIcon className="h-5 w-5 mr-2 inline-block" />
                      Analyze with {models.find(m => m.id === selectedModel)?.name}
                    </button>
                  </div>
                )}
                
                {isAnalyzing && (
                  <div className="text-center p-6 bg-gray-50 rounded-lg border border-gray-200">
                    <Loader2 className="h-10 w-10 text-terracotta mx-auto animate-spin mb-4" />
                    <h3 className="text-lg font-medium text-gray-900">Analyzing Image</h3>
                    <p className="text-gray-600 mt-1">
                      Our AI is examining the image using {models.find(m => m.id === selectedModel)?.name}...
                    </p>
                  </div>
                )}
                
                {analysisResult && (
                  <div className="bg-gray-50 p-6 rounded-lg border border-gray-200 animate-fade-in">
                    <div className="flex items-start">
                      <div className={`p-3 rounded-full mr-4 ${
                        analysisResult.urgency === 'high' 
                          ? 'bg-red-100 text-red-700' 
                          : analysisResult.urgency === 'medium' 
                            ? 'bg-yellow-100 text-yellow-700' 
                            : 'bg-blue-100 text-blue-700'
                      }`}>
                        {analysisResult.urgency === 'high' 
                          ? <AlertCircle className="h-6 w-6" /> 
                          : analysisResult.urgency === 'medium' 
                            ? <Info className="h-6 w-6" /> 
                            : <Check className="h-6 w-6" />
                        }
                      </div>
                      <div>
                        <h2 className="text-xl font-semibold text-gray-900">
                          {analysisResult.condition}
                          {selectedModel === 'brain-tumor' && (
                            <span className={`ml-2 px-2 py-1 text-sm rounded ${
                              analysisResult.condition === 'no tumor' 
                                ? 'bg-green-100 text-green-700' 
                                : 'bg-red-100 text-red-700'
                            }`}>
                              {analysisResult.condition === 'no tumor' ? 'No tumor detected' : 'Tumor detected'}
                            </span>
                          )}
                          {selectedModel === 'pneumonia' && (
                            <span className={`ml-2 px-2 py-1 text-sm rounded ${
                              analysisResult.condition === 'Normal' 
                                ? 'bg-green-100 text-green-700' 
                                : 'bg-red-100 text-red-700'
                            }`}>
                              {analysisResult.condition === 'Normal' ? 'No pneumonia detected' : 'Pneumonia detected'}
                            </span>
                          )}
                          {/* Low confidence warning */}
                          {analysisResult.lowConfidence && (
                            <span className="ml-2 px-2 py-1 text-xs font-normal rounded bg-yellow-100 text-yellow-800">
                              Low confidence prediction
                            </span>
                          )}
                        </h2>
                        {/* Display model info for skin infection */}
                        {selectedModel === 'skin-infection' && analysisResult.modelUsed && (
                          <div className="mt-1 text-xs font-medium text-indigo-600">
                            Using {analysisResult.modelUsed} model
                          </div>
                        )}
                        <p className="mt-1 text-gray-600">{analysisResult.description}</p>
                        <div className="mt-3 flex items-center">
                          <span className="text-sm font-medium text-gray-700">Confidence:</span>
                          <div className="ml-2 h-2 w-24 bg-gray-200 rounded-full">
                            <div 
                              className={`h-2 rounded-full ${
                                analysisResult.confidence > 80 
                                  ? 'bg-gradient-to-r from-green-500 to-green-300' 
                                  : analysisResult.confidence > 60 
                                    ? 'bg-gradient-to-r from-yellow-500 to-yellow-300' 
                                    : 'bg-gradient-to-r from-red-500 to-red-300'
                              }`} 
                              style={{ width: `${analysisResult.confidence}%` }}
                            ></div>
                          </div>
                          <span className="ml-2 text-sm text-gray-600">{analysisResult.confidence}%</span>
                          <span className="ml-2 text-xs text-gray-500">
                            {analysisResult.confidence > 80 
                              ? '(High confidence)' 
                              : analysisResult.confidence > 60 
                                ? '(Moderate confidence)' 
                                : '(Low confidence - results may be less reliable)'}
                          </span>
                        </div>
                      </div>
                    </div>   
                    {/* AI Explanation Section */}
                    <div className="mt-6 p-4 bg-indigo-50 rounded-lg border border-indigo-100">
                      <div className="flex">
                        <Info className="h-5 w-5 text-indigo-700 mr-2 flex-shrink-0 mt-1" />
                        <div className="w-full">
                          <h3 className="text-sm font-medium text-indigo-800">AI Detailed Analysis</h3>
                          <div className="mt-2 text-sm text-indigo-700 space-y-4">
                            {analysisResult.aiExplanation ? (
                              <div className="prose prose-indigo max-w-none">
                                {analysisResult.aiExplanation.split('\n').map((paragraph, index) => {
                                  // Check if this is a section header
                                  if (paragraph.match(/^\d+\.\s+[A-Za-z\s]+:/)) {
                                    return (
                                      <h4 key={index} className="text-indigo-900 font-medium mt-4 first:mt-0">
                                        {paragraph.replace(/^\d+\.\s+/, '')}
                                      </h4>
                                    );
                                  }
                                  // Regular paragraph
                                  else if (paragraph.trim() && !paragraph.trim().startsWith('*')) {
                                    return (
                                      <p key={index} className="text-indigo-800">
                                        {paragraph}
                                      </p>
                                    );
                                  }
                                  return null;
                                })}
                              </div>
                            ) : (
                              <div className="flex items-center justify-center p-4">
                                <Loader2 className="h-6 w-6 text-indigo-600 animate-spin mr-2" />
                                <p>Generating detailed AI analysis...</p>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Report Generation Buttons */}
                    <div className="mt-6 flex justify-end space-x-4">
                      <button
                        onClick={() => {
                          const now = new Date();
                          // Convert the preview image to base64
                          const imageData = preview || '';
                          
                          downloadReport({
                            date: now.toLocaleDateString(),
                            time: now.toLocaleTimeString(),
                            condition: analysisResult.condition,
                            confidence: analysisResult.confidence,
                            description: analysisResult.description,
                            urgency: analysisResult.urgency,
                            modelUsed: models.find(m => m.id === selectedModel)?.name,
                            imageData: imageData,
                            alternatives: analysisResult.alternatives,
                            aiExplanation: analysisResult.aiExplanation
                          });
                        }}
                        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors flex items-center"
                      >
                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Download Report
                      </button>
                      <button
                        onClick={() => {
                          const now = new Date();
                          // Convert the preview image to base64
                          const imageData = preview || '';
                          
                          printReport({
                            date: now.toLocaleDateString(),
                            time: now.toLocaleTimeString(),
                            condition: analysisResult.condition,
                            confidence: analysisResult.confidence,
                            description: analysisResult.description,
                            urgency: analysisResult.urgency,
                            modelUsed: models.find(m => m.id === selectedModel)?.name,
                            imageData: imageData,
                            alternatives: analysisResult.alternatives,
                            aiExplanation: analysisResult.aiExplanation
                          });
                        }}
                        className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors flex items-center"
                      >
                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
                        </svg>
                        Print Report
                      </button>
                    </div>
                    
                    {/* Display alternative diagnoses for skin infection */}
                    {selectedModel === 'skin-infection' && analysisResult.alternatives && analysisResult.alternatives.length > 0 && (
                      <div className="mt-6 p-4 bg-amber-50 rounded-lg border border-amber-100">
                        <h3 className="text-sm font-medium text-amber-800 mb-2">Alternative Possible Diagnoses</h3>
                        <div className="space-y-3">
                          {analysisResult.alternatives.map((alt: any, index: number) => (
                            <div key={index} className="flex items-start">
                              <div className={`mr-2 h-2 w-2 mt-2 rounded-full ${
                                alt.urgency === 'high' 
                                  ? 'bg-red-500' 
                                  : alt.urgency === 'medium' 
                                    ? 'bg-yellow-500' 
                                    : 'bg-blue-500'
                              }`} />
                              <div>
                                <div className="text-sm font-medium text-gray-800 flex items-center">
                                  {alt.condition}
                                  <span className="ml-2 text-xs text-gray-500">({alt.confidence}% confidence)</span>
                                </div>
                                <p className="text-xs text-gray-600">{alt.description}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                        <p className="mt-3 text-xs text-amber-700">
                          <strong>Note:</strong> If your condition looks more like any of these alternatives, 
                          consider mentioning them to your healthcare provider.
                        </p>
                      </div>
                    )}
                    
                    <div className="mt-4 text-xs text-gray-500">
                      <strong>Disclaimer:</strong> This is an AI-assisted analysis and not a definitive medical diagnosis. 
                      Please consult with a qualified healthcare professional for proper evaluation and treatment.
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
};

export default ImageDiagnosis;
