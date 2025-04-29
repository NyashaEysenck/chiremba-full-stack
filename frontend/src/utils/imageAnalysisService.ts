// This service handles interaction with the AI image analysis backend

interface AnalysisResult {
  predicted_class: string;
  confidence: number;
  model_used?: string;  // Add optional model_used field
  alternatives?: Array<{class: string, confidence: number}>;
}

// Map frontend model IDs to backend API endpoints
const API_ENDPOINTS: Record<string, string> = {
  'brain-tumor': '/braintumor_detection',
  'pneumonia': '/pneumonia_detection',
  'skin-infection': '/skindisease_classification',
  'lung-cancer': '/lungcancer_prediction'
};

// More detailed description for each condition to improve user feedback
export const CONDITION_DETAILS: Record<string, {
  description: string;
  recommendations: string;
  urgency: 'high' | 'medium' | 'low';
}> = {
  // Brain tumor conditions
  'glioma': {
    description: 'Glioma is a type of tumor that occurs in the brain and spinal cord. Gliomas begin in the glial cells that surround and support nerve cells.',
    recommendations: 'Immediate consultation with a neurosurgeon or neuro-oncologist is strongly recommended. Treatment typically involves surgery, radiation therapy, and/or chemotherapy.',
    urgency: 'high'
  },
  'meningioma': {
    description: 'Meningioma is a tumor that forms on membranes that cover the brain and spinal cord just inside the skull. Most meningiomas are noncancerous (benign).',
    recommendations: 'Consultation with a neurosurgeon is recommended. Depending on size and location, treatment may include observation, surgery, or radiation therapy.',
    urgency: 'medium'
  },
  'no tumor': {
    description: 'No signs of brain tumor were detected in the provided image.',
    recommendations: 'No specific action needed regarding brain tumors. If you are experiencing symptoms, please consult with a healthcare provider for proper evaluation.',
    urgency: 'low'
  },
  'pituitary': {
    description: 'Pituitary tumors are abnormal growths that develop in the pituitary gland. Most pituitary tumors are noncancerous (benign) and don\'t spread to other parts of your body.',
    recommendations: 'Consultation with an endocrinologist and neurosurgeon is recommended. Treatment depends on type, size, and hormonal activity.',
    urgency: 'medium'
  },
  
  // Pneumonia conditions
  'Normal': {
    description: 'The lung X-ray appears normal with no signs of pneumonia or other concerning lung abnormalities.',
    recommendations: 'No specific action needed regarding pneumonia. If you are experiencing respiratory symptoms, please consult with a healthcare provider.',
    urgency: 'low'
  },
  'Pneumonia': {
    description: 'Pneumonia is an infection that inflames the air sacs in one or both lungs. The X-ray shows areas of opacity that indicate pneumonia.',
    recommendations: 'Seek medical attention promptly. Treatment typically includes antibiotics for bacterial pneumonia, rest, fluids, and possibly hospitalization depending on severity.',
    urgency: 'high'
  },
  
  // Skin infection conditions
  'Chickenpox': {
    description: 'Chickenpox is a highly contagious viral infection characterized by an itchy, blister-like rash.',
    recommendations: 'Avoid scratching, use calamine lotion, and consult a healthcare provider for antiviral therapy if needed.',
    urgency: 'medium'
  },
  'Cowpox': {
    description: 'Cowpox is a rare viral skin infection that causes pustular lesions, usually acquired from animals.',
    recommendations: 'Seek medical attention for lesion care and monitoring. Most cases resolve without treatment.',
    urgency: 'medium'
  },
  'HFMD': {
    description: 'Hand, Foot, and Mouth Disease (HFMD) is a common viral illness in children causing sores and rash.',
    recommendations: 'Maintain hydration, manage fever, and consult a doctor if symptoms worsen.',
    urgency: 'low'
  },
  'Healthy': {
    description: 'No signs of skin disease detected in the provided image.',
    recommendations: 'No action needed. Maintain good hygiene and monitor for any changes.',
    urgency: 'low'
  },
  'Measles': {
    description: 'Measles is a highly contagious viral disease with fever and a characteristic rash.',
    recommendations: 'Seek medical attention. Supportive care and isolation are important. Vaccination prevents measles.',
    urgency: 'high'
  },
  'Monkeypox': {
    description: 'Monkeypox is a rare viral disease causing fever, rash, and swollen lymph nodes.',
    recommendations: 'Consult a healthcare provider for diagnosis and supportive care. Isolate to prevent spread.',
    urgency: 'high'
  },
  'Athlete-Foot': {
    description: 'Athlete\'s foot (tinea pedis) is a fungal infection that typically begins between the toes. It can cause a scaly, red rash that usually causes itching, stinging, and burning.',
    recommendations: 'Over-the-counter antifungal medications, keeping feet dry, and proper hygiene. If severe or persistent, consult a healthcare provider.',
    urgency: 'low'
  },
  'Impetigo': {
    description: 'Impetigo is a highly contagious skin infection that causes red sores that can break open, ooze fluid, and develop a honey-colored crust.',
    recommendations: 'Consult with a healthcare provider. Treatment typically includes antibiotic ointment or oral antibiotics, and keeping the area clean.',
    urgency: 'medium'
  },
  'Cutaneous Larva Migrans': {
    description: 'Cutaneous larva migrans is a skin condition caused by hookworm larvae that penetrate the skin and cause an intensely itchy, raised, red track or line.',
    recommendations: 'Consult with a healthcare provider. Treatment typically involves oral anti-parasitic medications like albendazole or ivermectin.',
    urgency: 'medium'
  },
  'Nail-Fungus': {
    description: 'Nail fungus (onychomycosis) is a common condition that causes a nail to become discolored, thickened, and more likely to crack and break.',
    recommendations: 'Oral antifungal medications, topical antifungal nail creams, or nail lacquer. Consult a healthcare provider for persistent cases.',
    urgency: 'low'
  },
  'Ringworm': {
    description: 'Ringworm (tinea corporis) is a common fungal infection that causes a red, circular, itchy rash. Despite its name, it has nothing to do with worms.',
    recommendations: 'Over-the-counter antifungal medications, keeping the area clean and dry. If widespread or severe, consult a healthcare provider for oral medication.',
    urgency: 'low'
  },
  'Shingles': {
    description: 'Shingles is a viral infection caused by the varicella-zoster virus that causes a painful rash. It commonly appears as a single stripe of blisters on one side of the body.',
    recommendations: 'Early treatment with antiviral medications can reduce the severity and duration. Pain management may include medications and topical treatments. Consult a healthcare provider promptly.',
    urgency: 'medium'
  },
  
  // Lung cancer conditions
  'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': {
    description: 'Adenocarcinoma is a type of non-small cell lung cancer that forms in the glandular cells of the lungs. This appears to be located in the left lower lobe, classified as Stage Ib (T2, N0, M0).',
    recommendations: 'Immediate consultation with an oncologist and thoracic surgeon is strongly recommended. Treatment typically involves surgery, possibly followed by chemotherapy and/or radiation therapy depending on specific characteristics.',
    urgency: 'high'
  },
  'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': {
    description: 'Large cell carcinoma is a type of non-small cell lung cancer characterized by large, abnormal-looking cells. This appears to be located in the left hilum (where blood vessels and airways enter the lung), classified as Stage IIIa (T2, N2, M0).',
    recommendations: 'Urgent consultation with an oncologist is required. Stage IIIa typically requires a combination of treatments, which may include chemotherapy, radiation therapy, and possibly surgery depending on specific factors.',
    urgency: 'high'
  },
  'normal': {
    description: 'No signs of lung cancer were detected in the provided CT scan image.',
    recommendations: 'No specific action needed regarding lung cancer. If you are experiencing respiratory symptoms, please consult with a healthcare provider for proper evaluation.',
    urgency: 'low'
  },
  'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': {
    description: 'Squamous cell carcinoma is a type of non-small cell lung cancer that forms in the flat cells lining the airways. This appears to be located in the left hilum (where blood vessels and airways enter the lung), classified as Stage IIIa (T1, N2, M0).',
    recommendations: 'Urgent consultation with an oncologist is required. Stage IIIa typically requires a combination of treatments, which may include chemotherapy, radiation therapy, and possibly surgery depending on specific factors.',
    urgency: 'high'
  }
};

// Utility to poll the backend health endpoint until ready
async function waitForBackendReady(maxRetries = 10, delayMs = 1000): Promise<boolean> {
  const healthUrl = `${import.meta.env.VITE_FASTAPI_URL}/health`;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const res = await fetch(healthUrl, { method: 'GET', mode: 'cors' });
      if (res.ok) {
        const data = await res.json();
        // Accept "healthy" or similar status (customize as needed)
        if (data.status === 'healthy' || data.status === true) {
          return true;
        }
      }
    } catch (e) {
      // Ignore fetch errors, will retry
    }
    await new Promise(resolve => setTimeout(resolve, delayMs));
  }
  return false;
}

// Add a new function to test connectivity first
export async function testServerConnection(file: File): Promise<boolean> {
  try {
    // Create a small FormData with the file to test the connection
    const formData = new FormData();
    formData.append('file', file);
    
    console.log('Testing connection to server...');
    
    // Use the test endpoint which is lightweight
    const response = await fetch(`${import.meta.env.VITE_FASTAPI_URL}/test`, {
      method: 'POST',
      body: formData,
      mode: 'cors',
      // Set a shorter timeout for the test
      signal: AbortSignal.timeout(5000) // 5 second timeout
    });
    
    if (response.ok) {
      console.log('Server connection successful');
      return true;
    } else {
      console.error('Server connection test failed:', response.status);
      return false;
    }
  } catch (error) {
    console.error('Server connection test error:', error);
    return false;
  }
}

/**
 * Analyzes an image using the appropriate model based on the selected model type
 * @param file The image file to analyze
 * @param modelType The type of model to use (brain-tumor, pneumonia, skin-infection)
 * @returns Promise with the analysis result including condition details
 */
export async function analyzeImage(file: File, modelType: string): Promise<any> {
  try {
    // Wait for backend readiness before proceeding (fix for initial connection issue)
    const backendReady = await waitForBackendReady(10, 1000); // 10 retries, 1s apart
    if (!backendReady) {
      throw new Error('The analysis server is still starting up or not ready. Please try again in a moment.');
    }
    // First test server connectivity
    const serverAvailable = await testServerConnection(file);
    
    if (!serverAvailable) {
      throw new Error(`Cannot connect to the analysis server. Please make sure the server is running at ${import.meta.env.VITE_FASTAPI_URL}`);
    }
    
    const endpoint = API_ENDPOINTS[modelType];
    if (!endpoint) {
      throw new Error(`Invalid model type: ${modelType}`);
    }

    // Check file type and size
    const validImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
    if (!validImageTypes.includes(file.type)) {
      throw new Error(`Invalid file type: ${file.type}. Please upload a valid image (JPEG, PNG, GIF, BMP, WEBP).`);
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10 MB limit
      throw new Error("File size too large. Please upload an image smaller than 10 MB.");
    }

    const formData = new FormData();
    formData.append('file', file);

    // Determine which API URL to use based on the model type
    const baseUrl = modelType === 'skin-infection' 
      ? import.meta.env.VITE_SKIN_SERVICE_URL 
      : import.meta.env.VITE_FASTAPI_URL;

    console.log(`Sending request to: ${baseUrl}${endpoint}`);
    
    // For testing purposes, use the test endpoint
    const useTestEndpoint = false; // Changed to false to use the real model endpoints
    
    const url = useTestEndpoint ? `${baseUrl}/test` : `${baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
      mode: 'cors',
      headers: {
        'Accept': 'application/json',
      }
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('API Error:', response.status, errorText);
      try {
        const errorData = JSON.parse(errorText);
        // Handle specific error cases
        if (errorData.detail && errorData.detail.includes("Depth of input must be a multiple")) {
          throw new Error("The image format is not compatible with the model. Please try a different image or convert it to standard RGB format.");
        }
        throw new Error(errorData.detail || `Server responded with status: ${response.status}`);
      } catch (e) {
        throw new Error(`Server error (${response.status}): ${errorText || 'Unknown error'}`);
      }
    }

    const result = await response.json();
    
    // Validate the result
    if (!result || typeof result.predicted_class === 'undefined' || typeof result.confidence === 'undefined') {
      throw new Error("Invalid response received from the server. The result format is incorrect.");
    }
    
    const predictedClass = result.predicted_class;
    const confidence = Math.round(result.confidence * 100);
    const modelUsed = result.model_used || "Standard";  // Track which model was used

    // Get additional details for the condition
    const details = CONDITION_DETAILS[predictedClass] || {
      description: `Analysis detected ${predictedClass}.`,
      recommendations: 'Please consult with a healthcare professional for proper evaluation.',
      urgency: 'medium' as const
    };

    // Process alternatives if they exist (for skin infection classification)
    let alternatives = [];
    if (result.alternatives && Array.isArray(result.alternatives)) {
      alternatives = result.alternatives.map((alt: {class: string, confidence: number}) => ({
        condition: alt.class,
        confidence: Math.round(alt.confidence * 100),
        description: CONDITION_DETAILS[alt.class]?.description || `Alternative possibility: ${alt.class}`,
        urgency: CONDITION_DETAILS[alt.class]?.urgency || 'medium'
      }));
    }

    // If confidence is low, mark this in the response
    const lowConfidence = confidence < 70;

    return {
      condition: predictedClass,
      confidence,
      description: details.description,
      recommendations: details.recommendations,
      urgency: details.urgency,
      alternatives,
      lowConfidence,
      modelUsed  // Include in the response
    };
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw error;
  }
} 