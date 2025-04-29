import { useState, useEffect } from 'react';
import { useToast } from '@/hooks/use-toast';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { ArrowRight, Heart, Search, Info, AlertCircle, Pill, Clock, HelpCircle, Thermometer, Weight, Ruler, Activity, Plus } from 'lucide-react';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { analyzeHealthSymptoms } from '@/utils/openAI';

const SymptomChecker = () => {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [symptomText, setSymptomText] = useState("");
  const [aiAnalysis, setAiAnalysis] = useState<string | null>(null);
  const [formattedAnalysis, setFormattedAnalysis] = useState<{
    possibleConditions: string[],
    treatments: string[],
    urgencyLevel: string,
    seekMedicalHelp: string,
    disclaimer: string
  } | null>(null);
  
  // Vital signs states
  const [temperature, setTemperature] = useState<string>("");
  const [weight, setWeight] = useState<string>("");
  const [bloodPressure, setBloodPressure] = useState<string>("");
  const [height, setHeight] = useState<string>("");
  const [showVitalSigns, setShowVitalSigns] = useState<boolean>(false);

  // Common symptoms for quick selection
  const commonSymptoms = [
    "Headache",
    "Fever",
    "Cough",
    "Sore throat",
    "Fatigue",
    "Nausea",
    "Abdominal pain",
    "Shortness of breath",
    "Dizziness",
    "Chest pain"
  ];

  // Scroll to top when page loads
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  // Add selected symptom to the text input
  const addSymptom = (symptom: string) => {
    const currentText = symptomText.trim();
    const separator = currentText ? ", " : "";
    setSymptomText(currentText + separator + symptom);
  };

  // Function to parse the AI response into structured sections and clean formatting
  const parseAnalysisResponse = (text: string) => {
    console.log("Raw AI response:", text); // Debug log
    
    // Initialize sections object
    const sections: any = {
      possibleConditions: [],
      treatments: [],
      urgencyLevel: '',
      seekMedicalHelp: '',
      disclaimer: ''
    };

    // Try to find section headers in the text
    const possibleConditionsMatch = text.match(/Possible conditions(.*?)(?=Suggested medications|$)/s);
    const treatmentsMatch = text.match(/Suggested medications(.*?)(?=Urgency level|$)/s);
    const urgencyMatch = text.match(/Urgency level(.*?)(?=When to seek|$)/s);
    const medicalHelpMatch = text.match(/When to seek professional medical help(.*?)(?=DISCLAIMER|Disclaimer|$)/s);
    const disclaimerMatch = text.match(/DISCLAIMER|Disclaimer(.*?)$/s);

    console.log("Parsing matches:", { 
      conditions: possibleConditionsMatch?.[1]?.trim(),
      treatments: treatmentsMatch?.[1]?.trim(),
      urgency: urgencyMatch?.[1]?.trim(),
      help: medicalHelpMatch?.[1]?.trim()
    }); // Debug log

    // Process each section if found and remove markdown formatting
    if (possibleConditionsMatch && possibleConditionsMatch[1]) {
      const conditionLines = possibleConditionsMatch[1].split('\n').filter(line => line.trim());
      sections.possibleConditions = conditionLines.map(line => {
        // Remove markdown ** formatting and numbers
        return line.replace(/^\d+\.\s*/, '')
                  .replace(/\*\*/g, '')
                  .trim();
      }).filter(Boolean);
    }

    if (treatmentsMatch && treatmentsMatch[1]) {
      const treatmentLines = treatmentsMatch[1].split('\n').filter(line => line.trim());
      sections.treatments = treatmentLines.map(line => {
        // Remove markdown ** formatting and list symbols
        return line.replace(/^\d+\.\s*/, '')
                  .replace(/^\-\s*/, '')
                  .replace(/\*\*/g, '')
                  .trim();
      }).filter(Boolean);
    }

    if (urgencyMatch && urgencyMatch[1]) {
      // Remove markdown ** formatting
      sections.urgencyLevel = urgencyMatch[1].replace(/\*\*/g, '').trim();
    }

    if (medicalHelpMatch && medicalHelpMatch[1]) {
      // Remove markdown ** formatting
      sections.seekMedicalHelp = medicalHelpMatch[1].replace(/\*\*/g, '').trim();
    }

    if (disclaimerMatch && disclaimerMatch[1]) {
      // Remove markdown ** formatting
      sections.disclaimer = disclaimerMatch[1].replace(/\*\*/g, '').trim();
    }

    console.log("Parsed sections:", sections); // Debug log
    return sections;
  };

  const handleTextSymptomSubmit = async () => {
    if (!symptomText.trim()) {
      toast({
        title: "Empty Input",
        description: "Please describe your symptoms first.",
        variant: "destructive"
      });
      return;
    }
    
    setIsLoading(true);
    setAiAnalysis(null);
    setFormattedAnalysis(null);
    
    try {
      // Create a comprehensive symptom description including vital signs if provided
      let fullSymptomDescription = symptomText;
      
      if (temperature || weight || bloodPressure || height) {
        fullSymptomDescription += "\n\nAdditional patient information:";
        if (temperature) fullSymptomDescription += `\n- Body temperature: ${temperature}°C`;
        if (weight) fullSymptomDescription += `\n- Weight: ${weight} kg`;
        if (height) fullSymptomDescription += `\n- Height: ${height} cm`;
        if (bloodPressure) fullSymptomDescription += `\n- Blood pressure: ${bloodPressure} mmHg`;
      }
      
      const result = await analyzeHealthSymptoms(fullSymptomDescription);
      
      if (result.success) {
        setAiAnalysis(result.analysisText);
        
        // Parse the response into structured sections
        const parsedAnalysis = parseAnalysisResponse(result.analysisText);
        setFormattedAnalysis(parsedAnalysis);
        
        // Debug: Check if analysis was parsed correctly
        console.log("AI Analysis result:", result.analysisText);
        console.log("Parsed analysis:", parsedAnalysis);
      } else {
        toast({
          title: "Analysis Failed",
          description: "There was an error analyzing your symptoms. Please try again.",
          variant: "destructive"
        });
      }
    } catch (error) {
      console.error("Error in symptom analysis:", error);
      toast({
        title: "System Error",
        description: "An unexpected error occurred. Please try again later.",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
      setIsComplete(true);
    }
  };

  const resetAssessment = () => {
    setIsComplete(false);
    setAiAnalysis(null);
    setFormattedAnalysis(null);
    setSymptomText("");
    setTemperature("");
    setWeight("");
    setBloodPressure("");
    setHeight("");
    setShowVitalSigns(false);
  };

  const getUrgencyColor = (urgencyLevel: string) => {
    if (!urgencyLevel) return 'bg-gray-100 text-gray-700 border-gray-200';
    
    const urgencyText = urgencyLevel.toLowerCase();
    if (urgencyText.includes('high')) return 'bg-red-100 text-red-700 border-red-200';
    if (urgencyText.includes('medium')) return 'bg-yellow-100 text-yellow-700 border-yellow-200';
    return 'bg-green-100 text-green-700 border-green-200';
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex flex-col pattern-bg">
        <Navbar />
        <div className="flex-grow flex items-center justify-center px-4 py-24">
          <div className="w-full max-w-lg text-center">
            <div className="flex flex-col items-center">
              <Heart className="h-16 w-16 text-terracotta animate-pulse" />
              <h2 className="mt-4 text-xl font-medium text-gray-900">
                Analyzing your symptoms...
              </h2>
              <p className="mt-2 text-gray-600">
                Our AI is processing your description to provide a preliminary assessment.
              </p>
            </div>
          </div>
        </div>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col pattern-bg">
      <Navbar />
      
      <div className="flex-grow flex items-center justify-center px-4 py-24">
        <div className="w-full max-w-lg">
          <div className="bg-white rounded-xl shadow-lg p-8">
            {!isComplete ? (
              <>
                <div className="mb-6">
                  <h1 className="text-2xl font-bold text-gray-900">Symptom Checker</h1>
                  <p className="mt-2 text-gray-600">
                    Get a preliminary assessment of your health concerns using our advanced AI assistant.
                  </p>
                </div>

                <div className="space-y-6">
                  <div>
                    <label htmlFor="symptoms" className="block text-sm font-medium text-gray-700 mb-2">
                      Describe your symptoms
                    </label>
                    <Textarea
                      id="symptoms"
                      placeholder="Describe your symptoms in detail. For example: I've had a headache for 2 days, along with fever and a sore throat..."
                      className="w-full h-32 resize-none"
                      value={symptomText}
                      onChange={(e) => setSymptomText(e.target.value)}
                    />
                    
                    {/* Common symptoms section */}
                    <div className="mt-3">
                      <p className="text-xs text-gray-500 mb-2">Common symptoms:</p>
                      <div className="flex flex-wrap gap-2">
                        {commonSymptoms.map((symptom, index) => (
                          <Button 
                            key={index}
                            variant="outline"
                            size="sm"
                            className="text-xs py-1 px-3 h-auto flex items-center"
                            onClick={() => addSymptom(symptom)}
                          >
                            <Plus className="h-3 w-3 mr-1" />
                            {symptom}
                          </Button>
                        ))}
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <button 
                      onClick={() => setShowVitalSigns(!showVitalSigns)}
                      className="text-sm flex items-center text-terracotta hover:underline"
                    >
                      {showVitalSigns ? "Hide vital signs" : "Add vital signs (optional)"}
                      <ArrowRight className={`ml-1 h-3 w-3 transition-transform ${showVitalSigns ? 'rotate-90' : ''}`} />
                    </button>
                    
                    {showVitalSigns && (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4 p-4 bg-gray-50 rounded-lg">
                        <div>
                          <label htmlFor="temperature" className="block text-xs font-medium text-gray-700 mb-1 flex items-center">
                            <Thermometer className="h-3 w-3 mr-1" />
                            Temperature (°C)
                          </label>
                          <Input
                            id="temperature"
                            type="number"
                            placeholder="36.5"
                            value={temperature}
                            onChange={(e) => setTemperature(e.target.value)}
                            className="w-full"
                          />
                        </div>
                        
                        <div>
                          <label htmlFor="weight" className="block text-xs font-medium text-gray-700 mb-1 flex items-center">
                            <Weight className="h-3 w-3 mr-1" />
                            Weight (kg)
                          </label>
                          <Input
                            id="weight"
                            type="number"
                            placeholder="70"
                            value={weight}
                            onChange={(e) => setWeight(e.target.value)}
                            className="w-full"
                          />
                        </div>
                        
                        <div>
                          <label htmlFor="bloodPressure" className="block text-xs font-medium text-gray-700 mb-1 flex items-center">
                            <Activity className="h-3 w-3 mr-1" />
                            Blood Pressure (mmHg)
                          </label>
                          <Input
                            id="bloodPressure"
                            placeholder="120/80"
                            value={bloodPressure}
                            onChange={(e) => setBloodPressure(e.target.value)}
                            className="w-full"
                          />
                        </div>
                        
                        <div>
                          <label htmlFor="height" className="block text-xs font-medium text-gray-700 mb-1 flex items-center">
                            <Ruler className="h-3 w-3 mr-1" />
                            Height (cm)
                          </label>
                          <Input
                            id="height"
                            type="number"
                            placeholder="175"
                            value={height}
                            onChange={(e) => setHeight(e.target.value)}
                            className="w-full"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <button
                    onClick={handleTextSymptomSubmit}
                    className="w-full py-3 px-6 bg-gradient-to-r from-terracotta to-ochre text-white rounded-lg hover:opacity-90 transition-colors flex items-center justify-center"
                  >
                    <Search className="mr-2 h-4 w-4" />
                    Analyze Symptoms
                  </button>
                </div>
              </>
            ) : (
              <div className="animate-fade-in">
                <div className="mb-6">
                  <h1 className="text-2xl font-bold text-gray-900">Assessment Result</h1>
                  <p className="mt-2 text-gray-600">
                    Based on your symptom description, here's a preliminary assessment:
                  </p>
                </div>

                {formattedAnalysis ? (
                  <div className="mb-8 space-y-6">
                    {/* Possible Conditions Section */}
                    {formattedAnalysis.possibleConditions && formattedAnalysis.possibleConditions.length > 0 && (
                      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 rounded-lg border border-blue-100">
                        <h2 className="text-lg font-semibold text-blue-800 mb-3 flex items-center">
                          <AlertCircle className="h-5 w-5 mr-2 text-blue-700" />
                          Possible Conditions
                        </h2>
                        <ul className="space-y-2 pl-2">
                          {formattedAnalysis.possibleConditions.map((condition, idx) => (
                            <li key={idx} className="flex items-start">
                              <span className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-blue-200 text-blue-700 text-xs font-medium mr-2 mt-0.5">
                                {idx + 1}
                              </span>
                              <span className="text-blue-900">{condition}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Treatments Section */}
                    {formattedAnalysis.treatments && formattedAnalysis.treatments.length > 0 && (
                      <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-5 rounded-lg border border-purple-100">
                        <h2 className="text-lg font-semibold text-purple-800 mb-3 flex items-center">
                          <Pill className="h-5 w-5 mr-2 text-purple-700" />
                          Suggested Treatments
                        </h2>
                        <ul className="space-y-2 pl-2">
                          {formattedAnalysis.treatments.map((treatment, idx) => (
                            <li key={idx} className="flex items-start">
                              <span className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-purple-200 text-purple-700 text-xs font-medium mr-2 mt-0.5">
                                {idx + 1}
                              </span>
                              <span className="text-purple-900">{treatment}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Urgency Level Section */}
                    {formattedAnalysis.urgencyLevel && (
                      <div className={`p-5 rounded-lg border ${getUrgencyColor(formattedAnalysis.urgencyLevel)}`}>
                        <h2 className="text-lg font-semibold mb-3 flex items-center">
                          <Clock className="h-5 w-5 mr-2" />
                          Urgency Level
                        </h2>
                        <div className="font-medium pl-7">
                          {formattedAnalysis.urgencyLevel}
                        </div>
                      </div>
                    )}

                    {/* When to Seek Medical Help */}
                    {formattedAnalysis.seekMedicalHelp && (
                      <div className="bg-gradient-to-r from-amber-50 to-yellow-50 p-5 rounded-lg border border-amber-100">
                        <h2 className="text-lg font-semibold text-amber-800 mb-3 flex items-center">
                          <HelpCircle className="h-5 w-5 mr-2 text-amber-700" />
                          When to Seek Professional Help
                        </h2>
                        <div className="text-amber-900 prose max-w-none pl-7">
                          {formattedAnalysis.seekMedicalHelp.split('\n').map((paragraph, idx) => (
                            <p key={idx}>{paragraph}</p>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Disclaimer */}
                    <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                      <div className="flex">
                        <Info className="h-5 w-5 text-gray-500 mr-3 flex-shrink-0 mt-1" />
                        <div>
                          <h3 className="text-sm font-semibold text-gray-700 mb-1">Important Notice</h3>
                          <p className="text-xs text-gray-600 leading-relaxed">
                            This AI-generated assessment is for informational purposes only and should not replace professional medical advice. 
                            Always consult with a qualified healthcare provider for diagnosis and treatment of any medical condition.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-6 bg-red-50 rounded-lg border border-red-100 text-center">
                    <AlertCircle className="h-10 w-10 text-red-500 mx-auto mb-3" />
                    <h3 className="text-lg font-medium text-red-800 mb-2">Analysis Error</h3>
                    <p className="text-red-700">
                      There was an issue processing your symptoms. Please try again.
                    </p>
                  </div>
                )}

                <button
                  onClick={resetAssessment}
                  className="w-full mt-6 py-3 px-6 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-lg transition-colors flex items-center justify-center"
                >
                  <ArrowRight className="mr-2 h-4 w-4 rotate-180" />
                  Start a New Assessment
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
      
      <Footer />
    </div>
  );
};

export default SymptomChecker;
