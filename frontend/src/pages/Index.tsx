import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '@/components/AuthContext';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { useToast } from '@/hooks/use-toast';
import { 
  MessageSquare, 
  Stethoscope, 
  Image, 
  Shield, 
  ArrowRight,
  Heart,
  ChevronRight
} from 'lucide-react';

const Index = () => {
  const { isAuthenticated } = useAuth();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  // This handler is not needed anymore since we're conditionally rendering the item
  // based on authentication status rather than showing a toast
  const handleRestrictedAccess = () => {
    if (!isAuthenticated) {
      toast({
        title: "Access Restricted",
        description: "You need to be logged in to access this feature.",
        variant: "destructive",
      });
      return false;
    }
    return true;
  };

  if (isLoading) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-background">
        <div className="flex flex-col items-center">
          <Heart className="h-16 w-16 text-terracotta animate-pulse" />
          <h2 className="mt-4 text-xl font-semibold bg-gradient-to-r from-terracotta to-ochre bg-clip-text text-transparent animate-pulse">
            Chiremba Health
          </h2>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col pattern-bg">
      <Navbar />
      
      {/* Hero Section */}
      <section className="pt-32 pb-20 md:pt-40 md:pb-28">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center">
            <div className="md:w-1/2 md:pr-12">
              <div className="animate-fade-in">
                <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold leading-tight text-gray-900">
                  <span className="bg-gradient-to-r from-terracotta to-ochre bg-clip-text text-transparent">
                    AI-Powered
                  </span>{" "}
                  Healthcare Diagnosis
                </h1>
                <p className="mt-6 text-lg md:text-xl text-gray-600 max-w-2xl animate-slide-up" style={{ animationDelay: "0.2s" }}>
                  Chiremba bridges the gap in healthcare accessibility in Zimbabwe,
                  bringing advanced diagnostics to underserved areas.
                </p>
                <div className="mt-8 flex flex-wrap gap-4 animate-slide-up" style={{ animationDelay: "0.4s" }}>
                  <Link 
                    to="/symptom-checker" 
                    className="btn-primary"
                  >
                    Start Diagnosis
                  </Link>
                  <Link 
                    to="/chatbot" 
                    className="btn-secondary"
                  >
                    Virtual Consultation
                  </Link>
                </div>
              </div>
            </div>
            <div className="md:w-1/2 mt-12 md:mt-0 animate-fade-in" style={{ animationDelay: "0.6s" }}>
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-terracotta/20 to-ochre/20 rounded-2xl transform rotate-3"></div>
                <div className="relative bg-white p-6 rounded-2xl shadow-xl transform -rotate-2 transition-all hover:rotate-0 duration-500">
                  <div className="aspect-w-16 aspect-h-9 rounded-lg overflow-hidden">
                    <img 
                      src="https://www.aamc.org/sites/default/files/styles/scale_and_crop_1200_x_666/public/female-doctor-with-patient-1393489803.jpg?itok=_rDboM2g" 
                      alt="Healthcare in Africa" 
                      className="object-cover w-full h-full rounded-lg"
                    />
                  </div>
                  <div className="mt-4 flex items-center justify-between">
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900">Accessible Healthcare</h3>
                      <p className="text-sm text-gray-600">Bringing advanced diagnostics to all</p>
                    </div>
                    <Shield className="h-8 w-8 text-terracotta" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900">
              Powerful AI-Driven Features
            </h2>
            <p className="mt-4 text-lg text-gray-600 max-w-2xl mx-auto">
              Chiremba combines cutting-edge AI with healthcare expertise to provide
              accurate diagnostics and consultations.
            </p>
          </div>

          <div className="mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <Link 
              to="/symptom-checker"
              className="bg-gray-50 p-6 rounded-xl shadow-sm card-hover"
            >
              <div className="h-12 w-12 bg-terracotta/10 flex items-center justify-center rounded-lg mb-4">
                <Stethoscope className="h-6 w-6 text-terracotta" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900">Symptom Checker</h3>
              <p className="mt-2 text-gray-600">
                Answer questions about your symptoms to receive preliminary diagnosis and advice.
              </p>
              <div className="mt-4 flex items-center text-terracotta font-medium">
                <span>Get started</span>
                <ChevronRight className="h-4 w-4 ml-1" />
              </div>
            </Link>

            {/* Feature 2 */}
            <Link 
              to="/chatbot" 
              className="bg-gray-50 p-6 rounded-xl shadow-sm card-hover"
            >
              <div className="h-12 w-12 bg-ochre/10 flex items-center justify-center rounded-lg mb-4">
                <MessageSquare className="h-6 w-6 text-ochre" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900">Virtual Consultation</h3>
              <p className="mt-2 text-gray-600">
                Chat with our AI assistant for personalized health guidance with voice support.
              </p>
              <div className="mt-4 flex items-center text-ochre font-medium">
                <span>Start consulting</span>
                <ChevronRight className="h-4 w-4 ml-1" />
              </div>
            </Link>

            {/* Feature 3 - Only show Image Diagnosis for authenticated users */}
            {isAuthenticated ? (
              <Link 
                to="/image-diagnosis"
                className="bg-gray-50 p-6 rounded-xl shadow-sm card-hover"
              >
                <div className="h-12 w-12 bg-sienna/10 flex items-center justify-center rounded-lg mb-4">
                  <Image className="h-6 w-6 text-sienna" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900">Image Diagnosis</h3>
                <p className="mt-2 text-gray-600">
                  Upload medical images for AI analysis and detection of potential conditions.
                </p>
                <div className="mt-4 flex items-center text-sienna font-medium">
                  <span>Upload images</span>
                  <ChevronRight className="h-4 w-4 ml-1" />
                </div>
              </Link>
            ) : (
              <div className="bg-gray-50 p-6 rounded-xl shadow-sm opacity-60 pointer-events-none">
                <div className="h-12 w-12 bg-gray-200 flex items-center justify-center rounded-lg mb-4">
                  <Shield className="h-6 w-6 text-gray-400" />
                </div>
                <h3 className="text-xl font-semibold text-gray-400">Premium Feature</h3>
                <p className="mt-2 text-gray-400">
                  Log in to unlock additional healthcare features and services.
                </p>
                <div className="mt-4 flex items-center text-gray-400">
                  <span>Login required</span>
                  <ChevronRight className="h-4 w-4 ml-1" />
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Testimonials/Info Section */}
      <section className="py-20 bg-gradient-to-b from-gray-50 to-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col lg:flex-row items-center lg:items-start gap-12">
            <div className="lg:w-1/2 order-2 lg:order-1">
              <div className="p-1 bg-gradient-to-r from-terracotta to-ochre rounded-xl">
                <div className="bg-white p-8 rounded-lg">
                  <h2 className="text-3xl font-bold text-gray-900">
                    Improving Healthcare Access in Zimbabwe
                  </h2>
                  <p className="mt-6 text-gray-600">
                    Zimbabwe faces significant healthcare challenges, with limited access to
                    medical professionals, especially in rural areas. Chiremba helps bridge
                    this gap through technology.
                  </p>

                  <div className="mt-8 space-y-4">
                    <div className="flex items-start">
                      <div className="flex-shrink-0">
                        <div className="flex items-center justify-center h-6 w-6 rounded-md bg-terracotta text-white">
                          <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </div>
                      </div>
                      <div className="ml-4">
                        <h3 className="text-lg font-medium text-gray-900">Advanced Diagnostics</h3>
                        <p className="mt-1 text-gray-600">AI-powered diagnosis available to everyone with internet access.</p>
                      </div>
                    </div>

                    <div className="flex items-start">
                      <div className="flex-shrink-0">
                        <div className="flex items-center justify-center h-6 w-6 rounded-md bg-terracotta text-white">
                          <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </div>
                      </div>
                      <div className="ml-4">
                        <h3 className="text-lg font-medium text-gray-900">Reducing Wait Times</h3>
                        <p className="mt-1 text-gray-600">Immediate health guidance without long waits or travel to clinics.</p>
                      </div>
                    </div>

                    <div className="flex items-start">
                      <div className="flex-shrink-0">
                        <div className="flex items-center justify-center h-6 w-6 rounded-md bg-terracotta text-white">
                          <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </div>
                      </div>
                      <div className="ml-4">
                        <h3 className="text-lg font-medium text-gray-900">Empowering Communities</h3>
                        <p className="mt-1 text-gray-600">Providing healthcare knowledge to underserved areas.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="lg:w-1/2 order-1 lg:order-2">
              <div className="grid grid-cols-2 gap-4">
                <div className="aspect-w-1 aspect-h-1">
                  <img 
                    src="https://www.aamc.org/sites/default/files/styles/scale_and_crop_1200_x_666/public/female-doctor-with-patient-1393489803.jpg?itok=_rDboM2g" 
                    alt="African healthcare professional with patient" 
                    className="object-cover w-full h-full rounded-lg shadow-md"
                  />
                </div>
                <div className="aspect-w-1 aspect-h-1 mt-8">
                  <img 
                    src="https://assets.medpagetoday.net/media/images/88xxx/88176.jpg" 
                    alt="Technology in healthcare" 
                    className="object-cover w-full h-full rounded-lg shadow-md"
                  />
                </div>
                <div className="aspect-w-1 aspect-h-1 col-span-2">
                  <img 
                    src="https://www.aljazeera.com/wp-content/uploads/2021/02/1299129881.jpg?w=770&resize=770%2C513" 
                    alt="Healthcare in Zimbabwe" 
                    className="object-cover w-full h-full rounded-lg shadow-md"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-terracotta/10 to-ochre/10">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900">
            Ready to experience the future of healthcare?
          </h2>
          <p className="mt-4 text-lg text-gray-600 max-w-2xl mx-auto">
            Start using Chiremba today and take control of your health with
            AI-powered diagnostics and virtual consultations.
          </p>
          <div className="mt-10 flex flex-wrap justify-center gap-4">
            <Link to="/symptom-checker" className="btn-primary">
              Start Symptom Check
            </Link>
            <Link to="/chatbot" className="btn-secondary">
              Talk to Chiremba AI
            </Link>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Index;
