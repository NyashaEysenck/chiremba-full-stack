
import { Heart } from "lucide-react";
import { useAuth } from "./AuthContext";
import { Link } from "react-router-dom";

const Footer = () => {
  const currentYear = new Date().getFullYear();
  const { isAuthenticated } = useAuth();

  return (
    <footer className="bg-gradient-to-r from-gray-50 to-gray-100 border-t border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Logo & Description */}
          <div className="col-span-1">
            <div className="flex items-center">
              <Heart className="h-6 w-6 text-terracotta" />
              <span className="ml-2 text-xl font-bold bg-gradient-to-r from-terracotta to-ochre bg-clip-text text-transparent">
                Chiremba
              </span>
            </div>
            <p className="mt-4 text-sm text-gray-600 max-w-xs">
              AI-powered health diagnosis system designed to improve healthcare accessibility
              in Zimbabwe and underserved areas.
            </p>
          </div>

          {/* Quick Links */}
          <div className="col-span-1">
            <h3 className="text-sm font-semibold text-gray-900 tracking-wider uppercase">
              Features
            </h3>
            <ul className="mt-4 space-y-2">
              <li>
                <Link to="/symptom-checker" className="text-sm text-gray-600 hover:text-terracotta transition-colors">
                  Symptom Checker
                </Link>
              </li>
              <li>
                <Link to="/chatbot" className="text-sm text-gray-600 hover:text-terracotta transition-colors">
                  Virtual Consultation
                </Link>
              </li>
              {isAuthenticated ? (
                <li>
                  <Link to="/image-diagnosis" className="text-sm text-gray-600 hover:text-terracotta transition-colors">
                    Image Diagnosis
                  </Link>
                </li>
              ) : (
                <li>
                  <span className="text-sm text-gray-400 cursor-not-allowed">
                    Image Diagnosis (Login Required)
                  </span>
                </li>
              )}
            </ul>
          </div>

          {/* Contact */}
          <div className="col-span-1">
            <h3 className="text-sm font-semibold text-gray-900 tracking-wider uppercase">
              Support
            </h3>
            <ul className="mt-4 space-y-2">
              <li>
                <a href="#" className="text-sm text-gray-600 hover:text-terracotta transition-colors">
                  Help Center
                </a>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-600 hover:text-terracotta transition-colors">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-600 hover:text-terracotta transition-colors">
                  Terms of Service
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-gray-200">
          <p className="text-center text-sm text-gray-600">
            &copy; {currentYear} Chiremba Health Diagnostics. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
