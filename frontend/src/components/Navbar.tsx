
import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from './AuthContext';
import { cn } from '@/lib/utils';
import { Menu, X, Heart, LogOut, LogIn, Shield } from 'lucide-react';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const { isAuthenticated, userRole, logout } = useAuth();
  const location = useLocation();

  // Handle scroll effect
  useEffect(() => {
    const handleScroll = () => {
      const offset = window.scrollY;
      if (offset > 50) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close mobile menu when route changes
  useEffect(() => {
    setIsOpen(false);
  }, [location.pathname]);

  return (
    <nav
      className={cn(
        'fixed top-0 left-0 right-0 z-50 transition-all duration-300 backdrop-blur-md',
        scrolled 
          ? 'py-2 bg-background/80 shadow-md' 
          : 'py-4 bg-transparent'
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <Link to="/" className="flex items-center">
              <Heart className="h-8 w-8 text-terracotta" />
              <span className="ml-2 text-2xl font-bold bg-gradient-to-r from-terracotta to-ochre bg-clip-text text-transparent">
                Chiremba
              </span>
            </Link>
          </div>

          {/* Desktop menu */}
          <div className="hidden md:flex items-center space-x-6">
            <Link 
              to="/" 
              className="text-foreground/80 hover:text-terracotta transition-colors"
            >
              Home
            </Link>
            <Link 
              to="/symptom-checker" 
              className="text-foreground/80 hover:text-terracotta transition-colors"
            >
              Symptom Checker
            </Link>
            <Link 
              to="/chatbot" 
              className="text-foreground/80 hover:text-terracotta transition-colors"
            >
              Virtual Consultation
            </Link>

            {/* Protected routes for authenticated users */}
            {isAuthenticated && (
              <Link 
                to="/image-diagnosis" 
                className="text-foreground/80 hover:text-terracotta transition-colors"
              >
                Image Diagnosis
              </Link>
            )}
            
            {/* Admin panel link - display only for users with admin role */}
            {isAuthenticated && userRole === 'admin' && (
              <Link 
                to="/admin" 
                className="text-foreground/80 hover:text-terracotta transition-colors flex items-center"
              >
                <Shield className="h-4 w-4 mr-1" />
                Admin
              </Link>
            )}

            {isAuthenticated ? (
              <button 
                onClick={logout}
                className="flex items-center px-4 py-2 bg-gradient-to-r from-terracotta to-ochre text-white rounded-md"
              >
                <LogOut className="h-4 w-4 mr-1" />
                Logout
              </button>
            ) : (
              <Link 
                to="/login"
                className="flex items-center px-4 py-2 bg-gradient-to-r from-terracotta to-ochre text-white rounded-md"
              >
                <LogIn className="h-4 w-4 mr-1" />
                Login
              </Link>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              type="button"
              className="text-foreground p-2"
              onClick={() => setIsOpen(!isOpen)}
            >
              {isOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      <div
        className={cn(
          'md:hidden absolute top-full left-0 right-0 bg-background/95 backdrop-blur-md shadow-lg transition-all duration-300 ease-in-out border-t border-border overflow-hidden',
          isOpen ? 'max-h-screen opacity-100' : 'max-h-0 opacity-0'
        )}
      >
        <div className="px-4 py-4 space-y-3">
          <Link 
            to="/" 
            className="block py-2 text-foreground/80 hover:text-terracotta"
          >
            Home
          </Link>
          <Link 
            to="/symptom-checker" 
            className="block py-2 text-foreground/80 hover:text-terracotta"
          >
            Symptom Checker
          </Link>
          <Link 
            to="/chatbot" 
            className="block py-2 text-foreground/80 hover:text-terracotta"
          >
            Virtual Consultation
          </Link>

          {/* Protected routes for authenticated users */}
          {isAuthenticated && (
            <Link 
              to="/image-diagnosis" 
              className="block py-2 text-foreground/80 hover:text-terracotta"
            >
              Image Diagnosis
            </Link>
          )}
          
          {/* Admin panel link - display only for users with admin role */}
          {isAuthenticated && userRole === 'admin' && (
            <Link 
              to="/admin" 
              className="block py-2 text-foreground/80 hover:text-terracotta flex items-center"
            >
              <Shield className="h-4 w-4 mr-1" />
              Admin
            </Link>
          )}

          {isAuthenticated ? (
            <button 
              onClick={logout}
              className="flex items-center px-4 py-2 w-full bg-gradient-to-r from-terracotta to-ochre text-white rounded-md mt-4"
            >
              <LogOut className="h-4 w-4 mr-1" />
              Logout
            </button>
          ) : (
            <Link 
              to="/login"
              className="flex items-center justify-center px-4 py-2 w-full bg-gradient-to-r from-terracotta to-ochre text-white rounded-md mt-4"
            >
              <LogIn className="h-4 w-4 mr-1" />
              Login
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
