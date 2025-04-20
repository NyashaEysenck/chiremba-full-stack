import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { User, signIn, signOut, getCurrentUser } from '@/utils/auth';

// Define the auth context type
interface AuthContextType {
  user: User | null;
  userRole: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => Promise<void>;
}

// Create the auth context
const AuthContext = createContext<AuthContextType>({
  user: null,
  userRole: null,
  isAuthenticated: false,
  isLoading: true,
  login: async () => false,
  logout: async () => {},
});

// Auth provider props
interface AuthProviderProps {
  children: ReactNode;
}

// Auth provider component
export const AuthProvider = ({ children }: AuthProviderProps) => {
  const [user, setUser] = useState<User | null>(null);
  const [userRole, setUserRole] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check for existing session on mount
  useEffect(() => {
    const checkSession = async () => {
      try {
        const currentUser = await getCurrentUser();
        
        if (currentUser) {
          setUser(currentUser);
          setUserRole(currentUser.role);
        } else {
          setUser(null);
          setUserRole(null);
        }
      } catch (error) {
        console.error('Session check error:', error);
        setUser(null);
        setUserRole(null);
      } finally {
        setIsLoading(false);
      }
    };

    checkSession();
  }, []);

  // Login function
  const login = async (email: string, password: string): Promise<boolean> => {
    setIsLoading(true);
    
    try {
      const response = await signIn(email, password);
      
      if (response.success && response.user) {
        setUser(response.user);
        setUserRole(response.user.role);
        return true;
      } else {
        return false;
      }
    } catch (error) {
      console.error('Login error:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  // Logout function
  const logout = async (): Promise<void> => {
    setIsLoading(true);
    
    try {
      await signOut();
      setUser(null);
      setUserRole(null);
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Auth context value
  const value = {
    user,
    userRole,
    isAuthenticated: !!user,
    isLoading,
    login,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Custom hook to use the auth context
export const useAuth = () => useContext(AuthContext);
