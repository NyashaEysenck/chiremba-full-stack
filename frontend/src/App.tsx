import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import { AuthProvider, useAuth } from "./components/AuthContext";
import Index from "./pages/Index";
import Login from "./pages/Login";
import SymptomChecker from "./pages/SymptomChecker";
import Chatbot from "./pages/Chatbot";
import ImageDiagnosis from "./pages/ImageDiagnosis";
import AdminPanel from "./pages/AdminPanel";
import NotFound from "./pages/NotFound";
import SetupPassword from '@/pages/SetupPassword';
import { useEffect } from "react";
import { toast } from "@/hooks/use-toast";
import { ensureAdminUser } from "./utils/auth";

const queryClient = new QueryClient();

// Protected route component that requires authentication
const ProtectedRoute = ({ children, requiredRole = null }: { children: React.ReactNode, requiredRole?: 'admin' | 'staff' | null }) => {
  const { isAuthenticated, userRole, isLoading } = useAuth();
  const location = useLocation();
  
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      toast({
        title: "Authentication Required",
        description: "Please log in to access this page.",
        variant: "destructive",
      });
    } else if (!isLoading && isAuthenticated && requiredRole && requiredRole === 'admin' && userRole !== requiredRole) {
      // Only show access denied for admin pages
      toast({
        title: "Access Denied",
        description: `You need admin access for this page.`,
        variant: "destructive",
      });
    }
  }, [isAuthenticated, userRole, requiredRole, isLoading]);
  
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-terracotta"></div>
      </div>
    );
  }
  
  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  
  if (requiredRole && requiredRole === 'admin' && userRole !== requiredRole) {
    return <Navigate to="/" replace />;
  }
  
  return <>{children}</>;
};

const AppRoutes = () => {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();
  
  // When app initializes, ensure the admin user exists
  useEffect(() => {
    // Call the function to ensure admin user has proper role
    ensureAdminUser()
      .then(success => {
        if (success) {
          console.log("Admin user privileges confirmed");
        }
      })
      .catch(error => {
        console.error("Failed to set admin privileges:", error);
      });
  }, []);

  // Fix the scroll position when navigating to new pages
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.pathname]);
  
  // While checking authentication, show a loading indicator
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-terracotta"></div>
      </div>
    );
  }
  
  // If user is already logged in, redirect them away from login page or reset password page
  if (isAuthenticated && (location.pathname === '/login' || location.pathname === '/setup-password')) {
    return <Navigate to="/" replace />;
  }

  return (
    <Routes>
      <Route path="/" element={<Index />} />
      <Route path="/login" element={<Login />} />
      <Route path="/symptom-checker" element={<SymptomChecker />} />
      <Route path="/chatbot" element={<Chatbot />} />
      <Route 
        path="/image-diagnosis" 
        element={
          <ProtectedRoute>
            <ImageDiagnosis />
          </ProtectedRoute>
        } 
      />
      <Route 
        path="/admin" 
        element={
          <ProtectedRoute requiredRole="admin">
            <AdminPanel />
          </ProtectedRoute>
        } 
      />
      <Route path="/setup-password" element={<SetupPassword />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <AuthProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <AppRoutes />
        </BrowserRouter>
      </TooltipProvider>
    </AuthProvider>
  </QueryClientProvider>
);

export default App;
