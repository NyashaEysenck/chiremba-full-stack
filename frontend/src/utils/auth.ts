import { mongoAPI } from '@/integrations/mongodb/client';

// Interface for user
export interface User {
  id: string;
  email: string;
  name: string;
  role: string;
  createdAt?: string;
  status?: string;
}

// Interface for auth response
export interface AuthResponse {
  success: boolean;
  user?: User;
  message?: string;
}

/**
 * Sign out the current user
 */
export const signOut = async (): Promise<{ success: boolean }> => {
  try {
    // Call the MongoDB API to logout
    mongoAPI.auth.logout();
    
    return { success: true };
  } catch (error) {
    console.error('Logout error:', error);
    return { success: false };
  }
};

/**
 * Create a new account
 */
export const createAccount = async (
  email: string, 
  password: string, 
  name: string,
  role: string = 'user'
): Promise<AuthResponse> => {
  try {
    // For staff/admin accounts, use the users API to create with specific role
    if (role === 'staff' || role === 'admin') {
      await mongoAPI.users.createUser({ email, password, name, role });
      return {
        success: true,
        message: `${role.charAt(0).toUpperCase() + role.slice(1)} account created successfully. Password setup email sent.`
      };
    } 
    // For regular users, use the register endpoint
    const response = await mongoAPI.auth.register(email, password, name);
    return {
      success: true,
      user: response.user
    };
  } catch (error: any) {
    console.error('Account creation error:', error);
    return {
      success: false,
      message: error.message || 'Failed to create account'
    };
  }
};

/**
 * Set up password using token
 */
export const setupPassword = async (token: string, password: string): Promise<{ success: boolean; message: string }> => {
  try {
    const apiUrl = import.meta.env.VITE_EXPRESS_API_URL || `http://${window.location.hostname}:5000`;
    const response = await fetch(`${apiUrl}/api/auth/setup-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token, password })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || 'Failed to set password');
    return { success: true, message: data.message };
  } catch (error: any) {
    return { success: false, message: error.message || 'Failed to set password' };
  }
};

/**
 * Delete a user account
 */
export const deleteUser = async (userId: string): Promise<{ success: boolean; message?: string }> => {
  try {
    // Call the MongoDB API to delete user
    await mongoAPI.users.deleteUser(userId);
    
    return {
      success: true,
      message: 'User deleted successfully'
    };
  } catch (error: any) {
    console.error('Delete user error:', error);
    return {
      success: false,
      message: error.message || 'Failed to delete user'
    };
  }
};

/**
 * Assign admin role to a user
 */
export const assignAdminRole = async (userId: string): Promise<{ success: boolean; message?: string }> => {
  try {
    // Call the MongoDB API to assign admin role
    await mongoAPI.users.assignRole(userId, 'admin');
    
    return {
      success: true,
      message: 'User promoted to admin successfully'
    };
  } catch (error: any) {
    console.error('Assign admin role error:', error);
    return {
      success: false,
      message: error.message || 'Failed to assign admin role'
    };
  }
};

/**
 * Reset a user's account (set password to null, status to pending, resend setup email)
 */
export const resetUserAccount = async (userId: string): Promise<{ success: boolean; message?: string }> => {
  try {
    // Call the MongoDB API to reset user account
    await mongoAPI.users.resetUserAccount(userId);
    return {
      success: true,
      message: 'User account reset and setup email resent.'
    };
  } catch (error: any) {
    console.error('Reset user account error:', error);
    return {
      success: false,
      message: error.message || 'Failed to reset user account'
    };
  }
};

/**
 * Get the current user
 */
export const getCurrentUser = async (): Promise<User | null> => {
  try {
    // Call the MongoDB API to get current user
    const user = await mongoAPI.auth.getCurrentUser();
    return user;
  } catch (error) {
    console.error('Get current user error:', error);
    return null;
  }
};

/**
 * Ensure that the admin user exists in the system
 * This is called when the app initializes to make sure there's always an admin user
 */
export const ensureAdminUser = async (): Promise<boolean> => {
  try {
    // Use environment variable with fallback for the API URL
    const apiUrl = import.meta.env.PROD
      ? `${import.meta.env.VITE_EXPRESS_API_URL}/api`
      : import.meta.env.VITE_EXPRESS_API_URL 
        ? `${import.meta.env.VITE_EXPRESS_API_URL}/api`
        : `http://${window.location.hostname}:5000/api`;
        
    // Call the init-admin endpoint to ensure admin user exists
    const response = await fetch(`${apiUrl}/init-admin`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (!response.ok) {
      throw new Error('Failed to initialize admin user');
    }
    
    const data = await response.json();
    console.log('Admin user initialization:', data.message);
    return true;
  } catch (error) {
    console.error('Error ensuring admin user:', error);
    return false;
  }
};

/**
 * Login a user with email and password
 */
export const login = async (email: string, password: string): Promise<{ success: boolean; user?: User; message?: string }> => {
  try {
    const response = await mongoAPI.auth.login(email, password);
    return {
      success: true,
      user: response.user
    };
  } catch (error: any) {
    let errorMessage = 'Invalid email or password. Please try again.';
    // Axios error: check for error.response.data.message
    if (error?.response?.data?.message) {
      errorMessage = error.response.data.message;
    } else if (error instanceof Error) {
      errorMessage = error.message;
    } else if (typeof error === 'string') {
      errorMessage = error;
    } else if (error && typeof error === 'object' && 'message' in error) {
      errorMessage = error.message as string;
    }
    return {
      success: false,
      message: errorMessage
    };
  }
};
