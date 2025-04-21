import { mongoAPI } from '@/integrations/mongodb/client';

// Interface for user
export interface User {
  id: string;
  email: string;
  name: string;
  role: string;
  createdAt?: string;
}

// Interface for auth response
export interface AuthResponse {
  success: boolean;
  user?: User;
  message?: string;
}

/**
 * Sign in a user with email and password
 */
export const signIn = async (email: string, password: string): Promise<AuthResponse> => {
  try {
    // Call the MongoDB API to login
    const response = await mongoAPI.auth.login(email, password);
    
    return {
      success: true,
      user: response.user
    };
  } catch (error: any) {
    console.error('Login error:', error);
    
    // Extract the most user-friendly error message
    let errorMessage = 'Invalid email or password';
    
    if (error instanceof Error) {
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
        message: `${role.charAt(0).toUpperCase() + role.slice(1)} account created successfully`
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

// Function to fetch config from backend
async function getConfig() {
  const response = await fetch('/api/config');
  return response.json();
}

let EXPRESS_API_URL = '';

getConfig().then(config => {
  EXPRESS_API_URL = config.EXPRESS_API_URL || '';
});

/**
 * Ensure that the admin user exists in the system
 * This is called when the app initializes to make sure there's always an admin user
 */
export const ensureAdminUser = async (): Promise<boolean> => {
  try {
    // Use environment variable with fallback for the API URL
    const apiUrl = import.meta.env.PROD
      ? `${EXPRESS_API_URL}/api`
      : EXPRESS_API_URL
        ? `${EXPRESS_API_URL}/api`
        : '';
        
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
