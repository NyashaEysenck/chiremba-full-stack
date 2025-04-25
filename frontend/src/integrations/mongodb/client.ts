import axios from 'axios';

// Define the API base URL using environment variable with fallback
const API_BASE_URL = import.meta.env.PROD 
  ? `${import.meta.env.VITE_EXPRESS_API_URL}/api` 
  : import.meta.env.VITE_EXPRESS_API_URL 
    ? `${import.meta.env.VITE_EXPRESS_API_URL}/api`
    : `http://${window.location.hostname}:5000/api`;

// User interface
export interface User {
  id: string;
  _id?: string;
  email: string;
  name: string;
  role: string;
  createdAt?: string;
  status?: string;
}

// Authentication API
const authAPI = {
  // Login with email and password
  login: async (email: string, password: string) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/auth/login`, { email, password });
      
      // Store the token in localStorage
      if (response.data.token) {
        localStorage.setItem('auth_token', response.data.token);
      }
      
      return response.data;
    } catch (error) {
      console.error('Login API error:', error);
      
      // Handle specific error types
      if (axios.isAxiosError(error)) {
        if (error.response) {
          // The request was made and the server responded with a status code
          // that falls out of the range of 2xx
          const statusCode = error.response.status;
          const errorData = error.response.data;
          // Always throw the backend message if present
          if (errorData && errorData.message) {
            throw new Error(errorData.message);
          }
          if (statusCode === 400) {
            throw new Error('Invalid email or password');
          } else if (statusCode === 401) {
            throw new Error('Unauthorized: Please check your credentials');
          } else if (statusCode === 403) {
            throw new Error('Access denied');
          } else if (statusCode >= 500) {
            throw new Error('Server error: Please try again later');
          }
        } else if (error.request) {
          // The request was made but no response was received
          throw new Error('No response from server. Please check your connection');
        }
      }
      
      // Generic error
      throw new Error('Login failed. Please try again.');
    }
  },
  
  // Register a new user
  register: async (email: string, password: string, name: string) => {
    const response = await axios.post(`${API_BASE_URL}/auth/register`, { email, password, name });
    
    // Store the token in localStorage
    if (response.data.token) {
      localStorage.setItem('auth_token', response.data.token);
    }
    
    return response.data;
  },
  
  // Get the current user
  getCurrentUser: async () => {
    const token = localStorage.getItem('auth_token');
    
    if (!token) {
      return null;
    }
    
    try {
      const response = await axios.get(`${API_BASE_URL}/auth/me`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      
      // Check if response contains user data in the expected format
      if (response.data && response.data.user) {
        return response.data.user;
      } else if (response.data && response.data._id) {
        // Handle legacy format for backward compatibility
        return {
          id: response.data._id,
          email: response.data.email,
          name: response.data.name,
          role: response.data.role,
          createdAt: response.data.createdAt
        };
      } else {
        console.error('Unexpected user data format:', response.data);
        return null;
      }
    } catch (error) {
      console.error('Error getting current user:', error);
      // Clear token if it's invalid
      localStorage.removeItem('auth_token');
      return null;
    }
  },
  
  // Logout the current user
  logout: () => {
    localStorage.removeItem('auth_token');
  }
};

// Users API
const usersAPI = {
  // Get all users (admin only)
  getAllUsers: async () => {
    const token = localStorage.getItem('auth_token');
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    try {
      const response = await axios.get(`${API_BASE_URL}/users`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      
      // Handle both formats: {users: [...]} and direct array
      const usersData = response.data.users || response.data;
      
      // Check if we have a valid array
      if (!Array.isArray(usersData)) {
        console.error('Unexpected users data format:', response.data);
        return [];
      }
      
      // Transform the user data to match our User interface
      return usersData.map((user: any) => ({
        id: user._id,
        email: user.email,
        name: user.name,
        role: user.role,
        createdAt: user.createdAt,
        status: user.status
      }));
    } catch (error) {
      console.error('Error fetching users:', error);
      throw error;
    }
  },
  
  // Create a new user with specific role (admin only)
  createUser: async (userData: { email: string; password: string; name: string; role: string }) => {
    const token = localStorage.getItem('auth_token');
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    // Use the correct endpoint based on the role
    const endpoint = userData.role === 'admin' ? '/users/admin' : '/users/staff';
    
    const response = await axios.post(`${API_BASE_URL}${endpoint}`, userData, {
      headers: {
        Authorization: `Bearer ${token}`
      }
    });
    
    return response.data;
  },
  
  // Delete a user (admin only)
  deleteUser: async (userId: string) => {
    const token = localStorage.getItem('auth_token');
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    const response = await axios.delete(`${API_BASE_URL}/users/${userId}`, {
      headers: {
        Authorization: `Bearer ${token}`
      }
    });
    
    return response.data;
  },
  
  // Assign a role to a user (admin only)
  assignRole: async (userId: string, role: string) => {
    const token = localStorage.getItem('auth_token');
    
    if (!token) {
      throw new Error('Authentication required');
    }
    
    const response = await axios.patch(`${API_BASE_URL}/users/${userId}/role`, { role }, {
      headers: {
        Authorization: `Bearer ${token}`
      }
    });
    
    return response.data;
  },
  
  // Reset a user's account (admin only)
  resetUserAccount: async (userId: string) => {
    const token = localStorage.getItem('auth_token');
    if (!token) {
      throw new Error('Authentication required');
    }
    // This assumes a backend endpoint exists for resetting a user account
    // POST /users/:userId/reset-account
    const response = await axios.post(`${API_BASE_URL}/users/${userId}/reset-account`, {}, {
      headers: {
        Authorization: `Bearer ${token}`
      }
    });
    return response.data;
  }
};

// Export the MongoDB API
export const mongoAPI = {
  auth: authAPI,
  users: usersAPI
};
