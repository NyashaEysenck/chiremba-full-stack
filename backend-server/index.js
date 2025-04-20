import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import dotenv from 'dotenv';
import { body, validationResult } from 'express-validator';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import path from 'path';

import { spawn } from 'child_process';

// Get the directory path for the current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables from the root directory
dotenv.config( '.env');

// Verify environment variables are loaded
console.log('Environment variables loaded:', {
  MONGODB_URI: process.env.MONGODB_URI ? 'Defined' : 'Not defined',
  JWT_SECRET: process.env.JWT_SECRET ? 'Defined' : 'Not defined',
  PORT: process.env.PORT
});

// Initialize Express app
const app = express();
const FRONTEND_ORIGINS = ['https://chiremba-ai-frontend-production.up.railway.app', 'http://localhost:5173'];

app.use(cors({
  origin: FRONTEND_ORIGINS,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));

app.options('*', cors()); // Enable preflight for all routes

app.use(express.json());
 
// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/ai-symptom-solutions')
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('MongoDB connection error:', err));

// Define User Schema
const userSchema = new mongoose.Schema({
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  name: { type: String, required: true },
  role: { type: String, enum: ['user', 'staff', 'admin'], default: 'user' },
  createdAt: { type: Date, default: Date.now }
});

const User = mongoose.model('User', userSchema);

// Middleware to verify JWT token
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  
  if (!token) return res.status(401).json({ message: 'Access denied. No token provided.' });
  
  try {
    const verified = jwt.verify(token, process.env.JWT_SECRET);
    req.user = verified;
    next();
  } catch (error) {
    res.status(400).json({ message: 'Invalid token' });
  }
};

// Middleware to check admin role
const isAdmin = (req, res, next) => {
  if (req.user.role !== 'admin') {
    return res.status(403).json({ message: 'Access denied. Admin privileges required.' });
  }
  next();
};

// Middleware to check staff or admin role
const isStaffOrAdmin = (req, res, next) => {
  if (req.user.role !== 'staff' && req.user.role !== 'admin') {
    return res.status(403).json({ message: 'Access denied. Staff privileges required.' });
  }
  next();
};

// API Routes

// Register a new user
app.post('/api/auth/register', [
  body('email').isEmail().withMessage('Enter a valid email'),
  body('password').isLength({ min: 6 }).withMessage('Password must be at least 6 characters'),
  body('name').notEmpty().withMessage('Name is required')
], async (req, res) => {
  // Validate request
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }

  try {
    // Check if user already exists
    const emailExists = await User.findOne({ email: req.body.email });
    if (emailExists) {
      return res.status(400).json({ message: 'Email already exists' });
    }

    // Hash the password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(req.body.password, salt);

    // Create new user
    const user = new User({
      email: req.body.email,
      password: hashedPassword,
      name: req.body.name,
      role: req.body.role || 'user'
    });

    // Save user to database
    const savedUser = await user.save();
    
    // Create and assign token
    const token = jwt.sign(
      { id: savedUser._id, email: savedUser.email, role: savedUser.role },
      process.env.JWT_SECRET,
      { expiresIn: '1d' }
    );

    res.status(201).json({
      token,
      user: {
        id: savedUser._id,
        email: savedUser.email,
        name: savedUser.name,
        role: savedUser.role
      }
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Login user
app.post('/api/auth/login', async (req, res) => {
  try {
    // Check if user exists
    const user = await User.findOne({ email: req.body.email });
    if (!user) {
      return res.status(400).json({ message: 'Invalid email or password' });
    }

    // Check password
    const validPassword = await bcrypt.compare(req.body.password, user.password);
    if (!validPassword) {
      return res.status(400).json({ message: 'Invalid email or password' });
    }

    // Verify JWT_SECRET is available
    if (!process.env.JWT_SECRET) {
      console.error('JWT_SECRET is not defined in environment variables');
      return res.status(500).json({ message: 'Server configuration error' });
    }

    // Create and assign token
    const token = jwt.sign(
      { id: user._id, email: user.email, role: user.role },
      process.env.JWT_SECRET,
      { expiresIn: '1d' }
    );

    res.status(200).json({
      token,
      user: {
        id: user._id,
        email: user.email,
        name: user.name,
        role: user.role
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ message: error.message });
  }
});

// Get current user
app.get('/api/auth/me', authenticateToken, async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select('-password');
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }
    res.status(200).json({
      user: {
        id: user._id,
        email: user.email,
        name: user.name,
        role: user.role,
        createdAt: user.createdAt
      }
    });
  } catch (error) {
    console.error('Get current user error:', error);
    res.status(500).json({ message: error.message });
  }
});

// Get all users (admin only)
app.get('/api/users', authenticateToken, isAdmin, async (req, res) => {
  try {
    const users = await User.find().select('-password');
    res.status(200).json({ users });
  } catch (error) {
    console.error('Get all users error:', error);
    res.status(500).json({ message: error.message });
  }
});

// Get all staff users (admin only)
app.get('/api/users/staff', authenticateToken, isAdmin, async (req, res) => {
  try {
    const staffUsers = await User.find({ role: 'staff' }).select('-password');
    res.status(200).json(staffUsers);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Create a staff user (admin only)
app.post('/api/users/staff', authenticateToken, isAdmin, [
  body('email').isEmail().withMessage('Enter a valid email'),
  body('password').isLength({ min: 6 }).withMessage('Password must be at least 6 characters'),
  body('name').notEmpty().withMessage('Name is required')
], async (req, res) => {
  // Validate request
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }

  try {
    // Check if user already exists
    const emailExists = await User.findOne({ email: req.body.email });
    if (emailExists) {
      return res.status(400).json({ message: 'Email already exists' });
    }

    // Hash the password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(req.body.password, salt);

    // Create new staff user
    const user = new User({
      email: req.body.email,
      password: hashedPassword,
      name: req.body.name,
      role: 'staff'
    });

    // Save user to database
    const savedUser = await user.save();
    
    res.status(201).json({
      user: {
        id: savedUser._id,
        email: savedUser.email,
        name: savedUser.name,
        role: savedUser.role
      }
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Delete a user (admin only)
app.delete('/api/users/:id', authenticateToken, isAdmin, async (req, res) => {
  try {
    const user = await User.findById(req.params.id);
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    // Prevent deleting the last admin
    if (user.role === 'admin') {
      const adminCount = await User.countDocuments({ role: 'admin' });
      if (adminCount <= 1) {
        return res.status(400).json({ message: 'Cannot delete the last admin user' });
      }
    }

    await User.findByIdAndDelete(req.params.id);
    res.status(200).json({ message: 'User deleted successfully' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Update user role (admin only)
app.patch('/api/users/:id/role', authenticateToken, isAdmin, async (req, res) => {
  try {
    const { role } = req.body;
    if (!role || !['user', 'staff', 'admin'].includes(role)) {
      return res.status(400).json({ message: 'Invalid role' });
    }

    const user = await User.findById(req.params.id);
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    // Prevent removing the last admin
    if (user.role === 'admin' && role !== 'admin') {
      const adminCount = await User.countDocuments({ role: 'admin' });
      if (adminCount <= 1) {
        return res.status(400).json({ message: 'Cannot remove the last admin' });
      }
    }

    user.role = role;
    await user.save();

    res.status(200).json({
      user: {
        id: user._id,
        email: user.email,
        name: user.name,
        role: user.role
      }
    });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Initialize admin user
app.post('/api/init-admin', async (req, res) => {
  try {
    // Check if admin already exists
    const adminExists = await User.findOne({ role: 'admin' });
    if (adminExists) {
      return res.status(200).json({ message: 'Admin user already exists' });
    }

    // Create admin user
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash('admin123', salt);

    const adminUser = new User({
      email: 'admin@chiremba.com',
      password: hashedPassword,
      name: 'Admin User',
      role: 'admin'
    });

    await adminUser.save();

    // Create staff user
    const staffPassword = await bcrypt.hash('staff123', salt);
    const staffUser = new User({
      email: 'staff@chiremba.com',
      password: staffPassword,
      name: 'Staff User',
      role: 'staff'
    });

    await staffUser.save();

    res.status(201).json({ message: 'Admin and staff users initialized successfully' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});


// 1. Start FastAPI Server
const startFastAPI = () => {
  const fastApiPath = path.join(__dirname, '../backend-image-diagnosis/hostlocal.py');
  
  const fastApiProcess = spawn('python', [fastApiPath], {
    stdio: 'inherit',
    shell: true // Needed for Windows compatibility
  });

  fastApiProcess.on('error', (err) => {
    console.error('Failed to start FastAPI:', err);
  });

  fastApiProcess.on('exit', (code) => {
    console.log(`FastAPI process exited with code ${code}`);
  });

  return fastApiProcess;
};

const fastApiProcess = startFastAPI();

// 2. Cleanup on Express shutdown
process.on('exit', () => {
  fastApiProcess.kill();
  console.log('Terminated FastAPI server');
});


// Serve static files from the frontend build
app.use(express.static(path.join(__dirname, '../frontend/dist')));


// Handle SPA client-side routing - return index.html for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/dist', 'index.html'));
});


// Start server
const PORT = process.env.PORT || 5000;
const HOST = process.env.EXPRESS_HOST || '0.0.0.0';

app.listen(PORT, HOST, () => {
  console.log(`Server is running on port ${PORT} and accessible on the network at http://${HOST}:${PORT}`);
});
