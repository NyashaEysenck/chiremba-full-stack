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
  password: { type: String, required: false },
  name: { type: String, required: true },
  role: { type: String, enum: ['user', 'staff', 'admin'], default: 'user' },
  status: { type: String, enum: ['pending', 'active'], default: 'pending' },
  passwordResetToken: { type: String },
  passwordResetExpires: { type: Date },
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

// Email sending setup
import nodemailer from 'nodemailer';

const EMAIL_HOST = process.env.EMAIL_HOST;
const EMAIL_PORT = process.env.EMAIL_PORT;
const EMAIL_USER = process.env.EMAIL_USER;
const EMAIL_PASS = process.env.EMAIL_PASS;
const EMAIL_FROM = process.env.EMAIL_FROM || 'no-reply@chiremba.com';
const FRONTEND_URL = process.env.VITE_EXPRESS_API_URL || 'http://localhost:5173';

const transporter = nodemailer.createTransport({
  host: EMAIL_HOST,
  port: EMAIL_PORT,
  secure: EMAIL_PORT == 465, // true for 465, false for other ports
  auth: {
    user: EMAIL_USER,
    pass: EMAIL_PASS,
  },
  tls: {
    rejectUnauthorized: false
  }
});

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

// Password setup endpoint
app.post('/api/auth/setup-password', async (req, res) => {
  const { token, password } = req.body;
  if (!token || !password) {
    return res.status(400).json({ message: 'Token and password are required' });
  }

  try {
    const user = await User.findOne({ passwordResetToken: token, passwordResetExpires: { $gt: new Date() } });
    if (!user) {
      return res.status(400).json({ message: 'Invalid or expired token' });
    }
    // Hash new password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);
    user.password = hashedPassword;
    user.status = 'active';
    user.passwordResetToken = undefined;
    user.passwordResetExpires = undefined;
    await user.save();
    res.status(200).json({ message: 'Password set successfully. You can now log in.' });
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
    
    // Check if user is active and has a password
    if (user.status !== 'active' || !user.password) {
     
      return res.status(403).json({ message: 'Account not active. Please set up your password from the email link.' });
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

    // Generate password setup token and expiry
    const crypto = await import('crypto');
    const token = crypto.randomBytes(32).toString('hex');
    const expiry = new Date(Date.now() + 60 * 60 * 1000); // 1 hour

    // Create new staff user with pending status and no password
    const user = new User({
      email: req.body.email,
      password: null,
      name: req.body.name,
      role: 'staff',
      status: 'pending',
      passwordResetToken: token,
      passwordResetExpires: expiry
    });

    // Save user to database
    const savedUser = await user.save();

    // Send password setup email to user with link containing token
    try {
      const setupLink = `${FRONTEND_URL}/setup-password?token=${token}`;
      await transporter.sendMail({
        from: EMAIL_FROM,
        to: req.body.email,
        subject: 'Welcome to Chiremba – Set Up Your Account',
        html: `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Set Up Your Chiremba Account</title>
</head>
<body style="background: #f9f9f9; font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background: #f9f9f9; padding: 40px 0;">
    <tr>
      <td align="center">
        <table width="100%" style="max-width: 480px; background: #fff; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.06); padding: 32px 28px;">
          <tr>
            <td align="center" style="padding-bottom: 18px;">
              <img src="https://imgur.com/pG9BKgz.png" width="64" height="64" alt="Chiremba Logo" style="border-radius: 8px;">
            </td>
          </tr>
          <tr>
            <td align="center" style="padding-bottom: 18px;">
              <h1 style="margin: 0; font-size: 1.8rem; color: #e05a47;">Welcome to Chiremba!</h1>
            </td>
          </tr>
          <tr>
            <td style="padding-bottom: 18px; color: #444; font-size: 1.05rem; line-height: 1.6;">
              <p style="margin: 0 0 12px 0;">You have been invited to join the Chiremba platform. To activate your account, please set your password by clicking the button below. This link will expire in 1 hour for your security.</p>
            </td>
          </tr>
          <tr>
            <td align="center" style="padding-bottom: 24px;">
              <a href="${setupLink}" style="display: inline-block; background: #e05a47; color: #fff; font-weight: 600; padding: 14px 32px; border-radius: 6px; text-decoration: none; font-size: 1.15rem; letter-spacing: 0.03em; box-shadow: 0 2px 8px rgba(224,90,71,0.12);">Set Up Password</a>
            </td>
          </tr>
          <tr>
            <td style="color: #888; font-size: 0.95rem; padding-bottom: 12px;">
              <p style="margin: 0;">If you did not expect this email, you can safely ignore it.</p>
            </td>
          </tr>
          <tr>
            <td align="center" style="padding-top: 12px; color: #bbb; font-size: 0.85rem;">
              &copy; ${new Date().getFullYear()} Chiremba. All rights reserved.
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
        `,
      });
    } catch (err) {
      console.error('Error sending setup email:', err);
    }

    res.status(201).json({
      user: {
        id: savedUser._id,
        email: savedUser.email,
        name: savedUser.name,
        role: savedUser.role,
        status: savedUser.status
      },
      message: 'User created. Password setup email sent.'
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
      role: 'admin',
      status: 'active'
    });

    await adminUser.save();

    // Create staff user
    const staffPassword = await bcrypt.hash('staff123', salt);
    const staffUser = new User({
      email: 'staff@chiremba.com',
      password: staffPassword,
      name: 'Staff User',
      role: 'staff',
      status: 'active'
    });

    await staffUser.save();

    res.status(201).json({ message: 'Admin and staff users initialized successfully' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Serve API keys and config to frontend (non-secret, or ensure proper auth for secret)
app.get('/api/config', (req, res) => {
  res.json({
    OPENAI_API_KEY: process.env.VITE_OPENAI_API_KEY || '',
    GOOGLE_API_KEY: process.env.VITE_GOOGLE_API_KEY || '',
    GOOGLE_CLOUD_TTS_API_KEY: process.env.VITE_GOOGLE_CLOUD_TTS_API_KEY || '',
    ELEVEN_LABS_API_KEY: process.env.VITE_ELEVEN_LABS_API_KEY || '',
    FASTAPI_URL: process.env.VITE_FASTAPI_URL || '',
    EXPRESS_API_URL: process.env.VITE_EXPRESS_API_URL || ''
  });
});

// Serve static files from the frontend build
app.use(express.static(path.join(__dirname, 'frontend/dist')));

// Handle SPA client-side routing - return index.html for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend/dist', 'index.html'));
});

// --- AI API Proxy Endpoints ---
import OpenAI from 'openai';
import fetch from 'node-fetch';

// OpenAI Chat/Completion
app.post('/api/ai/openai/chat', async (req, res) => {
  try {
    const { messages, prompt } = req.body;
    const openai = new OpenAI({ apiKey: process.env.VITE_OPENAI_API_KEY });
    let response;
    if (messages) {
      response = await openai.chat.completions.create({ model: 'gpt-4', messages });
    } else if (prompt) {
      response = await openai.chat.completions.create({ model: 'gpt-4', messages: [{ role: 'user', content: prompt }] });
    } else {
      return res.status(400).json({ error: 'Missing prompt or messages' });
    }
    res.json({ content: response.choices[0].message.content });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// OpenAI TTS
app.post('/api/ai/openai/tts', async (req, res) => {
  try {
    const { text } = req.body;
    const openai = new OpenAI({ apiKey: process.env.VITE_OPENAI_API_KEY });
    const response = await openai.audio.speech.create({ model: 'tts-1', voice: 'nova', input: text });
    const arrayBuffer = await response.arrayBuffer();
    res.set('Content-Type', 'audio/mpeg');
    res.send(Buffer.from(arrayBuffer));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Google Cloud TTS
app.post('/api/ai/google/tts', async (req, res) => {
  try {
    const { text } = req.body;
    const endpoint = `https://texttospeech.googleapis.com/v1/text:synthesize?key=${process.env.VITE_GOOGLE_CLOUD_TTS_API_KEY}`;
    const requestBody = {
      input: { text },
      voice: { languageCode: 'en-US', name: 'en-US-Wavenet-D' },
      audioConfig: { audioEncoding: 'MP3' },
    };
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    });
    const data = await response.json();
    if (!data.audioContent) return res.status(500).json({ error: 'No audio content' });
    const audioBuffer = Buffer.from(data.audioContent, 'base64');
    res.set('Content-Type', 'audio/mp3');
    res.send(audioBuffer);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ElevenLabs TTS
app.post('/api/ai/elevenlabs/tts', async (req, res) => {
  try {
    const { text } = req.body;
    const voiceId = 'tnSpp4vdxKPjI9w0GnoV';
    const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'xi-api-key': process.env.VITE_ELEVEN_LABS_API_KEY,
      },
      body: JSON.stringify({
        text,
        model_id: 'eleven_multilingual_v2',
        voice_settings: { stability: 0.5, similarity_boost: 0.75, style: 0.5, use_speaker_boost: true },
      }),
    });
    if (!response.ok) {
      const errorData = await response.json();
      return res.status(500).json({ error: errorData });
    }
    const audioBuffer = await response.buffer();
    res.set('Content-Type', 'audio/mpeg');
    res.send(audioBuffer);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Google Generative AI (Gemini) Proxy
import { GoogleGenerativeAI } from '@google/generative-ai';
app.post('/api/ai/googleai/chat', async (req, res) => {
  try {
    const { prompt } = req.body;
    const genAI = new GoogleGenerativeAI(process.env.VITE_GOOGLE_API_KEY);
    const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });
    const result = await model.generateContent([prompt]);
    res.json({ content: result.response.text() });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// (Google Generative AI endpoint placeholder: requires extra setup)
// You may add Gemini/Gemini-Proxy logic here if needed

// --- RESET USER ACCOUNT (ADMIN ONLY) ---
app.post('/api/users/:userId/reset-account', authenticateToken, isAdmin, async (req, res) => {
  try {
    const userId = req.params.userId;
    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }
    // Set password to null and status to pending
    user.password = null;
    user.status = 'pending';
    // Generate password reset token
    const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: '1h' });
    user.passwordResetToken = token;
    user.passwordResetExpires = Date.now() + 60 * 60 * 1000; // 1 hour
    await user.save();

    // Send password setup email
    const setupLink = `${FRONTEND_URL}/setup-password?token=${token}`;
    try {
      await transporter.sendMail({
        from: EMAIL_FROM,
        to: user.email,
        subject: 'Reset Your Chiremba Account',
        html: `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reset Your Chiremba Account</title>
</head>
<body style="background: #f9f9f9; font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background: #f9f9f9; padding: 40px 0;">
    <tr>
      <td align="center">
        <table width="100%" style="max-width: 480px; background: #fff; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.06); padding: 32px 28px;">
          <tr>
            <td align="center" style="padding-bottom: 18px;">
              <img src="https://imgur.com/pG9BKgz.png" width="64" height="64" alt="Chiremba Logo" style="border-radius: 8px;">
            </td>
          </tr>
          <tr>
            <td align="center" style="padding-bottom: 18px;">
              <h1 style="margin: 0; font-size: 1.8rem; color: #e05a47;">Reset Your Chiremba Account</h1>
            </td>
          </tr>
          <tr>
            <td style="padding-bottom: 18px; color: #444; font-size: 1.05rem; line-height: 1.6;">
              <p style="margin: 0 0 12px 0;">Your account has been reset by an administrator. To reactivate your account, please set a new password by clicking the button below. This link will expire in 1 hour for your security.</p>
            </td>
          </tr>
          <tr>
            <td align="center" style="padding-bottom: 24px;">
              <a href="${setupLink}" style="display: inline-block; background: #e05a47; color: #fff; font-weight: 600; padding: 14px 32px; border-radius: 6px; text-decoration: none; font-size: 1.15rem; letter-spacing: 0.03em; box-shadow: 0 2px 8px rgba(224,90,71,0.12);">Set Up Password</a>
            </td>
          </tr>
          <tr>
            <td style="color: #888; font-size: 0.95rem; padding-bottom: 12px;">
              <p style="margin: 0;">If you did not expect this email, you can safely ignore it.</p>
            </td>
          </tr>
          <tr>
            <td align="center" style="padding-top: 12px; color: #bbb; font-size: 0.85rem;">
              &copy; ${new Date().getFullYear()} Chiremba. All rights reserved.
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
        `,
      });
    } catch (err) {
      console.error('Error sending reset email:', err);
    }

    res.json({ success: true, message: 'User account reset and setup email sent.' });
  } catch (error) {
    res.status(500).json({ success: false, message: error.message });
  }
});

// Start server
const PORT = process.env.PORT || 5000;
const HOST = process.env.EXPRESS_HOST || '0.0.0.0';

app.listen(PORT, HOST, () => {
  console.log(`Server is running on port ${PORT} and accessible on the network at http://${HOST}:${PORT}`);
});
