# Chiremba Frontend and Deployment

This directory contains the frontend application and deployment configurations for the Chiremba medical diagnosis system.

## Frontend Application

A modern React-based web application that provides an intuitive interface for medical image analysis.
see img diagnosis backend : https://github.com/NyashaEysenck/chiremba-ai-backend-image-diagnosis

### Features

- User-friendly interface for medical image upload
- Support for multiple diagnosis types:
  - Skin disease analysis
  - Brain tumor detection
  - Lung cancer screening
  - Pneumonia diagnosis
- Real-time analysis results
- Detailed medical reports generation
- Responsive design for all devices

### Technology Stack

- React with TypeScript
- Material-UI for components
- Axios for API communication


### Setup

1. Install dependencies:
```bash
npm install
```

2. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Update API endpoints and configuration

3. Start development server:
```bash
node index
```

4. Build for production:
```bash
cd frontend
npm run build
```

## Environment Variables

- `REACT_APP_SKIN_SERVICE_URL`: Skin service API endpoint
- `REACT_APP_IMAGE_DIAGNOSIS_URL`: AI backend service endpoint
- `REACT_APP_API_KEY`: API authentication key (if required)


## Directory Structure

- `/frontend`: React application source code
- 
## Previews

### **Home Page**
<img src="app_previews/home.png" width="200"/>

---

### **Symptom Checker**
<div style="display: flex; gap: 10px;">
  <img src="app_previews/symptom1.png" width="200"/>
  <img src="app_previews/symptom2.png" width="200"/>
  <img src="app_previews/symptom3.png" width="200"/>
</div>

---

### **Virtual Chat**
<div style="display: flex; gap: 10px;">
  <img src="app_previews/chat1.png" width="200"/>
  <img src="app_previews/chat2.png" width="200"/>
</div>

---

### **Login**
<img src="app_previews/login.png" width="200"/>

---

### **Image Diagnosis**
<div style="display: flex; gap: 10px;">
  <img src="app_previews/diagnosis1.png" width="200"/>
  <img src="app_previews/diagnosis2.png" width="200"/>
  <img src="app_previews/diagnosis3.png" width="200"/>
</div>

## Contributing

Please follow the established coding style and commit message conventions when contributing to this project.
