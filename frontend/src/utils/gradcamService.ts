// Service to request Grad-CAM from backend for lung cancer
export async function requestLungCancerGradCAM(file: File): Promise<string> {
  const formData = new FormData();
  formData.append('file', file);
  const apiUrl = import.meta.env.VITE_FASTAPI_URL || 'http://localhost:8000';
  const res = await fetch(`${apiUrl}/lungcancer_gradcam`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    throw new Error('Failed to get Grad-CAM visualization');
  }
  const data = await res.json();
  return data.gradcam_image as string;
}
