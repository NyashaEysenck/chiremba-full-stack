import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

export interface ReportData {
  date: string;
  time: string;
  condition: string;
  confidence: number;
  description: string;
  urgency: string;
  modelUsed: string;
  imageData: string;
  alternatives?: Array<{ condition: string; confidence: number; description: string; urgency: string }>;
  aiExplanation?: string;
}

export const downloadReport = async (data: ReportData) => {
  const element = document.createElement('div');
  element.className = 'report-container';
  element.innerHTML = `
    <style>
      .report-container {
        font-family: 'Arial', sans-serif;
        max-width: 750px;
        margin: 0 auto;
        padding: 20px;
        color: #334155;
        font-size: 14px;
      }
      .header {
        text-align: center;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 2px solid #0ea5e9;
      }
      .header h1 {
        color: #0ea5e9;
        margin: 0;
        font-size: 20px;
      }
      .header p {
        color: #64748b;
        margin: 5px 0;
        font-size: 12px;
      }
      .section {
        margin-bottom: 20px;
      }
      .section-title {
        color: #0ea5e9;
        font-size: 16px;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #e2e8f0;
      }
      .result-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 15px;
      }
      .result-box h3 {
        font-size: 14px;
        margin: 0 0 8px 0;
      }
      .result-box p {
        margin: 6px 0;
        line-height: 1.4;
      }
      .confidence-bar {
        height: 6px;
        background-color: #e2e8f0;
        border-radius: 3px;
        margin: 8px 0;
        overflow: hidden;
      }
      .confidence-fill {
        height: 100%;
        background-color: #0ea5e9;
        border-radius: 3px;
      }
      .image-container {
        text-align: center;
        margin: 15px 0;
      }
      .image-container img {
        max-width: 100%;
        max-height: 250px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      }
      .ai-analysis {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 12px;
        margin-top: 15px;
      }
      .ai-analysis h3 {
        color: #0284c7;
        margin: 0 0 8px 0;
        font-size: 14px;
      }
      .ai-section {
        margin-bottom: 12px;
      }
      .ai-section h4 {
        color: #0284c7;
        margin: 0 0 4px 0;
        font-size: 13px;
      }
      .ai-section p {
        margin: 4px 0;
        line-height: 1.4;
      }
      .footer {
        margin-top: 20px;
        padding-top: 15px;
        border-top: 1px solid #e2e8f0;
        font-size: 11px;
        color: #64748b;
      }
      p {
        margin: 0 0 8px 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
      }
      .page-break {
        break-after: page;
        page-break-after: always;
      }
      @page {
        margin: 20mm;
      }
    </style>

    <div class="header">
      <h1>Chiremba Image Diagnosis Report</h1>
      <p>Generated on ${data.date} at ${data.time}</p>
    </div>

    <div class="section">
      <h2 class="section-title">Analysis Results</h2>
      <div class="result-box">
        <h3>Detected Condition: ${data.condition}</h3>
        <p>${data.description}</p>
        <div class="confidence-bar">
          <div class="confidence-fill" style="width: ${data.confidence}%"></div>
        </div>
        <p>Confidence Level: ${data.confidence}%</p>
        <p>Model Used: ${data.modelUsed}</p>
      </div>
    </div>

    <div class="image-container">
      <img src="${data.imageData}" alt="Analyzed Image" />
    </div>

    <div class="ai-analysis">
      <h3>AI Detailed Analysis</h3>
      ${data.aiExplanation ? data.aiExplanation.split('\n').map(paragraph => {
        const cleanText = paragraph.replace(/\*\*/g, '');
        if (cleanText.match(/^\d+\.\s+[A-Za-z\s]+:/)) {
          return `<div class="ai-section">
            <h4>${cleanText.replace(/^\d+\.\s+/, '')}</h4>
          </div>`;
        } else if (cleanText.trim()) {
          return `<p>${cleanText}</p>`;
        }
        return '';
      }).join('') : '<p>AI analysis not available</p>'}
    </div>

    ${data.alternatives && data.alternatives.length > 0 ? `
      <div class="section">
        <h2 class="section-title">Alternative Diagnoses</h2>
        <div class="result-box">
          ${data.alternatives.map(alt => `
            <div style="margin-bottom: 10px;">
              <p><strong>${alt.condition}</strong> (${alt.confidence}% confidence) - ${alt.description}</p>
            </div>
          `).join('')}
        </div>
      </div>
    ` : ''}

    <div class="footer">
      <p>This report was generated using AI-assisted analysis. Please consult with a qualified healthcare professional for proper evaluation and treatment.</p>
    </div>
  `;

  document.body.appendChild(element);
  await html2canvas(element, { scale: 2 }).then(canvas => {
    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
    pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
    pdf.save(`Chiremba_Image_Diagnosis_Report_${data.date.replace(/\//g, '-')}_${data.time.replace(/:/g, '-')}.pdf`);
  });
  document.body.removeChild(element);
};

export const printReport = (data: ReportData) => {
  const element = document.createElement('div');
  element.className = 'report-container';
  element.innerHTML = `
    <style>
      .report-container {
        font-family: 'Arial', sans-serif;
        max-width: 750px;
        margin: 0 auto;
        padding: 20px;
        color: #334155;
        font-size: 14px;
      }
      .header {
        text-align: center;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 2px solid #0ea5e9;
      }
      .header h1 {
        color: #0ea5e9;
        margin: 0;
        font-size: 20px;
      }
      .header p {
        color: #64748b;
        margin: 5px 0;
        font-size: 12px;
      }
      .section {
        margin-bottom: 20px;
      }
      .section-title {
        color: #0ea5e9;
        font-size: 16px;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #e2e8f0;
      }
      .result-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 15px;
      }
      .result-box h3 {
        font-size: 14px;
        margin: 0 0 8px 0;
      }
      .result-box p {
        margin: 6px 0;
        line-height: 1.4;
      }
      .confidence-bar {
        height: 6px;
        background-color: #e2e8f0;
        border-radius: 3px;
        margin: 8px 0;
        overflow: hidden;
      }
      .confidence-fill {
        height: 100%;
        background-color: #0ea5e9;
        border-radius: 3px;
      }
      .image-container {
        text-align: center;
        margin: 15px 0;
      }
      .image-container img {
        max-width: 100%;
        max-height: 250px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      }
      .ai-analysis {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 12px;
        margin-top: 15px;
      }
      .ai-analysis h3 {
        color: #0284c7;
        margin: 0 0 8px 0;
        font-size: 14px;
      }
      .ai-section {
        margin-bottom: 12px;
      }
      .ai-section h4 {
        color: #0284c7;
        margin: 0 0 4px 0;
        font-size: 13px;
      }
      .ai-section p {
        margin: 4px 0;
        line-height: 1.4;
      }
      .footer {
        margin-top: 20px;
        padding-top: 15px;
        border-top: 1px solid #e2e8f0;
        font-size: 11px;
        color: #64748b;
      }
      p {
        margin: 0 0 8px 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
      }
      .page-break {
        break-after: page;
        page-break-after: always;
      }
      @page {
        margin: 20mm;
      }
    </style>

    <div class="header">
      <h1>Chiremba Image Diagnosis Report</h1>
      <p>Generated on ${data.date} at ${data.time}</p>
    </div>

    <div class="section">
      <h2 class="section-title">Analysis Results</h2>
      <div class="result-box">
        <h3>Detected Condition: ${data.condition}</h3>
        <p>${data.description}</p>
        <div class="confidence-bar">
          <div class="confidence-fill" style="width: ${data.confidence}%"></div>
        </div>
        <p>Confidence Level: ${data.confidence}%</p>
        <p>Model Used: ${data.modelUsed}</p>
      </div>
    </div>

    <div class="image-container">
      <img src="${data.imageData}" alt="Analyzed Image" />
    </div>

    <div class="ai-analysis">
      <h3>AI Detailed Analysis</h3>
      ${data.aiExplanation ? data.aiExplanation.split('\n').map(paragraph => {
        const cleanText = paragraph.replace(/\*\*/g, '');
        if (cleanText.match(/^\d+\.\s+[A-Za-z\s]+:/)) {
          return `<div class="ai-section">
            <h4>${cleanText.replace(/^\d+\.\s+/, '')}</h4>
          </div>`;
        } else if (cleanText.trim()) {
          return `<p>${cleanText}</p>`;
        }
        return '';
      }).join('') : '<p>AI analysis not available</p>'}
    </div>

    ${data.alternatives && data.alternatives.length > 0 ? `
      <div class="section">
        <h2 class="section-title">Alternative Diagnoses</h2>
        <div class="result-box">
          ${data.alternatives.map(alt => `
            <div style="margin-bottom: 10px;">
              <p><strong>${alt.condition}</strong> (${alt.confidence}% confidence) - ${alt.description}</p>
            </div>
          `).join('')}
        </div>
      </div>
    ` : ''}

    <div class="footer">
      <p>This report was generated using AI-assisted analysis. Please consult with a qualified healthcare professional for proper evaluation and treatment.</p>
    </div>
  `;

  const printWindow = window.open('', '_blank');
  if (printWindow) {
    printWindow.document.write(element.innerHTML);
    printWindow.document.close();
    printWindow.onload = function() {
      printWindow.print();
    };
  }
};