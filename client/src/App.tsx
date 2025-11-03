import React, { useState, useRef } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

type InferResult = {
  filename: string;
  predicted_class?: string;
  confidence?: number;
  class_names?: string[];
  error?: string;
};

function App() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [inferResults, setInferResults] = useState<InferResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [trainLoading, setTrainLoading] = useState(false);
  const [trainMessage, setTrainMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInferResults(null);
    setError(null);
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setSelectedFiles(files);
      setPreviewUrls(files.map(file => URL.createObjectURL(file)));
    }
  };

  const handleUpload = async () => {
    if (!selectedFiles.length) {
      setError("Please select images first.");
      return;
    }
    setLoading(true);
    setInferResults(null);
    setError(null);
    try {
      const formData = new FormData();
      selectedFiles.forEach(file => formData.append("files", file));
      const response = await fetch(`${API_URL}/infer/`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error((await response.json()).detail || "Inference failed");
      }
      const data = await response.json();
      setInferResults(data);
    } catch (err: any) {
      setError(err.message || "Inference failed");
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async () => {
    setTrainLoading(true);
    setTrainMessage(null);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/train/`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error((await response.json()).detail || "Training failed");
      }
      const data = await response.json();
      setTrainMessage(data.message || "Model retrained!");
    } catch (err: any) {
      setError(err.message || "Training failed");
    } finally {
      setTrainLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFiles([]);
    setPreviewUrls([]);
    setInferResults(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const filterByClass = (className: string) =>
    inferResults?.filter(
      (res) => res.predicted_class === className && !res.error
    ) || [];

  return (
    <div className="cv-container">
      <header className="cv-header">
        <h1>All-Maintainability-CV</h1>
        <p style={{ marginBottom: "1rem", marginTop: "2rem" }}>
          Upload one or more images to classify their condition using the AI prediction model.<br />
          You can also retrain the model with new data.
        </p>
      </header>

      <section className="cv-upload-section">
        <input
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileChange}
          ref={fileInputRef}
          style={{ display: "none" }}
        />
        <button
          className="cv-btn cv-btn-upload"
          onClick={() => fileInputRef.current?.click()}
        >
          {selectedFiles.length ? "Change Images" : "Select Images"}
        </button>
        {selectedFiles.length > 0 && (
          <button className="cv-btn cv-btn-reset" onClick={handleReset}>
            Reset
          </button>
        )}
      </section>

      {previewUrls.length > 0 && (
        <section className="cv-preview-section" style={{display: "flex", flexWrap: "wrap", gap: "1rem"}}>
          {previewUrls.map((url, idx) => (
            <img
              key={idx}
              src={url}
              alt={`Preview ${idx}`}
              className="cv-preview-img"
              style={{ maxWidth: 120, maxHeight: 120, borderRadius: 12 }}
            />
          ))}
        </section>
      )}

      <section className="cv-action-section">
        <button
          className="cv-btn cv-btn-infer"
          onClick={handleUpload}
          disabled={!selectedFiles.length || loading}
        >
          {loading ? "Analyzing..." : "Run Inference"}
        </button>
        <button
          className="cv-btn cv-btn-train"
          onClick={handleRetrain}
          disabled={trainLoading}
        >
          {trainLoading ? "Retraining..." : "Retrain Model"}
        </button>
      </section>

      {error && (
        <div className="cv-error">
          <span>⚠️ {error}</span>
        </div>
      )}

      {trainMessage && (
        <div className="cv-success">
          <span>{trainMessage}</span>
        </div>
      )}

      {inferResults && (
        <section className="cv-result-section">
          <h2>Prediction Results</h2>
          <div className="cv-result-card" style={{overflowX: "auto"}}>
            <table style={{width: "100%", color: "#fff", borderCollapse: "collapse"}}>
              <thead>
                <tr>
                  <th style={{textAlign: "left", padding: "0.3em 0.5em"}}>Image</th>
                  <th style={{textAlign: "left", padding: "0.3em 0.5em"}}>Class</th>
                  <th style={{textAlign: "left", padding: "0.3em 0.5em"}}>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {inferResults.map((res, idx) => (
                  <tr key={idx}>
                    <td style={{padding: "0.3em 0.5em"}}>{res.filename}</td>
                    <td style={{padding: "0.3em 0.5em"}}>{res.predicted_class || "-"}</td>
                    <td style={{padding: "0.3em 0.5em"}}>
                      {res.confidence !== undefined
                        ? (res.confidence * 100).toFixed(2) + "%"
                        : "-"}
                    </td>
                    <td style={{padding: "0.3em 0.5em", color: "red"}}>{res.error || ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {/* Example: Filtered lists */}
          <div style={{marginTop: "1.5em"}}>
            <h3 style={{marginBottom: "0.3em"}}>Images classified as <span style={{color: "#ef4444"}}>agrietadas</span>:</h3>
            <ul>
              {filterByClass("paredes_exteriores_agrietadas").map((res, idx) => (
                <li key={idx}>{res.filename}</li>
              ))}
            </ul>
            <h3 style={{marginBottom: "0.3em"}}>Images classified as <span style={{color: "#ef4444"}}>damaged</span>:</h3>
            <ul>
              {filterByClass("ceilingDamaged").map((res, idx) => (
                <li key={idx}>{res.filename}</li>
              ))}
            </ul>
          </div>
        </section>
      )}


      <footer className="cv-footer">
        <p>
          <a href="https://github.com/labyrinthgarden/All-Maintainability-CV" target="_blank" rel="noopener noreferrer">
            GitHub
          </a>{" "}
          | Powered by FastAPI + React + TensorFlow
        </p>
      </footer>
    </div>
  );
}

export default App;
