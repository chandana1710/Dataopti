import { useState, useRef } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Login from "./Login.jsx";
import Register from "./Register.jsx";
import "./App.css";
import "./Redesign.css";

function Dashboard({ user, onLogout }) {
  const [file, setFile] = useState(null);
  const [sourceFiles, setSourceFiles] = useState([]);
  const [page, setPage] = useState("home");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [downloadLoading, setDownloadLoading] = useState(false);
  const [error, setError] = useState("");
  const [historyNotice, setHistoryNotice] = useState("");

  const historyKey = `dataopti_history_${user?.email || user?.name || "guest"}`;
  const [history, setHistory] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(historyKey) || "[]");
    } catch {
      return [];
    }
  });

  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);

  const outlierCount = Number.isFinite(result?.outliers)
    ? result.outliers
    : null;

  const processFiles = async (selectedFiles, isFolder = false) => {
    if (!selectedFiles || selectedFiles.length === 0) {
      return;
    }

    const filesArray = Array.from(selectedFiles);

    if (!isFolder && filesArray.length > 2) {
      setError("You can upload up to 2 files per analysis.");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    setSourceFiles(filesArray);

    /*
     * Do not reject large datasets on the frontend with
     * an unnecessarily small limit.
     *
     * Backend controls the actual HTTP limit.
     */
    const totalSize = filesArray.reduce(
      (acc, f) => acc + f.size,
      0
    );

    const firstFile = filesArray[0];

    const isMultiple = filesArray.length > 1;
    const folderName = isFolder
      ? firstFile.webkitRelativePath?.split("/")[0] || "Selected folder"
      : "";
    const displayName = isFolder
      ? folderName
      : isMultiple
        ? filesArray.length + " files"
        : firstFile.name;

    setFile({
      name: displayName,
      size: totalSize,
      isFolder,
      count: filesArray.length,
    });

    setPage("dashboard");
    setLoading(true);
    setError("");
    setHistoryNotice("");
    setResult(null);

    const formData = new FormData();

    filesArray.forEach((selectedFile) => {
      formData.append("files", selectedFile);
    });

    try {
      const response = await fetch(
        "http://127.0.0.1:5000/api/upload",
        {
          method: "POST",
          body: formData,
        }
      ).catch(() => {
        throw new Error(
          "Cannot connect to server. Ensure Flask backend is running at http://127.0.0.1:5000"
        );
      });

      const contentType =
        response.headers.get("content-type") || "";

      if (response.status === 413) {
        throw new Error(
          "Dataset is larger than the server upload limit. Increase MAX_GB in backend/app.py or upload the dataset in smaller batches."
        );
      }

      if (
        contentType.includes("application/json")
      ) {
        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data.message ||
              "Upload processing failed."
          );
        }

        setResult(data);
        const historyEntry = {
          id: Date.now(),
          name: displayName,
          count: filesArray.length,
          size: totalSize,
          createdAt: new Date().toISOString(),
          rows: data.rows ?? 0,
          columns: data.columns ?? 0,
          missingValues: data.missing_values ?? 0,
          result: data,
        };
        setHistory((currentHistory) => {
          const nextHistory = [historyEntry, ...currentHistory];
          try {
            localStorage.setItem(historyKey, JSON.stringify(nextHistory));
          } catch (storageError) {
            console.warn("Could not save analysis history:", storageError);
          }
          return nextHistory;
        });
      } else {
        const rawText =
          await response.text();

        console.error(
          "Non-JSON response:",
          rawText
        );

        throw new Error(
          `Server returned HTTP ${response.status}. Check Flask backend console.`
        );
      }
    } catch (err) {
      console.error(err);
      setError(
        err?.message ||
          "Upload processing failed."
      );
    } finally {
      setLoading(false);
    }
  };

  const openHistoryEntry = (entry) => {
    setFile({
      name: entry.name,
      size: entry.size ?? 0,
      isFolder: false,
      count: entry.count ?? 1,
    });
    setSourceFiles([]);
    setResult(entry.result || {
      rows: entry.rows ?? 0,
      columns: entry.columns ?? 0,
      missing_values: entry.missingValues ?? 0,
    });
    setHistoryNotice(entry.result ? "" : "This older history item only has summary metrics. Upload the original files to see the full analysis.");
    setError("");
    setPage("dashboard");
  };

  const clearHistory = () => {
    setHistory([]);
    try {
      localStorage.removeItem(historyKey);
    } catch (storageError) {
      console.warn("Could not clear analysis history:", storageError);
    }
  };

  const downloadPreparedData = async () => {
    if (!sourceFiles.length) return;

    setDownloadLoading(true);
    setError("");

    try {
      const formData = new FormData();
      sourceFiles.forEach((sourceFile) => formData.append("files", sourceFile));
      formData.append("model_pipeline", result?.model_results?.best_pipeline || "Pipeline B");

      const response = await fetch("http://127.0.0.1:5000/api/download-cleaned", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.message || "Could not prepare model-ready data.");
      }

      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      const disposition = response.headers.get("content-disposition") || "";
      const filenameMatch = disposition.match(/filename="?([^";]+)"?/i);
      link.download = filenameMatch?.[1] || (blob.type.includes("zip") ? "dataopti_prepared_exports.zip" : "dataopti_model_ready_data.csv");
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
    } catch (downloadError) {
      setError(downloadError?.message || "Could not download model-ready data.");
    } finally {
      setDownloadLoading(false);
    }
  };

  return (
    <div className="app">

      {/* FILE INPUT */}
      <input
        type="file"
        ref={fileInputRef}
        multiple
        onClick={(e) => { e.currentTarget.value = ""; }}
        accept="
          .csv,
          .xlsx,
          .xls,
          .json,
          .parquet,
          .txt,
          .py,
          .md,
          .log,
          .pdf,
          .ipynb,
          .zip,
          .png,
          .jpg,
          .jpeg,
          .webp,
          .gif,
          .bmp,
          .tif,
          .tiff
        "
        onChange={(e) =>
          processFiles(e.target.files)
        }
        style={{ display: "none" }}
      />
      <input
        type="file"
        ref={folderInputRef}
        multiple
        webkitdirectory=""
        directory=""
        onClick={(e) => { e.currentTarget.value = ""; }}
        onChange={(e) => processFiles(e.target.files, true)}
        style={{ display: "none" }}
      />

      {/* NAVBAR */}
      <nav className="navbar">

        <div className="logo">
          <span className="logo-icon">
            ⚡
          </span>
          DataOpti <span>AI</span>
        </div>

        <div className="nav-links">
          <button
            onClick={() =>
              setPage("home")
            }
          >
            Home
          </button>

          <button
            onClick={() =>
              setPage("dashboard")
            }
          >
            Analyze
          </button>

          <button onClick={() => setPage("history")}>
            History
          </button>

          <button onClick={() => setPage("about")}>
            About
          </button>
        </div>

        <div className="account-menu">
          <span>Welcome, {user?.name || user?.email}</span>
          <button type="button" onClick={onLogout}>
            Sign out
          </button>
        </div>

      </nav>

      {/* HOME */}
      {page === "home" && (
        <main className="home-page">

          <section className="hero home-hero">

            <div className="hero-text">

              <div className="badge">
                ✨ AI-Powered Data Optimization
              </div>

              <h1>
                Better data.
                <span>Better decisions.</span>
              </h1>

              <p>
                Get a clear picture of your dataset, fix what matters, and find a stronger machine-learning pipeline—all in one workspace.
              </p>

              <div className="hero-buttons">

                <button
                  className="primary-btn"
                  onClick={() =>
                    setPage("dashboard")
                  }
                >
                  Start Analyzing →
                </button>

                <button className="secondary-btn" onClick={() => document.getElementById("features")?.scrollIntoView({ behavior: "smooth" })}>
                  Explore Features
                </button>

              </div>

              <div className="supported">

                <span>✓ CSV</span>
                <span>✓ Excel</span>
                <span>✓ JSON</span>
                <span>✓ Parquet</span>
                <span>✓ PDF / TXT</span>
                <span>✓ Folders / ZIP</span>
                <span>✓ Notebooks</span>
                <span>✓ Images</span>

              </div>

            </div>

            <div className="hero-card hero-preview">

              <div className="card-header">

                <div>
                  <small>
                    DATA QUALITY
                  </small>

                  <h3>
                    Dataset Analysis
                  </h3>
                </div>

                <div className="status">
                  ● Sample
                </div>

              </div>

              <div className="quality-circle">

                <div>
                  <strong>
                    87
                  </strong>

                  <small>
                    /100
                  </small>
                </div>

              </div>

              <div className="quality-label">
                Good Quality
              </div>

              <div className="mini-stats">

                <div>
                  <strong>
                    10K
                  </strong>
                  <small>
                    Rows
                  </small>
                </div>

                <div>
                  <strong>
                    18
                  </strong>
                  <small>
                    Features
                  </small>
                </div>

                <div>
                  <strong>
                    2.4s
                  </strong>
                  <small>
                    Runtime
                  </small>
                </div>

              </div>

            </div>

          </section>

          <section className="features-section" id="features">

            <div className="section-title">

              <span>
                POWERFUL FEATURES
              </span>

              <h2>
                Everything You Need Before ML
              </h2>

              <p>
                One intelligent platform for
                understanding and optimizing
                your datasets.
              </p>

            </div>

            <div className="feature-grid">

              <Feature
                icon="🔍"
                title="Data Profiling"
                text="Automatically understand rows, columns, types, duplicates and data quality."
              />

              <Feature
                icon="🧹"
                title="Data Cleaning"
                text="Detect and handle missing values, outliers and noisy data."
              />

              <Feature
                icon="🎯"
                title="Feature Selection"
                text="Identify the most important features using multiple techniques."
              />

              <Feature
                icon="🤖"
                title="ML Comparison"
                text="Compare machine-learning performance before and after optimization."
              />

              <Feature
                icon="⚙️"
                title="Pipeline Optimization"
                text="Evaluate multiple preprocessing pipelines automatically."
              />

              <Feature
                icon="📊"
                title="Cost Analysis"
                text="Compare accuracy, execution time and feature reduction."
              />

            </div>

          </section>

        </main>
      )}

      {/* DASHBOARD */}
      {page === "dashboard" && (

        <main className="dashboard dashboard-redesign">

          <div className="dashboard-heading dashboard-hero">

            <div className="dashboard-title-block">
              <span className="section-kicker">YOUR DATA WORKSPACE</span>
              <h1>Dataset dashboard</h1>
              <p>Upload a dataset to uncover quality issues and prepare it for machine learning.</p>
            </div>

          </div>

          {!file && !loading && (
            <section className="upload-dropzone upload-empty-state">
              
              <span className="section-kicker">YOUR NEXT ANALYSIS STARTS HERE</span>
              <h2>Bring your data into focus</h2>
              <p>Choose to upload a folder to analyze its contents together.</p>
              <div className="upload-choice-buttons">
                <button className="primary-btn" type="button" onClick={() => fileInputRef.current?.click()}>
                  <span aria-hidden="true">＋</span> Upload files
                </button>
                <button className="secondary-btn" type="button" onClick={() => folderInputRef.current?.click()}>
                  <span aria-hidden="true">▰</span> Upload folders
                </button>
              </div>
              <div className="dropzone-formats">
                <span>CSV</span><span>Excel</span><span>JSON</span><span>Parquet</span><span>PDF</span><span>ZIP</span>
              </div>
            </section>
          )}

          {error && !file && (
            <div className="upload-error" role="alert">{error}</div>
          )}
          {file && (

            <>

              {/* UPLOAD INFORMATION */}
              <div className="file-info active-file-card">

                <div className="file-icon">
                  {file.count > 1 ? "🗂️" : "📄"}
                </div>

                <div>

                  <strong>
                    {file.name}
                  </strong>

                  <p>
                    {(
                      file.size /
                      (1024 * 1024)
                    ).toFixed(2)}{" "}
                    MB{" "}

                    {file.count > 1 && "(" + file.count + " files)"}
                  </p>

                </div>

                <div className="file-status">

                  {loading
                    ? "⏳ Analyzing..."
                    : result
                    ? "✓ Analyzed"
                    : "Uploaded"}

                </div>

              </div>

              {/* ERROR */}
              {error && (

                <div className="panel">

                  <h3>
                    ⚠️ Error
                  </h3>

                  <p>
                    {error}
                  </p>

                </div>

              )}

              {/* LOADING */}
              {loading && (

                <div className="panel">

                  <h2>
                    🔄 Analyzing Dataset...
                  </h2>

                  <p>
                    DataOpti AI is profiling,
                    cleaning, selecting features
                    and evaluating the dataset.
                  </p>

                </div>

              )}

              {result && (

                <>

                  {historyNotice && (
                    <div className="panel" role="status">{historyNotice}</div>
                  )}

                  {/* ==================================
                      1. DATASET DETAILS
                  ================================== */}

                  <div className="stats-grid">

                    <Stat
                      icon="📋"
                      title="Rows"
                      value={
                        result.rows?.toLocaleString() ??
                        "0"
                      }
                      change="Dataset size"
                    />

                    <Stat
                      icon="📊"
                      title="Columns"
                      value={
                        result.columns ?? 0
                      }
                      change="Features"
                    />

                    <Stat
                      icon="⚠️"
                      title="Missing Values"
                      value={
                        result.missing_values?.toLocaleString() ??
                        "0"
                      }
                      change={
                        result.missing_values > 0
                          ? "Needs attention"
                          : "No missing values"
                      }
                    />

                    <Stat
                      icon="🔴"
                      title="Outliers"
                      value={
                        outlierCount === null
                          ? "—"
                          : outlierCount.toLocaleString()
                      }
                      change={
                        outlierCount === null
                          ? "Not provided"
                          : outlierCount > 0
                          ? "Detected"
                          : "No outliers"
                      }
                    />

                  </div>


                  {/* ==================================
                      DATA QUALITY
                  ================================== */}

                  <div className="dashboard-grid">

                    <div className="panel">

                      <div className="panel-heading">

                        <h2>
                          Data Quality
                        </h2>

                        <span className="good">
                          Analyzed
                        </span>

                      </div>

                      <div className="quality-items">

                        <div>
                          <span>
                            Rows
                          </span>

                          <strong>
                            {result.rows?.toLocaleString() ??
                              "0"}
                          </strong>
                        </div>

                        <div>
                          <span>
                            Columns
                          </span>

                          <strong>
                            {result.columns ?? 0}
                          </strong>
                        </div>

                        <div>
                          <span>
                            Missing Values
                          </span>

                          <strong>
                            {result.missing_values?.toLocaleString() ??
                              "0"}
                          </strong>
                        </div>

                        <div>
                          <span>
                            Duplicates
                          </span>

                          <strong>
                            {result.duplicate_rows?.toLocaleString() ??
                              "0"}
                          </strong>
                        </div>

                      </div>

                    </div>


                    {/* AI RECOMMENDATION */}

                    <div className="panel recommendation">

                      <div className="ai-title">
                        ✨ AI RECOMMENDATION
                      </div>

                      <h2>
                        {getRecommendationModelName(result.recommendation?.model)}
                      </h2>

                      <p>
                        {result.recommendation?.reason ||
                          "DataOpti AI analyzed your dataset."}
                      </p>

                      <div className="pipeline">
                        <div>1</div>
                        <span>
                          Data Profiling
                        </span>
                      </div>

                      <div className="pipeline">
                        <div>2</div>
                        <span>
                          Data Cleaning
                        </span>
                      </div>

                      <div className="pipeline">
                        <div>3</div>
                        <span>
                          Feature Selection
                        </span>
                      </div>

                      <div className="pipeline">
                        <div>4</div>
                        <span>
                          Best Model Recommendation
                        </span>
                      </div>

                      {sourceFiles.length > 0 && result.dataset_type && result.dataset_type !== "unknown" && (
                        <div className="recommendation-download-row">
                          <button
                            className="recommendation-download-button"
                            type="button"
                            onClick={downloadPreparedData}
                            disabled={downloadLoading}
                          >
                            {downloadLoading ? "Preparing download…" : result.dataset_type === "tabular" ? "Download model-ready data ↓" : "Download prepared data ↓"}
                          </button>
                          <small>
                            {sourceFiles.length > 1
                              ? "Downloads prepared exports in a ZIP file."
                              : result.dataset_type === "tabular"
                              ? `Downloads data processed for ${getRecommendationModelName(result.recommendation?.model)} (up to ${result.performance?.ml_sample_limit?.toLocaleString() ?? "20,000"} rows), with the target labels retained.`
                              : "Downloads extracted text or file metadata as CSV."}
                          </small>
                        </div>
                      )}

                    </div>

                  </div>


                  {/* ==================================
                      2. BEFORE / AFTER CLEANING
                  ================================== */}

                  <CleaningPanel
                    cleaning={
                      result.cleaning
                    }
                    accuracyComparison={result.model_results?.accuracy_comparison}
                  />


                  {/* ==================================
                      3. FEATURE SELECTION
                  ================================== */}

                  <FeatureSelectionPanel
                    modelResults={
                      result.model_results
                    }
                  />


                  {/* ==================================
                      4. PIPELINE A / B / C
                  ================================== */}

                  <PipelinePanel
                    modelResults={
                      result.model_results
                    }
                  />


                  {/* ==================================
                      5. MODEL RECOMMENDATION
                  ================================== */}

                  {result.model_results && (

                    <div className="panel model-panel">

                      <div className="panel-heading">

                        <div>

                          <h2>
                            Best Model for This Dataset
                          </h2>

                          <p>
                            Automatically selected from
                            the evaluated pipelines.
                          </p>

                        </div>

                      </div>

                      <div className="supported">

                        <span>
                          🏆 Best Pipeline:{" "}
                          <strong>
                            {result.model_results.best_pipeline ||
                              "—"}
                          </strong>
                        </span>

                        <span>
                          🎯 Accuracy:{" "}
                          {formatPercent(
                            result.model_results.best_score
                          )}
                        </span>

                        <span>
                          ⏱ Training:{" "}
                          {result.model_results.training_seconds ??
                            "—"}{" "}
                          seconds
                        </span>

                        <span>
                          🎯 Target:{" "}
                          {result.model_results.target_column ||
                            "—"}
                        </span>

                      </div>

                    </div>

                  )}


                  {/* ==================================
                      6. DATASET COLUMNS
                  ================================== */}

                  <div className="panel model-panel">

                    <div className="panel-heading">

                      <div>

                        <h2>
                          Dataset Columns
                        </h2>

                        <p>
                          Columns detected by DataOpti AI
                        </p>

                      </div>

                    </div>

                    <div className="supported">

                      {(
                        Array.isArray(
                          result.column_names
                        )
                          ? result.column_names
                          : []
                      ).map(
                        (column, index) => (

                          <span
                            key={`${column}-${index}`}
                          >
                            ✓ {column}
                          </span>

                        )
                      )}

                    </div>

                  </div>


                  {/* ==================================
                      7. SKIPPED FILES
                  ================================== */}

                  {result.skipped_files?.length > 0 && (

                    <div className="panel">

                      <h2>
                        ⚠️ Files Requiring Attention
                      </h2>

                      <p>
                        These files did not stop the
                        rest of the dataset analysis.
                      </p>

                      <div className="supported">

                        {result.skipped_files.map(
                          (item, index) => (

                            <span
                              key={index}
                            >
                              ⚠️ {item.filename}:{" "}
                              {item.reason}
                            </span>

                          )
                        )}

                      </div>

                    </div>

                  )}


                  {/* ==================================
                      PERFORMANCE
                  ================================== */}

                  <div className="panel">

                    <div className="panel-heading">

                      <div>

                        <h2>
                          Processing Performance
                        </h2>

                        <p>
                          DataOpti adapts processing
                          for large datasets.
                        </p>

                      </div>

                    </div>

                    <div className="quality-items">

                      <div>
                        <span>
                          Processing Time
                        </span>

                        <strong>
                          {result.performance?.processing_seconds ??
                            "—"}{" "}
                          sec
                        </strong>
                      </div>

                      <div>
                        <span>
                          ML Sample
                        </span>

                        <strong>
                          {result.performance?.ml_sample_limit?.toLocaleString() ??
                            "—"}
                        </strong>
                      </div>

                      <div>
                        <span>
                          Strategy
                        </span>

                        <strong>
                          Chunked / Sampled
                        </strong>
                      </div>

                    </div>

                    {result.performance?.profile_sampled && (
                      <p className="sampled-analysis-note">
                        Large-file quality metrics use a sample of up to {result.performance.profile_sample_rows?.toLocaleString() ?? "—"} rows. Model evaluation uses up to {result.performance.ml_sample_limit?.toLocaleString() ?? "—"} rows for faster analysis; reported row counts may be approximate for sampled formats.
                      </p>
                    )}

                  </div>

                </>

              )}

            </>

          )}

          <div className="dashboard-bottom-actions">
            <button className="secondary-btn" type="button" onClick={() => setPage("home")}>
              ← Back to home
            </button>
          </div>

        </main>

      )}

      {page === "history" && (
        <main className="dashboard dashboard-redesign history-page">
          <div className="dashboard-heading dashboard-hero">
            <div className="dashboard-title-block">
              <span className="section-kicker">YOUR WORKSPACE</span>
              <h1>Analysis history</h1>
              
            </div>
            <div className="dashboard-actions">
              
            </div>
          </div>
          {history.length === 0 ? (
            <section className="upload-dropzone upload-empty-state history-empty">
              <div className="dropzone-icon" aria-hidden="true">◷</div>
              <h2>No analyses yet</h2>
              <p>When an upload finishes successfully, its summary will appear here.</p>
              <button className="primary-btn" type="button" onClick={() => { setFile(null); setResult(null); setError(""); setPage("dashboard"); }}>Analyze a dataset</button>
            </section>
          ) : (
            <section className="history-list" aria-label="Past analyses">
              {history.map((entry) => (
                <article
                  className="history-card"
                  key={entry.id}
                  role="button"
                  tabIndex={0}
                  aria-label={`Open analysis for ${entry.name}`}
                  onClick={() => openHistoryEntry(entry)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      openHistoryEntry(entry);
                    }
                  }}
                >
                  <div className="history-file-icon" aria-hidden="true">📄</div>
                  <div className="history-main">
                    <h2>{entry.name}</h2>
                    <p>{new Date(entry.createdAt).toLocaleString()} · {entry.count} {entry.count === 1 ? "file" : "files"}</p>
                  </div>
                  <div className="history-metrics">
                    <span><strong>{Number(entry.rows).toLocaleString()}</strong> rows</span>
                    <span><strong>{entry.columns}</strong> columns</span>
                    <span><strong>{Number(entry.missingValues).toLocaleString()}</strong> missing</span>
                  </div>
                </article>
              ))}
              <button className="history-clear-button history-clear-button-end" type="button" onClick={clearHistory}>
                Clear all
              </button>
            </section>
          )}
        </main>
      )}

      {page === "about" && (
        <main className="about-page">
          <section className="about-hero">
            <span className="section-kicker">ABOUT DATAOPTI AI</span>
            <h1>Understand your data before you build with it.</h1>
            <p>
              DataOpti AI is a data preparation and analysis workspace. Upload files or a folder to get a readable overview of data quality, spot issues that may affect your work, and—when the input is tabular—review cleaning and machine-learning pipeline results.
            </p>
            <button className="primary-btn" type="button" onClick={() => setPage("dashboard")}>
              Start analyzing →
            </button>
          </section>

          <section className="about-section" aria-labelledby="about-workflow-title">
            <div className="about-section-heading">
              <span className="section-kicker">HOW IT WORKS</span>
              <h2 id="about-workflow-title">From upload to useful next steps</h2>
              <p>Each analysis brings the important information into one place.</p>
            </div>
            <div className="about-steps">
              <article className="about-card">
                <span className="about-step-number">01</span>
                <h3>Choose your data</h3>
                <p>Upload up to two individual files, or select a folder to send its contents together. ZIP archives are also supported.</p>
              </article>
              <article className="about-card">
                <span className="about-step-number">02</span>
                <h3>Review the data profile</h3>
                <p>See the rows and columns found, missing values, duplicate rows, outliers, and the column names detected in your input.</p>
              </article>
              <article className="about-card">
                <span className="about-step-number">03</span>
                <h3>Inspect preparation results</h3>
                <p>For tabular data, review cleaning summaries and feature selection results when available. The application also evaluates machine-learning pipelines and reports a recommendation when it can.</p>
              </article>
              <article className="about-card">
                <span className="about-step-number">04</span>
                <h3>Compare and decide</h3>
                <p>Use the reported quality metrics, pipeline scores, and processing details to decide what to investigate or try next.</p>
              </article>
            </div>
          </section>

          <section className="about-section about-capabilities" aria-labelledby="about-capabilities-title">
            <div className="about-section-heading">
              <span className="section-kicker">WHAT YOU CAN EXPLORE</span>
              <h2 id="about-capabilities-title">A clearer picture of dataset readiness</h2>
            </div>
            <div className="about-capability-grid">
              <article><span aria-hidden="true">⌕</span><div><h3>Data profiling</h3><p>Understand dataset size, columns, and common quality signals.</p></div></article>
              <article><span aria-hidden="true">✳</span><div><h3>Quality checks</h3><p>Find missing values, duplicates, and outliers that may need attention.</p></div></article>
              <article><span aria-hidden="true">⟲</span><div><h3>Cleaning overview</h3><p>Review before-and-after summaries for cleaning operations on tabular data.</p></div></article>
              <article><span aria-hidden="true">◎</span><div><h3>Features and pipelines</h3><p>When available, inspect selected features, pipeline comparisons, and the recommended result.</p></div></article>
            </div>
          </section>

          <section className="about-bottom-grid">
            <article className="about-detail-card">
              <span className="section-kicker">SUPPORTED INPUTS</span>
              <h2>Bring common data formats</h2>
              <p>DataOpti can process CSV, Excel, JSON, and Parquet datasets, along with ZIP archives, PDFs, text files, notebooks, and common image formats. Results depend on the contents and format of the files.</p>
              <div className="about-format-list" aria-label="Supported formats">
                <span>CSV</span><span>Excel</span><span>JSON</span><span>Parquet</span><span>PDF / TXT</span><span>ZIP</span><span>Notebooks</span><span>Images</span>
              </div>
            </article>
            <article className="about-detail-card about-history-note">
              <span className="section-kicker">YOUR WORKSPACE</span>
              <h2>Return to earlier analyses</h2>
              <p>Successful analyses appear in History and are saved in this browser for the signed-in account. Open an entry to review its analysis results.</p>
              <button className="secondary-btn" type="button" onClick={() => setPage("history")}>View history →</button>
            </article>
          </section>
        </main>
      )}

      <footer>

        <div>

          <strong>
            ⚡ DataOpti AI
          </strong>

          <p>
            Intelligent data optimization for better ML.
          </p>

        </div>

        <div>
          © 2026 DataOpti AI
        </div>

      </footer>

    </div>
  );
}


function App() {
  const [user, setUser] = useState(() => {
    try {
      return JSON.parse(
        localStorage.getItem("dataopti_logged_user") || "null"
      );
    } catch {
      return null;
    }
  });

  const handleLogout = () => {
    localStorage.removeItem("dataopti_logged_user");
    setUser(null);
  };

  return (
    <Routes>
      <Route
        path="/login"
        element={
          user
            ? <Navigate to="/" replace />
            : <Login onLogin={setUser} />
        }
      />
      <Route
        path="/register"
        element={
          user
            ? <Navigate to="/" replace />
            : <Register />
        }
      />
      <Route
        path="/*"
        element={
          user
            ? <Dashboard user={user} onLogout={handleLogout} />
            : <Navigate to="/login" replace />
        }
      />
    </Routes>
  );
}
/* ========================================================
   CLEANING PANEL
======================================================== */

function CleaningPanel({ cleaning, accuracyComparison }) {

  if (!cleaning) {
    return null;
  }

  return (
    <div className="panel model-panel">

      <div className="panel-heading">

        <div>

          <h2>
            🧹 Data Cleaning
          </h2>

          <p>Before cleaning vs after cleaning</p>

        </div>

      </div>

      <div className="quality-items">

        <div>
          <span>
            Rows
          </span>

          <strong>
            {cleaning.rows_before?.toLocaleString() ??
              "—"}
            {" → "}
            {cleaning.rows_after?.toLocaleString() ??
              "—"}
          </strong>
        </div>

        <div>
          <span>
            Missing Values
          </span>

          <strong>
            {cleaning.missing_before?.toLocaleString() ??
              "—"}
            {" → "}
            {cleaning.missing_after?.toLocaleString() ??
              "—"}
          </strong>
        </div>

        <div>
          <span>
            Duplicates
          </span>

          <strong>
            {cleaning.duplicates_before?.toLocaleString() ??
              "—"}
            {" → "}
            {cleaning.duplicates_after?.toLocaleString() ??
              "—"}
          </strong>
        </div>

        <div>
          <span>
            Outliers
          </span>

          <strong>
            {cleaning.outliers_before?.toLocaleString() ??
              "—"}
            {" → "}
            {cleaning.outliers_after?.toLocaleString() ??
              "—"}
          </strong>
        </div>

      </div>

      <MonitoringChart accuracyComparison={accuracyComparison} />

    </div>
  );
}

function MonitoringChart({ accuracyComparison }) {
  const before = Number(accuracyComparison?.before);
  const after = Number(accuracyComparison?.after);
  const hasScores = accuracyComparison?.before !== null && accuracyComparison?.before !== undefined
    && accuracyComparison?.after !== null && accuracyComparison?.after !== undefined
    && Number.isFinite(before) && Number.isFinite(after);
  const changePoints = hasScores ? (after - before) * 100 : null;
  const chartMaximum = 100;

  return (
    <section className="monitoring-chart" aria-labelledby="monitoring-title">
      <div className="monitoring-chart-heading">
        <div>
          <span className="section-kicker">MONITORING</span>
          <h3 id="monitoring-title">Model accuracy before and after cleaning</h3>
        </div>
        {hasScores && (
          <span className={`monitoring-change ${changePoints >= 0 ? "positive" : "negative"}`}>
            {changePoints > 0 ? "+" : ""}{changePoints.toFixed(1)} percentage points
          </span>
        )}
      </div>

      {hasScores ? (
        <div className="accuracy-chart" role="img" aria-label={`Model accuracy before cleaning ${formatPercent(before)} and after cleaning ${formatPercent(after)}`}>
          {[
            { label: "Before cleaning", value: before, color: "before" },
            { label: "After cleaning", value: after, color: "after" },
          ].map((item) => (
            <div className="accuracy-chart-row" key={item.label}>
              <span>{item.label}</span>
              <div className="accuracy-track" aria-hidden="true">
                <div className={`accuracy-bar ${item.color}`} style={{ width: `${Math.max(0, Math.min(chartMaximum, item.value * 100))}%` }} />
              </div>
              <strong>{formatPercent(item.value)}</strong>
            </div>
          ))}
          <div className="accuracy-axis" aria-hidden="true"><span>0%</span><span>50%</span><span>100%</span></div>
        </div>
      ) : (
        <p className="monitoring-unavailable">Accuracy comparison is unavailable for this dataset. A labeled target column and enough usable rows are needed.</p>
      )}

      <p className="monitoring-explainer">
        Accuracy is the share of held-out rows correctly classified by the best tested pipeline at each stage. Both runs use the same sampled input, though removing duplicate rows can change the rows in the after split. A higher score is a comparison signal, not a guarantee of better results on new data.
      </p>
    </section>
  );
}


/* ========================================================
   FEATURE SELECTION
======================================================== */

function FeatureSelectionPanel({
  modelResults
}) {

  if (
    !modelResults ||
    modelResults.status !== "complete"
  ) {
    return null;
  }

  return (
    <div className="panel model-panel">

      <div className="panel-heading">

        <div>

          <h2>
            🎯 Feature Selection
          </h2>

          <p>
            Useful and non-redundant features
            selected automatically.
          </p>

        </div>

      </div>

      <div className="quality-items">

        <div>
          <span>
            Features Before
          </span>

          <strong>
            {modelResults.feature_count_before ??
              "—"}
          </strong>
        </div>

        <div>
          <span>
            Features After
          </span>

          <strong>
            {modelResults.feature_count_after ??
              "—"}
          </strong>
        </div>

        <div>
          <span>
            Target
          </span>

          <strong>
            {modelResults.target_column ||
              "—"}
          </strong>
        </div>

        <div>
          <span>
            ML Sample
          </span>

          <strong>
            {modelResults.sample_rows?.toLocaleString() ??
              "—"}
          </strong>
        </div>

      </div>

      {modelResults.selected_features?.length > 0 && (

        <div className="supported">

          {modelResults.selected_features.map(
            (feature, index) => (

              <span
                key={`${feature}-${index}`}
              >
                ✓ {feature}
              </span>

            )
          )}

        </div>

      )}

    </div>
  );
}


/* ========================================================
   PIPELINE A / B / C
======================================================== */

function PipelinePanel({
  modelResults
}) {

  if (
    !modelResults ||
    !Array.isArray(
      modelResults.pipelines
    )
  ) {
    return null;
  }

  return (
    <div className="panel model-panel">

      <div className="panel-heading">

        <div>

          <h2>
            ⚙️ Feature Selection & ML Pipelines
          </h2>

          <p>
            Three approaches are tested and the
            strongest one is recommended.
          </p>

        </div>

      </div>

      <div className="quality-items">

        {modelResults.pipelines.map(
          (pipeline, index) => {

            const isBest =
              pipeline.name ===
              modelResults.best_pipeline;

            return (

              <div
                key={`${pipeline.name}-${index}`}
              >

                <span>

                  {isBest
                    ? "🏆 "
                    : ""}

                  {pipeline.name}

                </span>

                <strong>

                  {formatPercent(
                    pipeline.accuracy
                  )}

                </strong>

                <small>

                  {pipeline.selected_feature_count ??
                    0}{" "}
                  features ·{" "}

                  {pipeline.training_seconds ??
                    "—"}{" "}
                  sec

                </small>

              </div>

            );
          }
        )}

      </div>

      <div className="supported">

        {modelResults.pipelines.map(
          (pipeline, index) => (

            <span
              key={`${pipeline.name}-description-${index}`}
            >

              {pipeline.name}:{" "}
              {pipeline.description}

            </span>

          )
        )}

      </div>

    </div>
  );
}


/* ========================================================
   HELPERS
======================================================== */

function formatPercent(value) {

  if (
    value === null ||
    value === undefined ||
    !Number.isFinite(
      Number(value)
    )
  ) {
    return "—";
  }

  return `${(
    Number(value) * 100
  ).toFixed(1)}%`;
}

function getRecommendationModelName(model) {
  const modelNames = {
    "Pipeline A": "Logistic Regression",
    "Pipeline B": "Random Forest",
    "Pipeline C": "Extra Trees Classifier",
  };

  return modelNames[model] || model || "Initial Analysis";
}


/* ========================================================
   FEATURE CARD
======================================================== */

function Feature({
  icon,
  title,
  text
}) {

  return (

    <div className="feature-card">

      <div className="feature-icon">
        {icon}
      </div>

      <h3>
        {title}
      </h3>

      <p>
        {text}
      </p>

      <span className="learn">
        Learn more →
      </span>

    </div>
  );
}


/* ========================================================
   STAT CARD
======================================================== */

function Stat({
  icon,
  title,
  value,
  change
}) {

  return (

    <div className="stat-card">

      <div className="stat-icon">
        {icon}
      </div>

      <div>

        <small>
          {title}
        </small>

        <h2>
          {value}
        </h2>

        <span>
          {change}
        </span>

      </div>

    </div>
  );
}


export default App;
