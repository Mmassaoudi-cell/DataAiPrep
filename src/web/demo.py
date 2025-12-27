"""
DataAiPrep Web Demo Application

A lightweight browser-based interface for quick evaluation of DataAiPrep
capabilities without requiring PyQt6 installation.

Usage:
    python -m src.web.demo
    
    Then open http://localhost:8000 in your browser.
"""

import io
import json
import tempfile
from pathlib import Path
from typing import Optional
import pandas as pd

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def create_app() -> "FastAPI":
    """Create and configure the FastAPI application"""
    
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for the web demo. "
            "Install with: pip install fastapi uvicorn python-multipart"
        )
    
    app = FastAPI(
        title="DataAiPrep Web Demo",
        description="Browser-based data quality assessment tool",
        version="1.0.0"
    )
    
    # Store uploaded data temporarily
    app.state.current_data = None
    app.state.current_filename = None
    
    @app.get("/", response_class=HTMLResponse)
    async def home():
        """Serve the main demo page"""
        return get_demo_html()
    
    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """Upload a data file for analysis"""
        try:
            # Read file content
            content = await file.read()
            
            # Determine file type and load
            filename = file.filename.lower()
            
            if filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content))
            elif filename.endswith('.json'):
                df = pd.read_json(io.BytesIO(content))
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(content))
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file format. Use CSV, JSON, or Excel."
                )
            
            # Store in app state
            app.state.current_data = df
            app.state.current_filename = file.filename
            
            return {
                "success": True,
                "filename": file.filename,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                "preview": df.head(5).to_dict(orient='records'),
                "memory_mb": df.memory_usage(deep=True).sum() / 1024**2
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/analyze")
    async def analyze_data(target_column: Optional[str] = Form(None)):
        """Run data quality analysis on uploaded data"""
        
        if app.state.current_data is None:
            raise HTTPException(
                status_code=400,
                detail="No data uploaded. Please upload a file first."
            )
        
        try:
            df = app.state.current_data
            
            # Import analyzers
            from ..analysis.completeness_analyzer import CompletenessAnalyzer
            from ..analysis.distribution_analyzer import DistributionAnalyzer
            from ..analysis.leakage_detector import LeakageDetector
            from ..analysis.feature_quality_analyzer import FeatureQualityAnalyzer
            
            # Run analysis
            completeness = CompletenessAnalyzer().analyze(df)
            distribution = DistributionAnalyzer().analyze(
                df, 
                target_column if target_column and target_column != "None" else None
            )
            leakage = LeakageDetector().analyze(
                df,
                target_column if target_column and target_column != "None" else None
            )
            feature_quality = FeatureQualityAnalyzer().analyze(
                df,
                target_column if target_column and target_column != "None" else None
            )
            
            # Compile results
            results = {
                "filename": app.state.current_filename,
                "shape": list(df.shape),
                "target_column": target_column,
                "summary": {
                    "completeness_score": completeness.get('completeness_score', 0),
                    "missing_percentage": completeness.get('overall_missing_percentage', 0),
                    "leakage_score": leakage.get('leakage_score', 0),
                    "perfect_correlations": len(leakage.get('perfect_correlations', [])),
                    "duplicate_features": len(leakage.get('duplicate_features', [])),
                    "high_outlier_columns": len([
                        c for c, v in distribution.get('outliers', {}).items()
                        if isinstance(v, dict) and v.get('outlier_percentage', 0) > 10
                    ])
                },
                "completeness": {
                    "total_rows": completeness.get('total_rows', 0),
                    "total_columns": completeness.get('total_columns', 0),
                    "complete_columns_count": completeness.get('complete_columns_count', 0),
                    "columns_with_missing": completeness.get('columns_with_missing_count', 0),
                    "recommendations": completeness.get('recommendations', [])[:5]
                },
                "leakage": {
                    "perfect_correlations": leakage.get('perfect_correlations', [])[:10],
                    "duplicate_features": leakage.get('duplicate_features', [])[:10],
                    "suspicious_features": leakage.get('suspicious_features', [])[:10],
                    "recommendations": leakage.get('recommendations', [])[:5]
                },
                "distribution": {
                    "numeric_columns": distribution.get('numeric_columns', [])[:20],
                    "categorical_columns": distribution.get('categorical_columns', [])[:20],
                    "skewness_issues": [
                        k for k, v in distribution.get('skewness_analysis', {}).items()
                        if isinstance(v, dict) and abs(v.get('skewness', 0)) > 1
                    ][:10],
                    "recommendations": distribution.get('recommendations', [])[:5]
                },
                "issues": completeness.get('issues', [])[:5] + 
                         leakage.get('leakage_issues', [])[:5] +
                         distribution.get('issues', [])[:5]
            }
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/benchmark")
    async def list_benchmarks():
        """List available benchmark datasets"""
        try:
            from ..benchmark import list_available_benchmarks, BenchmarkDatasetGenerator
            
            available = list_available_benchmarks()
            
            if not available:
                # Generate if not exists
                generator = BenchmarkDatasetGenerator()
                summary = generator.get_dataset_summary()
                return {
                    "available": False,
                    "message": "Benchmark datasets not generated yet",
                    "datasets": summary.to_dict(orient='records')
                }
            
            return {
                "available": True,
                "datasets": available
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/benchmark/generate")
    async def generate_benchmarks():
        """Generate benchmark datasets"""
        try:
            from ..benchmark import BenchmarkDatasetGenerator
            
            save_path = Path(__file__).parent.parent.parent / 'benchmark_datasets'
            generator = BenchmarkDatasetGenerator(random_state=42)
            datasets = generator.generate_all_benchmarks(save_path=save_path)
            
            return {
                "success": True,
                "generated": len(datasets),
                "path": str(save_path),
                "datasets": list(datasets.keys())
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "version": "1.0.0"}
    
    return app


def get_demo_html() -> str:
    """Generate the demo HTML page"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataAiPrep - Web Demo</title>
    <style>
        :root {
            --primary: #2E86AB;
            --primary-dark: #1E5F8B;
            --success: #28a745;
            --warning: #ffc107;
            --danger: #dc3545;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #333333;
            --text-muted: #6c757d;
            --border: #dee2e6;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: var(--text);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        
        header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        header p {
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        .card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .card h2 {
            color: var(--primary);
            margin-bottom: 16px;
            font-size: 1.4rem;
            border-bottom: 2px solid var(--primary);
            padding-bottom: 8px;
        }
        
        .upload-zone {
            border: 3px dashed var(--border);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: var(--bg);
        }
        
        .upload-zone:hover {
            border-color: var(--primary);
            background: #e8f4f8;
        }
        
        .upload-zone.dragover {
            border-color: var(--primary);
            background: #d0e8f0;
        }
        
        .upload-zone input[type="file"] {
            display: none;
        }
        
        .upload-icon {
            font-size: 3rem;
            margin-bottom: 10px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
        }
        
        .btn-primary:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .metric-card {
            background: var(--bg);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
        }
        
        .metric-label {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 5px;
        }
        
        .metric-good { color: var(--success); }
        .metric-warning { color: var(--warning); }
        .metric-bad { color: var(--danger); }
        
        .issue-list {
            list-style: none;
        }
        
        .issue-list li {
            padding: 12px;
            margin-bottom: 8px;
            background: #fff3cd;
            border-left: 4px solid var(--warning);
            border-radius: 4px;
        }
        
        .issue-list li.critical {
            background: #f8d7da;
            border-left-color: var(--danger);
        }
        
        .recommendation-list li {
            background: #d4edda;
            border-left-color: var(--success);
        }
        
        select {
            padding: 10px 15px;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 1rem;
            width: 100%;
            max-width: 300px;
        }
        
        .data-preview {
            overflow-x: auto;
            margin-top: 15px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        th {
            background: var(--bg);
            font-weight: 600;
            color: var(--primary);
        }
        
        tr:hover td {
            background: #f5f5f5;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .hidden {
            display: none;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .badge-success { background: #d4edda; color: #155724; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-danger { background: #f8d7da; color: #721c24; }
        
        footer {
            text-align: center;
            margin-top: 30px;
            color: white;
            opacity: 0.8;
        }
        
        footer a {
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 DataAiPrep Web Demo</h1>
            <p>ML Data Quality Assessment - Browser-Based Evaluation</p>
        </header>
        
        <!-- Upload Section -->
        <div class="card">
            <h2>📁 Upload Data</h2>
            <div class="upload-zone" id="dropZone">
                <div class="upload-icon">📤</div>
                <p><strong>Drag & drop</strong> your data file here</p>
                <p style="color: var(--text-muted); margin: 10px 0;">or click to browse</p>
                <p style="font-size: 0.85rem; color: var(--text-muted);">
                    Supports: CSV, JSON, Excel (xlsx, xls)
                </p>
                <input type="file" id="fileInput" accept=".csv,.json,.xlsx,.xls">
            </div>
            
            <div id="fileInfo" class="hidden" style="margin-top: 20px;">
                <h3>📊 Loaded: <span id="fileName"></span></h3>
                <p>Shape: <span id="fileShape"></span> | Memory: <span id="fileMemory"></span></p>
                
                <div style="margin-top: 15px; display: flex; align-items: center; gap: 15px;">
                    <label for="targetSelect"><strong>Target Column:</strong></label>
                    <select id="targetSelect">
                        <option value="None">None (Unsupervised)</option>
                    </select>
                    <button class="btn btn-primary" id="analyzeBtn" onclick="analyzeData()">
                        🔍 Analyze Data Quality
                    </button>
                </div>
                
                <div class="data-preview" id="previewTable"></div>
            </div>
        </div>
        
        <!-- Loading Indicator -->
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing data quality...</p>
        </div>
        
        <!-- Results Section -->
        <div id="results" class="hidden">
            <div class="card">
                <h2>📈 Quality Summary</h2>
                <div class="row">
                    <div class="metric-card">
                        <div class="metric-value" id="completenessScore">-</div>
                        <div class="metric-label">Completeness Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="missingPct">-</div>
                        <div class="metric-label">Missing Data</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="leakageScore">-</div>
                        <div class="metric-label">Leakage Risk</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="correlations">-</div>
                        <div class="metric-label">Perfect Correlations</div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="card">
                    <h2>⚠️ Issues Found</h2>
                    <ul class="issue-list" id="issueList"></ul>
                </div>
                
                <div class="card">
                    <h2>💡 Recommendations</h2>
                    <ul class="issue-list recommendation-list" id="recommendationList"></ul>
                </div>
            </div>
            
            <div class="card">
                <h2>🔗 Data Leakage Details</h2>
                <div id="leakageDetails"></div>
            </div>
        </div>
        
        <footer>
            <p>DataAiPrep - ML Data Quality Assessment Tool</p>
            <p>Open Source | <a href="https://github.com/massaoudi-lab/dataiprep">GitHub</a></p>
        </footer>
    </div>
    
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        
        // Drag and drop handlers
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                uploadFile(e.dataTransfer.files[0]);
            }
        });
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                uploadFile(e.target.files[0]);
            }
        });
        
        async function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Upload failed');
                }
                
                const data = await response.json();
                showFileInfo(data);
                
            } catch (error) {
                alert('Error uploading file: ' + error.message);
            }
        }
        
        function showFileInfo(data) {
            document.getElementById('fileName').textContent = data.filename;
            document.getElementById('fileShape').textContent = 
                `${data.rows.toLocaleString()} rows × ${data.columns} columns`;
            document.getElementById('fileMemory').textContent = 
                data.memory_mb.toFixed(2) + ' MB';
            
            // Populate target selector
            const select = document.getElementById('targetSelect');
            select.innerHTML = '<option value="None">None (Unsupervised)</option>';
            data.column_names.forEach(col => {
                select.innerHTML += `<option value="${col}">${col}</option>`;
            });
            
            // Show preview table
            if (data.preview.length > 0) {
                let html = '<table><tr>';
                Object.keys(data.preview[0]).forEach(key => {
                    html += `<th>${key}</th>`;
                });
                html += '</tr>';
                data.preview.forEach(row => {
                    html += '<tr>';
                    Object.values(row).forEach(val => {
                        const display = val === null ? '<em>null</em>' : 
                            String(val).substring(0, 50);
                        html += `<td>${display}</td>`;
                    });
                    html += '</tr>';
                });
                html += '</table>';
                document.getElementById('previewTable').innerHTML = html;
            }
            
            document.getElementById('fileInfo').classList.remove('hidden');
            document.getElementById('results').classList.add('hidden');
        }
        
        async function analyzeData() {
            const targetColumn = document.getElementById('targetSelect').value;
            const formData = new FormData();
            formData.append('target_column', targetColumn);
            
            document.getElementById('loading').classList.add('active');
            document.getElementById('results').classList.add('hidden');
            document.getElementById('analyzeBtn').disabled = true;
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Analysis failed');
                }
                
                const results = await response.json();
                showResults(results);
                
            } catch (error) {
                alert('Error analyzing data: ' + error.message);
            } finally {
                document.getElementById('loading').classList.remove('active');
                document.getElementById('analyzeBtn').disabled = false;
            }
        }
        
        function showResults(results) {
            const s = results.summary;
            
            // Update metrics
            updateMetric('completenessScore', s.completeness_score.toFixed(1) + '%', 
                s.completeness_score >= 95 ? 'good' : s.completeness_score >= 80 ? 'warning' : 'bad');
            updateMetric('missingPct', s.missing_percentage.toFixed(1) + '%',
                s.missing_percentage <= 5 ? 'good' : s.missing_percentage <= 15 ? 'warning' : 'bad');
            updateMetric('leakageScore', s.leakage_score.toFixed(0) + '/100',
                s.leakage_score <= 30 ? 'good' : s.leakage_score <= 60 ? 'warning' : 'bad');
            updateMetric('correlations', s.perfect_correlations,
                s.perfect_correlations === 0 ? 'good' : s.perfect_correlations <= 5 ? 'warning' : 'bad');
            
            // Issues list
            const issueList = document.getElementById('issueList');
            issueList.innerHTML = '';
            if (results.issues.length === 0) {
                issueList.innerHTML = '<li style="background: #d4edda; border-left-color: #28a745;">No critical issues found!</li>';
            } else {
                results.issues.forEach(issue => {
                    issueList.innerHTML += `<li>${issue}</li>`;
                });
            }
            
            // Recommendations
            const recList = document.getElementById('recommendationList');
            recList.innerHTML = '';
            const recs = [
                ...results.completeness.recommendations,
                ...results.leakage.recommendations,
                ...results.distribution.recommendations
            ].slice(0, 5);
            
            if (recs.length === 0) {
                recList.innerHTML = '<li>Data looks good! No specific recommendations.</li>';
            } else {
                recs.forEach(rec => {
                    recList.innerHTML += `<li>${rec}</li>`;
                });
            }
            
            // Leakage details
            const leakageDiv = document.getElementById('leakageDetails');
            let html = '';
            
            if (results.leakage.perfect_correlations.length > 0) {
                html += '<h4>Perfect Correlations:</h4><ul>';
                results.leakage.perfect_correlations.forEach(pair => {
                    html += `<li>${pair[0]} ↔ ${pair[1]} (correlation: ${pair[2]?.toFixed(3) || '1.0'})</li>`;
                });
                html += '</ul>';
            }
            
            if (results.leakage.duplicate_features.length > 0) {
                html += '<h4>Duplicate Features:</h4><ul>';
                results.leakage.duplicate_features.forEach(pair => {
                    html += `<li>${pair[0]} = ${pair[1]}</li>`;
                });
                html += '</ul>';
            }
            
            if (html === '') {
                html = '<p style="color: #28a745;">✓ No data leakage detected</p>';
            }
            
            leakageDiv.innerHTML = html;
            
            document.getElementById('results').classList.remove('hidden');
        }
        
        function updateMetric(id, value, status) {
            const el = document.getElementById(id);
            el.textContent = value;
            el.className = 'metric-value metric-' + status;
        }
    </script>
</body>
</html>
'''


def main():
    """Run the web demo server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DataAiPrep Web Demo')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    if not HAS_FASTAPI:
        print("Error: FastAPI is required for the web demo.")
        print("Install with: pip install fastapi uvicorn python-multipart")
        return
    
    print(f"\n🌐 Starting DataAiPrep Web Demo...")
    print(f"📍 Open http://{args.host}:{args.port} in your browser\n")
    
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

