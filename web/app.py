"""
FastAPI web application for xstat dashboard.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import os
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn


def create_app(data_dir: str = "runs") -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="xstat Dashboard",
        description="Cross-Asset Statistical Arbitrage Research Platform",
        version="0.1.0"
    )
    
    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Setup templates
    templates = Jinja2Templates(directory=str(static_dir / "templates"))
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Main dashboard page."""
        return templates.TemplateResponse("dashboard.html", {"request": request})
    
    @app.get("/api/runs")
    async def get_runs():
        """Get list of research runs."""
        try:
            runs = []
            runs_dir = Path(data_dir)
            
            if runs_dir.exists():
                for run_dir in runs_dir.iterdir():
                    if run_dir.is_dir():
                        run_info = {
                            "name": run_dir.name,
                            "path": str(run_dir),
                            "created": datetime.fromtimestamp(run_dir.stat().st_ctime).isoformat(),
                            "modified": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat()
                        }
                        
                        # Check for summary file
                        summary_file = run_dir / "research_summary.yaml"
                        if summary_file.exists():
                            try:
                                import yaml
                                with open(summary_file) as f:
                                    summary = yaml.safe_load(f)
                                run_info["summary"] = summary
                            except Exception:
                                pass
                        
                        runs.append(run_info)
            
            return {"runs": runs}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/runs/{run_name}")
    async def get_run_details(run_name: str):
        """Get details for a specific run."""
        try:
            run_dir = Path(data_dir) / run_name
            
            if not run_dir.exists():
                raise HTTPException(status_code=404, detail="Run not found")
            
            # Load summary
            summary_file = run_dir / "research_summary.yaml"
            if summary_file.exists():
                import yaml
                with open(summary_file) as f:
                    summary = yaml.safe_load(f)
            else:
                summary = {}
            
            # Load candidates
            candidates_file = run_dir / "candidates.csv"
            candidates = []
            if candidates_file.exists():
                import pandas as pd
                df = pd.read_csv(candidates_file)
                candidates = df.to_dict('records')
            
            # Load performance
            performance_file = run_dir / "performance.json"
            performance = {}
            if performance_file.exists():
                with open(performance_file) as f:
                    performance = json.load(f)
            
            return {
                "name": run_name,
                "summary": summary,
                "candidates": candidates,
                "performance": performance
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/runs/{run_name}/candidates")
    async def get_candidates(run_name: str, limit: int = 50):
        """Get candidates for a run."""
        try:
            candidates_file = Path(data_dir) / run_name / "candidates.csv"
            
            if not candidates_file.exists():
                raise HTTPException(status_code=404, detail="Candidates file not found")
            
            import pandas as pd
            df = pd.read_csv(candidates_file)
            
            # Sort by overall score
            if 'overall_score' in df.columns:
                df = df.sort_values('overall_score', ascending=False)
            
            # Limit results
            df = df.head(limit)
            
            return {"candidates": df.to_dict('records')}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/runs/{run_name}/performance")
    async def get_performance(run_name: str):
        """Get performance metrics for a run."""
        try:
            performance_file = Path(data_dir) / run_name / "performance.json"
            
            if not performance_file.exists():
                raise HTTPException(status_code=404, detail="Performance file not found")
            
            with open(performance_file) as f:
                performance = json.load(f)
            
            return performance
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    @app.get("/api/stats")
    async def get_stats():
        """Get platform statistics."""
        try:
            runs_dir = Path(data_dir)
            total_runs = 0
            total_pairs = 0
            
            if runs_dir.exists():
                for run_dir in runs_dir.iterdir():
                    if run_dir.is_dir():
                        total_runs += 1
                        
                        # Count pairs in candidates file
                        candidates_file = run_dir / "candidates.csv"
                        if candidates_file.exists():
                            import pandas as pd
                            df = pd.read_csv(candidates_file)
                            total_pairs += len(df)
            
            return {
                "total_runs": total_runs,
                "total_pairs": total_pairs,
                "data_dir": str(runs_dir.absolute())
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
