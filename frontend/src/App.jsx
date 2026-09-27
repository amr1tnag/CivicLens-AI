import { useEffect, useState } from "react";
import ComplaintForm from "./components/ComplaintForm";
import Dashboard from "./components/Dashboard";
import { getHealth } from "./api";
import "./App.css";

export default function App() {
  const [tab, setTab] = useState("submit");
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    getHealth()
      .then((h) => setApiStatus(h.models_loaded ? "ok" : "no-models"))
      .catch(() => setApiStatus("down"));
  }, []);

  return (
    <div className="app">
      <header className="title-block">
        <div className="title-block-mark">
          <CompassMark />
          <div className="title-block-name">
            <h1>CivicLens</h1>
            <span className="title-block-sub">Report &amp; Routing System</span>
          </div>
        </div>

        <div className="title-block-meta">
          <nav className="tabs">
            <button
              className={tab === "submit" ? "active" : ""}
              onClick={() => setTab("submit")}
            >
              Report an issue
            </button>
            <button
              className={tab === "dashboard" ? "active" : ""}
              onClick={() => setTab("dashboard")}
            >
              Staff dashboard
            </button>
          </nav>
          <ApiStatusBadge status={apiStatus} />
        </div>
      </header>

      <main>
        {tab === "submit" ? <ComplaintForm /> : <Dashboard />}
      </main>
    </div>
  );
}

function CompassMark() {
  return (
    <svg viewBox="0 0 40 40" className="compass-mark" aria-hidden="true">
      <circle cx="20" cy="20" r="17" fill="none" stroke="currentColor" strokeWidth="1.4" />
      <path d="M20 6 L20 10 M20 30 L20 34 M6 20 L10 20 M30 20 L34 20" stroke="currentColor" strokeWidth="1.4" />
      <path d="M20 12 L25 20 L20 28 L15 20 Z" fill="none" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

function ApiStatusBadge({ status }) {
  const labels = {
    checking: "Checking API",
    ok: "API connected",
    "no-models": "API up, models not trained",
    down: "API unreachable — start uvicorn on :8000",
  };
  return <span className={`api-badge api-${status}`}>{labels[status]}</span>;
}
