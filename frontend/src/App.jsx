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
      <header className="app-header">
        <h1>CivicLens AI</h1>
        <div className="header-right">
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

function ApiStatusBadge({ status }) {
  const labels = {
    checking: "Checking API…",
    ok: "API connected",
    "no-models": "API up, models not trained",
    down: "API unreachable (start uvicorn on :8000)",
  };
  return <span className={`api-badge api-${status}`}>{labels[status]}</span>;
}
