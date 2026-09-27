import { useEffect, useState, useCallback } from "react";
import { getDashboardStats, listComplaints, updateStatus } from "../api";

const STATUSES = ["Open", "In Progress", "Resolved"];

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [complaints, setComplaints] = useState([]);
  const [filters, setFilters] = useState({ category: "", urgency: "", department: "", status: "" });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [statsData, complaintsData] = await Promise.all([
        getDashboardStats(),
        listComplaints(filters),
      ]);
      setStats(statsData);
      setComplaints(complaintsData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    load();
  }, [load]);

  async function handleStatusChange(complaintId, status) {
    try {
      await updateStatus(complaintId, status);
      setComplaints((prev) =>
        prev.map((c) => (c.complaint_id === complaintId ? { ...c, status } : c))
      );
    } catch (err) {
      setError(err.message);
    }
  }

  function setFilter(key, value) {
    setFilters((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <div className="panel">
      <div className="dashboard-header">
        <h2>Staff triage dashboard</h2>
        <button type="button" className="secondary" onClick={load} disabled={loading}>
          Refresh
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      {stats && (
        <div className="stats-row">
          <StatTile label="Total" value={stats.total_complaints} />
          <StatTile label="Open" value={stats.open_count} />
          <StatTile label="In progress" value={stats.in_progress_count} />
          <StatTile label="Resolved" value={stats.resolved_count} />
          <StatTile label="Avg priority" value={stats.average_priority_score} />
        </div>
      )}

      {stats && (
        <div className="breakdowns">
          <Breakdown title="By category" items={stats.by_category} keyField="category" />
          <Breakdown title="By urgency" items={stats.by_urgency} keyField="urgency" />
          <Breakdown title="By department" items={stats.by_department} keyField="department" />
        </div>
      )}

      <div className="filters">
        <select value={filters.status} onChange={(e) => setFilter("status", e.target.value)}>
          <option value="">All statuses</option>
          {STATUSES.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <select value={filters.urgency} onChange={(e) => setFilter("urgency", e.target.value)}>
          <option value="">All urgency levels</option>
          {["Critical", "High", "Medium", "Low"].map((u) => (
            <option key={u} value={u}>{u}</option>
          ))}
        </select>
        <input
          placeholder="Filter by category..."
          value={filters.category}
          onChange={(e) => setFilter("category", e.target.value)}
        />
        <input
          placeholder="Filter by department..."
          value={filters.department}
          onChange={(e) => setFilter("department", e.target.value)}
        />
      </div>

      {loading ? (
        <p className="muted">Loading...</p>
      ) : complaints.length === 0 ? (
        <p className="muted">No complaints match these filters yet.</p>
      ) : (
        <table className="complaints-table">
          <thead>
            <tr>
              <th>Priority</th>
              <th>Category</th>
              <th>Urgency</th>
              <th>Department</th>
              <th>Location</th>
              <th>Text</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {complaints.map((c) => (
              <tr key={c.complaint_id}>
                <td>{c.priority_score.toFixed(1)}</td>
                <td>{c.category}</td>
                <td>{c.final_urgency_level}</td>
                <td>{c.escalated_department || c.department}</td>
                <td>{c.location || "—"}</td>
                <td className="text-cell" title={c.text}>{c.text}</td>
                <td>
                  <select
                    value={c.status}
                    onChange={(e) => handleStatusChange(c.complaint_id, e.target.value)}
                  >
                    {STATUSES.map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function StatTile({ label, value }) {
  return (
    <div className="stat-tile">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

function Breakdown({ title, items, keyField }) {
  return (
    <div className="breakdown">
      <h3>{title}</h3>
      <ul>
        {items.map((item) => (
          <li key={item[keyField]}>
            <span>{item[keyField]}</span>
            <span className="muted">{item.count}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
