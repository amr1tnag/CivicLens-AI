import { useState } from "react";
import { submitComplaint } from "../api";

const URGENCY_TONE = {
  Critical: "urgency-critical",
  High: "urgency-high",
  Medium: "urgency-medium",
  Low: "urgency-low",
};

export default function ComplaintForm() {
  const [text, setText] = useState("");
  const [location, setLocation] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [ticket, setTicket] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const result = await submitComplaint({ text, location });
      setTicket(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  function handleReset() {
    setText("");
    setLocation("");
    setTicket(null);
    setError(null);
  }

  return (
    <div className="sheet">
      <div className="panel panel-corners">
        <span className="sheet-no">FORM 01</span>
        <h2>Report a civic issue</h2>
        <p className="subtitle">
          Describe the problem in your own words. Potholes, water outages, garbage
          collection, streetlights, and more — we&apos;ll route it to the right department.
        </p>

        <form onSubmit={handleSubmit} className="complaint-form">
          <label htmlFor="text" className="field-label">What&apos;s the issue</label>
          <textarea
            id="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            minLength={10}
            maxLength={4000}
            rows={5}
            required
            placeholder="e.g. There's a huge pothole on MG Road, a bike already crashed into it, please fix immediately!"
          />

          <label htmlFor="location" className="field-label">Location (optional)</label>
          <input
            id="location"
            type="text"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            maxLength={200}
            placeholder="e.g. Ward 2 - Old Town"
          />

          <div className="form-actions">
            <button type="submit" disabled={submitting || text.trim().length < 10}>
              {submitting ? "Submitting…" : "Submit report"}
            </button>
            {(ticket || error) && (
              <button type="button" className="secondary" onClick={handleReset}>
                Report another
              </button>
            )}
          </div>
        </form>

        {error && <div className="error-box">Couldn&apos;t submit: {error}</div>}
      </div>

      {ticket && (
        <div className={`work-order work-order-enter ${URGENCY_TONE[ticket.final_urgency_level] || ""}`}>
          <div className="work-order-stub">
            <span className="stub-label">TICKET</span>
            <span className="stub-id">{ticket.complaint_id}</span>
          </div>

          <div className="work-order-body">
            <div className="work-order-header">
              <span className={`urgency-badge ${URGENCY_TONE[ticket.final_urgency_level] || ""}`}>
                {ticket.final_urgency_level} urgency
              </span>
              <span className="priority-score">Priority {ticket.priority_score.toFixed(1)} / 100</span>
            </div>

            <dl className="ticket-grid">
              <dt>Category</dt>
              <dd>
                {ticket.category}{" "}
                <span className="muted">({(ticket.category_confidence * 100).toFixed(0)}% confidence)</span>
              </dd>

              <dt>Routed to</dt>
              <dd>{ticket.escalated_department || ticket.department}</dd>

              <dt>SLA deadline</dt>
              <dd className="mono">
                {ticket.sla_hours}h — {new Date(ticket.sla_deadline).toLocaleString()}
              </dd>

              {ticket.duplicate_count > 0 && (
                <>
                  <dt>Similar reports</dt>
                  <dd>{ticket.duplicate_count} other citizen(s) reported this already</dd>
                </>
              )}
            </dl>

            {ticket.explanation?.matched_urgent_terms?.length > 0 && (
              <p className="explanation">
                Flagged as urgent because of: {ticket.explanation.matched_urgent_terms.join(", ")}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
