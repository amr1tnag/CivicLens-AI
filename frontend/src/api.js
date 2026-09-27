const BASE_URL = "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ? JSON.stringify(body.detail) : detail;
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export function getHealth() {
  return request("/health");
}

export function submitComplaint({ text, location }) {
  return request("/complaints", {
    method: "POST",
    body: JSON.stringify({ text, location: location || null }),
  });
}

export function listComplaints(filters = {}) {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value) params.set(key, value);
  });
  const qs = params.toString();
  return request(`/complaints${qs ? `?${qs}` : ""}`);
}

export function updateStatus(complaintId, status) {
  return request(`/complaints/${complaintId}/status`, {
    method: "PATCH",
    body: JSON.stringify({ status }),
  });
}

export function getDashboardStats() {
  return request("/stats/dashboard");
}
