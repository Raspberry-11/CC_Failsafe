const API = "http://localhost:8000";

// ── Auth helpers ──────────────────────────────────────────────────────────

function requireAuth() {
  if (!localStorage.getItem("token")) {
    window.location.href = "login.html";
  }
}

function logout() {
  localStorage.removeItem("token");
  localStorage.removeItem("user_name");
  localStorage.removeItem("user_role");
  window.location.href = "login.html";
}

// ── Authenticated fetch ───────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const token = localStorage.getItem("token");
  const headers = {
    ...(options.headers || {}),
    Authorization: `Bearer ${token}`,
  };

  try {
    const res = await fetch(`${API}${path}`, { ...options, headers });

    if (res.status === 401) {
      logout();
      return null;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      showToast(err.detail || "Request failed", true);
      return null;
    }

    return res.status === 204 ? {} : await res.json();
  } catch {
    showToast("Cannot reach backend — is the server running?", true);
    return null;
  }
}

// ── HTML escape ────────────────────────────────────────────────────────────

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c]
  ));
}

// ── Toast notification ────────────────────────────────────────────────────

function showToast(msg, isError = false) {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = msg;
  el.style.borderColor  = isError ? "var(--red)"  : "var(--border)";
  el.style.color        = isError ? "var(--red)"  : "var(--text)";
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 3000);
}
