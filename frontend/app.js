const API_BASE = window.location.origin;

// State
let currentMode = "HYBRID";
let currentAlpha = 0.7;
let currentTopK = 5;

// Sample Technical Corpora
const SAMPLE_CORPORA = {
  raft: {
    title: "Raft Distributed Consensus Protocol",
    author: "Systems Research Lab",
    tags: "distributed-systems, consensus, raft, replication",
    content: `The Raft consensus protocol was developed to solve distributed state machine consensus while remaining significantly easier to understand than Paxos. Raft decomposes the consensus problem into three independent subproblems: leader election, log replication, and safety guarantees.

During normal operation, the cluster consists of exactly one leader and multiple followers. The leader continuously sends heartbeats to maintain its authority. If a follower detects that an election timeout has elapsed without receiving a heartbeat from the leader, it transitions to candidate state, increments its term counter, votes for itself, and broadcasts RequestVote RPCs.

When a candidate secures votes from a strict majority of cluster nodes, it transitions into leader state and begins servicing client requests. Client commands are appended to the leader's write-ahead log as log entries. The leader then issues AppendEntries RPCs to replicate these entries across follower nodes. Once an entry is committed by a majority of nodes, the leader applies it to its local state machine and returns success to the client.`
  },
  wal: {
    title: "Database Write-Ahead Logging & Crash Recovery",
    author: "Database Engine Architecture",
    tags: "database, storage, durability, logging, acid",
    content: `Write-Ahead Logging (WAL) is the fundamental architectural pattern used by modern relational and transactional databases to satisfy ACID durability guarantees without sacrificing write throughput. In a WAL-enabled storage engine, changes made to data pages must first be sequentially written and flushed to an append-only log on disk before dirty in-memory pages are flushed to the primary table storage.

This append-only invariant ensures that sequential disk write speeds are leveraged during active online transaction processing. In the event of an unpredicted power loss or system crash, the database engine executes recovery by scanning the write-ahead log from the last recorded checkpoint.

During analysis and redo passes, all committed transactions recorded in the log that were not yet flushed to disk pages are reapplied. During the undo pass, active transactions that were left uncommitted at the time of failure are cleanly rolled back, restoring the storage engine to a mathematically consistent state.`
  },
  zerotrust: {
    title: "Zero-Trust Cybersecurity Architecture & Microsegmentation",
    author: "Enterprise Security Architecture",
    tags: "security, zerotrust, identity, network, microsegmentation",
    content: `The Zero-Trust security model operates on the core axiom of 'Never Trust, Always Verify'. Unlike legacy perimeter-based security architectures that assume all devices and users inside the internal network are benign, Zero-Trust treats all network traffic as untrusted regardless of physical or logical network origin.

Continuous identity verification is enforced through dynamic multi-factor authentication, risk-based posture assessment, and least-privilege role-based access control. Every application transaction and microservice-to-microservice call requires mutual cryptographic TLS authentication and short-lived JSON Web Tokens.

Network microsegmentation creates isolated security zones across workloads, preventing lateral movement by unauthorized attackers in the event of an isolated credential compromise. Comprehensive telemetry and automated threat scoring analyze anomaly signals in real time.`
  }
};

// DOM Elements
const metricDocs = document.getElementById("metric-docs");
const metricChunks = document.getElementById("metric-chunks");
const metricSearches = document.getElementById("metric-searches");
const metricLatency = document.getElementById("metric-latency");

const btnToggleIngest = document.getElementById("btn-toggle-ingest");
const ingestDrawer = document.getElementById("ingest-drawer");
const btnCloseDrawer = document.getElementById("btn-close-drawer");
const btnCancelIngest = document.getElementById("btn-cancel-ingest");

const ingestForm = document.getElementById("ingest-form");
const docTitle = document.getElementById("doc-title");
const docAuthor = document.getElementById("doc-author");
const docTags = document.getElementById("doc-tags");
const docContent = document.getElementById("doc-content");

const searchForm = document.getElementById("search-form");
const queryInput = document.getElementById("query-input");
const btnSearch = document.getElementById("btn-search");
const alphaSlider = document.getElementById("alpha-slider");
const alphaLabel = document.getElementById("alpha-label");
const alphaSliderWrap = document.getElementById("alpha-slider-wrap");
const topKSelect = document.getElementById("top-k-select");

const resultsContainer = document.getElementById("results-container");
const resultsHeadline = document.getElementById("results-headline");
const resultsMeta = document.getElementById("results-meta");

const docList = document.getElementById("doc-list");
const corpusCount = document.getElementById("corpus-count");

// Initialization
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadAnalytics();
  loadCorpus();
});

function setupEventListeners() {
  btnToggleIngest.addEventListener("click", () => ingestDrawer.classList.toggle("hidden"));
  btnCloseDrawer.addEventListener("click", () => ingestDrawer.classList.add("hidden"));
  btnCancelIngest.addEventListener("click", () => ingestDrawer.classList.add("hidden"));

  // Preset Corpora
  document.querySelectorAll(".btn-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.sample;
      if (SAMPLE_CORPORA[key]) {
        const s = SAMPLE_CORPORA[key];
        docTitle.value = s.title;
        docAuthor.value = s.author;
        docTags.value = s.tags;
        docContent.value = s.content;
      }
    });
  });

  ingestForm.addEventListener("submit", handleIngestSubmit);
  searchForm.addEventListener("submit", handleSearchSubmit);

  // Quick Query Chips
  document.querySelectorAll(".chip-item").forEach(chip => {
    chip.addEventListener("click", () => {
      queryInput.value = chip.dataset.q;
      handleSearchSubmit(new Event("submit"));
    });
  });

  // Mode Buttons
  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentMode = btn.dataset.mode;

      if (currentMode === "HYBRID") {
        alphaSliderWrap.style.display = "flex";
      } else {
        alphaSliderWrap.style.display = "none";
      }

      if (queryInput.value.trim()) {
        handleSearchSubmit(new Event("submit"));
      }
    });
  });

  // Alpha Slider
  alphaSlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    currentAlpha = val / 100.0;
    alphaLabel.textContent = `${val}% Semantic / ${100 - val}% Lexical`;
  });

  // Top K
  topKSelect.addEventListener("change", (e) => {
    currentTopK = parseInt(e.target.value, 10);
    if (queryInput.value.trim()) {
      handleSearchSubmit(new Event("submit"));
    }
  });
}

async function loadAnalytics() {
  try {
    const res = await fetch(`${API_BASE}/api/analytics/stats`);
    if (!res.ok) return;
    const stats = await res.json();
    metricDocs.textContent = stats.total_documents;
    metricChunks.textContent = stats.total_chunks;
    metricSearches.textContent = stats.total_searches;
    metricLatency.textContent = `${stats.avg_latency_ms} ms`;
  } catch (err) {
    console.error("Failed to load analytics:", err);
  }
}

async function loadCorpus() {
  try {
    const res = await fetch(`${API_BASE}/api/documents`);
    if (!res.ok) throw new Error("Failed to load corpus");
    const docs = await res.json();
    corpusCount.textContent = `${docs.length} Docs`;
    renderCorpusList(docs);

    // If corpus is empty, auto-populate with the Raft sample
    if (docs.length === 0) {
      await autoSeedCorpus();
    }
  } catch (err) {
    docList.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;">No documents indexed.</div>`;
  }
}

async function autoSeedCorpus() {
  for (const key of ["raft", "wal", "zerotrust"]) {
    const s = SAMPLE_CORPORA[key];
    await fetch(`${API_BASE}/api/documents/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(s)
    });
  }
  await loadAnalytics();
  const res = await fetch(`${API_BASE}/api/documents`);
  const docs = await res.json();
  corpusCount.textContent = `${docs.length} Docs`;
  renderCorpusList(docs);
}

function renderCorpusList(docs) {
  if (docs.length === 0) {
    docList.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem;padding:0.5rem;">Corpus is currently empty.</div>`;
    return;
  }

  docList.innerHTML = docs.map(d => `
    <div class="doc-card">
      <div class="doc-card-title">${escapeHtml(d.title)}</div>
      <div class="doc-card-sub">
        <span>${d.total_chunks} chunks · ${d.raw_length} chars</span>
        <button class="btn-del-doc" onclick="deleteDocument('${d.id}')">Delete</button>
      </div>
    </div>
  `).join("");
}

async function handleIngestSubmit(e) {
  e.preventDefault();
  const submitBtn = document.getElementById("btn-submit-doc");
  submitBtn.disabled = true;
  submitBtn.textContent = "Vectorizing Chunks...";

  const payload = {
    title: docTitle.value.trim(),
    author: docAuthor.value.trim(),
    tags: docTags.value.trim(),
    content: docContent.value.trim(),
    chunk_size: 450,
    chunk_overlap: 75
  };

  try {
    const res = await fetch(`${API_BASE}/api/documents/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Ingestion failed: ${err.detail || "Error"}`);
      return;
    }

    ingestForm.reset();
    ingestDrawer.classList.add("hidden");
    await loadCorpus();
    await loadAnalytics();
  } catch (err) {
    alert(`Request failed: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Process & Vectorize Document";
  }
}

async function handleSearchSubmit(e) {
  e.preventDefault();
  const q = queryInput.value.trim();
  if (!q) return;

  btnSearch.disabled = true;
  btnSearch.textContent = "Searching...";

  const payload = {
    query: q,
    mode: currentMode,
    alpha: currentAlpha,
    top_k: currentTopK
  };

  try {
    const res = await fetch(`${API_BASE}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("Search execution failed");
    const data = await res.json();
    renderSearchResults(data);
    await loadAnalytics();
  } catch (err) {
    resultsContainer.innerHTML = `<div style="color:var(--danger);padding:1rem;">${err.message}</div>`;
  } finally {
    btnSearch.disabled = false;
    btnSearch.textContent = "Search";
  }
}

function renderSearchResults(data) {
  resultsHeadline.textContent = `Results for "${data.query}"`;
  resultsMeta.textContent = `${data.results.length} hits · ${data.total_evaluated} chunks evaluated in ${data.latency_ms} ms (${data.mode})`;

  if (data.results.length === 0) {
    resultsContainer.innerHTML = `
      <div class="empty-placeholder">
        <div class="placeholder-icon">⚠️</div>
        <h3>No matching chunks found</h3>
        <p>Try modifying your search query or switching to Semantic Only mode.</p>
      </div>
    `;
    return;
  }

  const queryTerms = data.query.toLowerCase().split(/\s+/).filter(w => w.length > 2);

  resultsContainer.innerHTML = data.results.map((r, idx) => {
    const semWidth = Math.round(r.semantic_score * 100);
    const lexWidth = Math.round(r.lexical_score * 100);
    const highlightedSnippet = highlightText(r.snippet, queryTerms);

    return `
      <div class="result-card">
        <div class="result-top">
          <div class="doc-info">
            <h3>#${idx + 1} ${escapeHtml(r.document_title)}</h3>
            <span>Chunk ${r.chunk_index + 1} · Source: ${escapeHtml(r.document_author)}</span>
          </div>
          <div class="score-badge">
            <div class="hybrid-pill">${(r.hybrid_score * 100).toFixed(1)}% Score</div>
            <div class="score-breakdown-text">Sem: ${(r.semantic_score * 100).toFixed(0)}% · Lex: ${(r.lexical_score * 100).toFixed(0)}%</div>
          </div>
        </div>

        <div class="score-bar-wrap">
          <div class="score-sem" style="width: ${semWidth * 0.5}%;" title="Semantic Score: ${r.semantic_score}"></div>
          <div class="score-lex" style="width: ${lexWidth * 0.5}%;" title="BM25 Lexical Score: ${r.lexical_score}"></div>
        </div>

        <div class="snippet-box">${highlightedSnippet}</div>
      </div>
    `;
  }).join("");
}

function highlightText(text, terms) {
  let escaped = escapeHtml(text);
  for (const term of terms) {
    const regex = new RegExp(`(${escapeRegex(term)})`, 'gi');
    escaped = escaped.replace(regex, '<span class="highlight-token">$1</span>');
  }
  return escaped;
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function deleteDocument(id) {
  if (!confirm("Are you sure you want to remove this document and all its indexed chunks?")) return;
  try {
    const res = await fetch(`${API_BASE}/api/documents/${id}`, { method: "DELETE" });
    if (res.ok) {
      await loadCorpus();
      await loadAnalytics();
      if (queryInput.value.trim()) {
        handleSearchSubmit(new Event("submit"));
      }
    }
  } catch (err) {
    alert("Delete failed");
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
