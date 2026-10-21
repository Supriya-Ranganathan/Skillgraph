
"use strict";

const $ = (selector) => document.querySelector(selector);

const STORAGE_KEYS = {
  plan: "skillgraph-x:personal-plan:v1",
  recent: "skillgraph-x:recent-analysis:v1",
};

const appState = {
  roles: [],
  categories: [],
  topSkills: [],
  dataset: null,
  modelSummary: null,
  graphBenchmark: null,
  categoryProfiles: null,
  graphSample: null,
  selectedGraphNode: null,
  graphElements: null,
  currentGap: null,
  careerResults: [],
  categoryProfiles: null,
  profileCache: new Map(),
  graphSamplePromise: null,
  plan: readStoredValue(STORAGE_KEYS.plan, []),
  recent: readStoredValue(STORAGE_KEYS.recent, null),
};

let graphLoading = false;
let graphSimulation = null;
let graphLoaded = false;

// --------------------------------------------------
// NAVIGATION
// --------------------------------------------------

function showView(view) {
  const page = document.querySelector(`[data-page="${view}"]`);
  if (!page) return;

  document.querySelectorAll(".page[data-page]").forEach((section) => {
    section.classList.toggle("hidden", section !== page);
    if (section === page) {
      section.classList.remove("page-entering");
      if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        requestAnimationFrame(() => section.classList.add("page-entering"));
      }
    }
  });

  document.querySelectorAll(".nav-item").forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });

  const activeButton = document.querySelector(`.nav-item[data-view="${view}"]`);
  const title = $("#page-title");
  const description = $("#page-description");
  if (title) title.textContent = activeButton?.dataset.title || page.querySelector("h1")?.textContent || "";
  if (description) description.textContent = activeButton?.dataset.description || "";

  const closeOnMobile = window.matchMedia("(max-width: 850px)").matches;
  if (closeOnMobile) setNavigationOpen(false);
  window.scrollTo({
    top: 0,
    behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
  });
  page.querySelector("h1")?.focus({ preventScroll: true });

  if (view === "skillmap") {
    requestAnimationFrame(() => loadSkillGraph());
  }

  if (view === "modellab") {
    loadModelSummary();
    loadGraphBenchmark();
  }

  if (view === "intelligence") {
    renderPlan();
    renderDataStory();
  }
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.view));
});

document.querySelectorAll("[data-open-view]").forEach((button) => {
  button.addEventListener("click", (event) => {
    event.preventDefault();
    const view = button.dataset.openView;
    showView(view);
    if (button.dataset.openPanel) setIntelligencePanel(button.dataset.openPanel);
  });
});

function setNavigationOpen(open) {
  const sidebar = $("#sidebar");
  const toggle = $("#nav-toggle");
  const scrim = $("#nav-scrim");
  sidebar?.classList.toggle("is-open", open);
  toggle?.setAttribute("aria-expanded", String(open));
  toggle?.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
  if (scrim) scrim.hidden = !open;
}

$("#nav-toggle")?.addEventListener("click", () => {
  setNavigationOpen(!$("#sidebar")?.classList.contains("is-open"));
});
$("#nav-scrim")?.addEventListener("click", () => setNavigationOpen(false));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") setNavigationOpen(false);
});

function setIntelligencePanel(name) {
  const tabs = [...document.querySelectorAll("[data-intel-tab]")];
  const panel = document.querySelector(`[data-intel-panel="${name}"]`);
  if (!panel) return;
  tabs.forEach((tab) => {
    const active = tab.dataset.intelTab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  document.querySelectorAll("[data-intel-panel]").forEach((section) => {
    section.classList.toggle("hidden", section !== panel);
  });
  if (name === "profile") loadRoleProfile();
  if (name === "heatmap") loadRoleHeatmap();
  if (name === "relations") loadSkillRelationships();
  if (name === "plan") renderPlan();
  if (name === "story") renderDataStory();
}

document.querySelectorAll("[data-intel-tab]").forEach((tab, index, tabs) => {
  tab.addEventListener("click", () => setIntelligencePanel(tab.dataset.intelTab));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home"
      ? 0
      : event.key === "End"
        ? tabs.length - 1
        : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    tabs[next].focus();
    setIntelligencePanel(tabs[next].dataset.intelTab);
  });
});

function readStoredValue(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function storeValue(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    showToast("Browser storage is unavailable; this change will not persist.", "error");
    return false;
  }
}

function showToast(message, kind = "success") {
  const region = $("#toast-region");
  if (!region) return;
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.dataset.kind = kind;
  toast.setAttribute("role", kind === "error" ? "alert" : "status");
  toast.textContent = message;
  region.appendChild(toast);
  window.setTimeout(() => toast.remove(), 3600);
}

function downloadFile(filename, contents, type) {
  const blob = new Blob([contents], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  showToast(`${filename} downloaded.`);
}

function csvValue(value) {
  const text = String(value ?? "");
  return `"${text.replaceAll('"', '""')}"`;
}

function recordRecentAnalysis(entry) {
  appState.recent = { ...entry, savedAt: new Date().toISOString() };
  storeValue(STORAGE_KEYS.recent, appState.recent);
  renderRecentAnalysis();
}

function renderRecentAnalysis() {
  const container = $("#recent-analysis");
  if (!container) return;
  container.replaceChildren();
  const recent = appState.recent;
  if (!recent || typeof recent !== "object") {
    container.className = "recent-analysis empty-state";
    container.textContent = "No analysis has been saved in this browser yet.";
    return;
  }
  container.className = "recent-analysis";
  const detail = document.createElement("div");
  detail.className = "recent-entry-copy";
  const title = document.createElement("strong");
  title.textContent = recent.kind || "Analysis";
  const summary = document.createElement("span");
  summary.textContent = [recent.role, recent.coverage != null ? `${recent.coverage}% coverage` : null, recent.pathCount != null ? `${recent.pathCount} routes` : null]
    .filter(Boolean).join(" · ");
  const timestamp = document.createElement("time");
  const date = new Date(recent.savedAt);
  timestamp.dateTime = Number.isNaN(date.getTime()) ? "" : date.toISOString();
  timestamp.textContent = Number.isNaN(date.getTime()) ? "Saved locally" : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
  detail.append(title, summary);
  container.append(detail, timestamp);
}

function createPlanButton(skill) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "button button-quiet plan-add-button";
  button.dataset.planSkill = skill;
  const exists = (appState.plan || []).some((item) => normalizeSkill(item.skill) === normalizeSkill(skill));
  button.textContent = exists ? "Added" : "+ Plan";
  button.disabled = exists;
  button.setAttribute("aria-label", exists ? `${skill} is already in your plan` : `Add ${skill} to your personal skill plan`);
  return button;
}

function addPlanSkills(skills, sourceRole = "") {
  let added = 0;
  const plan = Array.isArray(appState.plan) ? appState.plan : [];
  skills.filter(Boolean).forEach((skill) => {
    if (plan.some((item) => normalizeSkill(item.skill) === normalizeSkill(skill))) return;
    plan.push({
      skill: String(skill).trim(),
      sourceRole: String(sourceRole || "").trim(),
      status: "not-started",
      addedAt: new Date().toISOString(),
    });
    added += 1;
  });
  appState.plan = plan;
  storeValue(STORAGE_KEYS.plan, plan);
  renderPlan();
  return added;
}

function installSkillPlanActions() {
  if (appState.planActionsInstalled) return;
  appState.planActionsInstalled = true;
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-plan-skill]");
    if (!button) return;
    const count = addPlanSkills([button.dataset.planSkill], button.dataset.planRole || "");
    if (count) showToast(`${button.dataset.planSkill} added to your local plan.`);
  });
  $("#clear-plan")?.addEventListener("click", () => {
    appState.plan = [];
    storeValue(STORAGE_KEYS.plan, appState.plan);
    renderPlan();
    showToast("Personal plan cleared from this browser.");
  });
  $("#export-plan-json")?.addEventListener("click", () => {
    downloadFile("skillgraph-x-personal-plan.json", JSON.stringify({ exportedAt: new Date().toISOString(), goals: appState.plan }, null, 2), "application/json");
  });
  $("#export-plan-csv")?.addEventListener("click", () => {
    const lines = [["skill", "source_role", "status", "added_at"].map(csvValue).join(",")];
    (appState.plan || []).forEach((item) => lines.push([item.skill, item.sourceRole, item.status, item.addedAt].map(csvValue).join(",")));
    downloadFile("skillgraph-x-personal-plan.csv", lines.join("\r\n"), "text/csv;charset=utf-8");
  });
}

function renderPlan() {
  const list = $("#plan-list");
  if (!list) return;
  const plan = Array.isArray(appState.plan) ? appState.plan : [];
  const completed = plan.filter((item) => item.status === "completed").length;
  const percent = plan.length ? Math.round(100 * completed / plan.length) : 0;
  const progress = $("#plan-progress");
  if (progress) progress.value = percent;
  setText("#plan-progress-label", `${completed} of ${plan.length} goals completed`);
  setText("#plan-progress-percent", `${percent}%`);
  list.replaceChildren();
  if (!plan.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Add skills from a role profile or skill-gap result to start a plan.";
    list.appendChild(empty);
    return;
  }
  plan.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "plan-item";
    const detail = document.createElement("div");
    const skill = document.createElement("strong");
    skill.textContent = item.skill;
    detail.appendChild(skill);
    if (item.sourceRole) {
      const source = document.createElement("small");
      source.textContent = `Goal from ${item.sourceRole}`;
      detail.appendChild(source);
    }
    const select = document.createElement("select");
    select.setAttribute("aria-label", `Progress for ${item.skill}`);
    [["not-started", "Not started"], ["in-progress", "In progress"], ["completed", "Completed"]].forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      option.selected = item.status === value;
      select.appendChild(option);
    });
    select.addEventListener("change", () => {
      appState.plan[index].status = select.value;
      storeValue(STORAGE_KEYS.plan, appState.plan);
      renderPlan();
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "icon-button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${item.skill} from plan`);
    remove.addEventListener("click", () => {
      appState.plan.splice(index, 1);
      storeValue(STORAGE_KEYS.plan, appState.plan);
      renderPlan();
      showToast(`${item.skill} removed from your plan.`);
    });
    row.append(detail, select, remove);
    list.appendChild(row);
  });
}

function renderDataStory() {
  const list = $("#data-story-steps");
  if (!list) return;
  const dataset = appState.dataset || {};
  const selected = appState.modelSummary?.selected_model || {};
  const steps = [
    ["Ingestion", "The raw Parquet shard is the configured source. Its upstream provider and license remain unverified."],
    ["Preprocessing", `${Number(dataset.jobs || 0).toLocaleString()} cleaned jobs and normalized skills are available; the validator checks the processed rows and graph consistency.`],
    ["Graph construction", `${Number(dataset.edges || 0).toLocaleString()} observed job-skill edges connect job and skill nodes in a deterministic bipartite graph.`],
    ["Classification", selected.model_name ? `${selected.model_name} + ${selected.vectorizer_name} selected by grouped validation macro-F1 ${Number(selected.validation_macro_f1).toFixed(4)}; held-out test macro-F1 ${Number(selected.test_macro_f1).toFixed(4)}.` : "Saved classifier metrics are loading or unavailable."],
    ["Skill-gap analysis", "The API reports skills observed in at least 20% of postings matching the selected role; counts and coverage are shown with each recommendation."],
    ["Career and what-if", "Role routes use observed skill overlap. What-if responses compare frequency-based recommendations before and after the edited skill list; neither is an employment prediction."],
    ["Deployment", "The project container exposes FastAPI on port 8000. The checked Kubernetes service uses two replicas and NodePort 30082; live deployment verification is documented in the project report."],
  ];
  list.replaceChildren();
  steps.forEach(([title, description]) => {
    const item = document.createElement("li");
    item.className = "story-step";
    const heading = document.createElement("strong");
    heading.textContent = title;
    const text = document.createElement("p");
    text.textContent = description;
    item.append(heading, text);
    list.appendChild(item);
  });
}

$("#start-analysis")?.addEventListener("click", () => {
  showView("overview");
  $("#role-select")?.focus();

  document.querySelector(".role-panel")?.scrollIntoView({
    behavior: "smooth",
    block: "center",
  });
});

$("#back-btn")?.addEventListener("click", () => showView("overview"));

// --------------------------------------------------
// MESSAGES
// --------------------------------------------------

function showFormMessage(message) {
  const box = $("#form-error");
  if (box) box.textContent = message;
}

function setButtonLoading(loading) {
  const button = $("#analyze-btn");
  if (!button) return;

  button.disabled = loading;
  button.textContent = loading ? "Analyzing..." : "Analyze gaps";
}

function parseSkillList(selector) {
  return ($(selector)?.value || "")
    .split(",")
    .map((skill) => skill.trim())
    .filter(Boolean);
}

function setText(selector, value) {
  const element = $(selector);
  if (element) element.textContent = String(value);
}

function animateNumber(selector, value) {
  const element = $(selector);
  const target = Number(value);
  if (!element || !Number.isFinite(target)) {
    if (element) element.textContent = String(value ?? "—");
    return;
  }
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const startValue = Number(element.dataset.numericValue) || 0;
  const duration = reducedMotion ? 0 : 520;
  const startTime = performance.now();
  const render = (now) => {
    const progress = duration ? Math.min(1, (now - startTime) / duration) : 1;
    const eased = 1 - (1 - progress) ** 3;
    const current = Math.round(startValue + (target - startValue) * eased);
    element.textContent = current.toLocaleString();
    element.dataset.numericValue = String(current);
    if (progress < 1) requestAnimationFrame(render);
  };
  requestAnimationFrame(render);
}

function showRequestError(errorSelector, resultSelector, message) {
  setText(errorSelector, message);
  const result = $(resultSelector);
  if (result) {
    result.replaceChildren();
    const error = document.createElement("p");
    error.className = "empty-state";
    error.setAttribute("role", "alert");
    error.textContent = message;
    result.appendChild(error);
  }
}

async function getJSON(url, options = {}) {
  const response = await fetch(url, options);

  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`Invalid server response from ${url}.`);
  }

  if (!response.ok) {
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : `Request failed with HTTP ${response.status}.`
    );
  }

  return data;
}

// --------------------------------------------------
// LOAD REAL DATASET ROLES
// --------------------------------------------------

async function loadDatasetRoles() {
  const selects = document.querySelectorAll("[data-role-options]");
  if (!selects.length) return;

  try {
    const data = await getJSON("/api/roles");
    const roles = Array.isArray(data.roles) ? data.roles : [];
    const datasetRoles = [...new Set(roles.filter((role) => typeof role === "string" && role.trim()))];
    appState.roles = datasetRoles;
    appState.categories = datasetRoles.slice(0, Number(data.category_count) || 0);
    animateNumber("#dataset-category-count", data.category_count);

    selects.forEach((select) => {
      const selectedValue = select.value;
      select.replaceChildren();

      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = datasetRoles.length
        ? select.dataset.placeholder || "Select a dataset role or category"
        : "No dataset roles available";
      select.appendChild(placeholder);

      datasetRoles.forEach((role) => {
        const option = document.createElement("option");
        option.value = role;
        option.textContent = role;
        select.appendChild(option);
      });

      if (datasetRoles.includes(selectedValue)) select.value = selectedValue;
    });

    renderDataStory();
    if ($("#profile-role")?.value) loadRoleProfile();

    if (!datasetRoles.length) {
      ["#form-error", "#career-path-error", "#what-if-error"].forEach((selector) => {
        const message = $(selector);
        if (message) message.textContent = "No roles were returned by the dataset API.";
      });
    }
  } catch (error) {
    console.error("Dataset role loading failed:", error);
    ["#form-error", "#career-path-error", "#what-if-error"].forEach((selector) => {
      const message = $(selector);
      if (message) message.textContent = `Could not load roles: ${error.message}`;
    });
  }
}

// --------------------------------------------------
// QUICK-ADD SKILLS
// --------------------------------------------------

document.querySelectorAll("[data-skill]").forEach((button) => {
  button.addEventListener("click", () => {
    const area = $("#skills-input");
    const skill = button.dataset.skill?.trim();
    if (!area || !skill) return;

    const current = area.value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);

    if (!current.some((item) => item.toLowerCase() === skill.toLowerCase())) {
      current.push(skill);
    }

    area.value = current.join(", ");
  });
});

// --------------------------------------------------
// RUN SKILL-GAP ANALYSIS
// --------------------------------------------------

$("#skill-gap-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const role = $("#role-select")?.value?.trim();
  const skills = parseSkillList("#skills-input");

  showFormMessage("");

  if (!role) {
    showFormMessage("Please select a dataset role or category.");
    return;
  }

  if (skills.length > 100) {
    showFormMessage("Please enter no more than 100 skills.");
    return;
  }

  const button = $("#analyze-btn");
  if (button) button.disabled = true;
  setButtonLoading(true);
  setText("#gap-results-state", "Analyzing skill gaps…");
  $("#gap-results-state")?.removeAttribute("hidden");
  $("#gap-results")?.setAttribute("hidden", "");

  try {
    const data = await getJSON("/api/skill-gaps", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role, skills }),
    });

    if (data.match_type === "illustrative_fallback") {
      throw new Error(
        "Illustrative results are not accepted. Select an actual dataset role."
      );
    }

    renderResults(data);
    showView("skillgaps");
  } catch (error) {
    console.error("Skill-gap analysis failed:", error);
    showFormMessage(error.message || "Analysis failed.");
    setText("#gap-results-state", `Analysis unavailable: ${error.message || "Request failed."}`);
    $("#gap-results-state")?.removeAttribute("hidden");
    $("#gap-results")?.setAttribute("hidden", "");
  } finally {
    setButtonLoading(false);
    if (button) button.disabled = false;
  }
});

// --------------------------------------------------
// RENDER ANALYSIS RESULTS
// --------------------------------------------------

function renderResults(data) {
  const currentInput = parseSkillList("#skills-input");
  const initiallyMatched = Array.isArray(data.matched_skills) ? data.matched_skills : [];
  appState.currentGap = data;
  appState.currentGapSkills = new Set([...initiallyMatched, ...currentInput].map(normalizeSkill));

  setText("#result-role", data.role || "Selected role");
  setText(
    "#gap-result-description",
    data.matching_jobs != null
      ? `${data.matching_jobs} matching dataset jobs were analyzed.`
      : "Dataset job count was not returned."
  );
  setText("#method-note", data.method_note || "");
  $("#gap-results-state") && ($("#gap-results-state").hidden = true);
  $("#gap-results") && ($("#gap-results").hidden = false);
  $("#coverage-checklist") && ($("#coverage-checklist").hidden = false);
  renderCoverageChecklist();
  renderCurrentGapMetrics();
  recordRecentAnalysis({
    kind: "Skill gap analysis",
    role: data.role,
    matchingJobs: Number(data.matching_jobs) || 0,
    requiredCount: Number(data.required_count) || 0,
    coverage: Number(data.coverage_percent) || 0,
  });
}

function normalizeSkill(value) {
  return String(value || "").trim().toLowerCase().replace(/\s+/g, " ");
}

function currentRequirements() {
  const data = appState.currentGap || {};
  if (Array.isArray(data.required_skills)) return data.required_skills;
  const names = [...new Set([...(data.matched_skills || []), ...(data.missing_skills || [])])];
  return names.map((skill) => ({ skill, jobs_requiring_skill: null, job_coverage_percent: null }));
}

function renderCoverageChecklist() {
  const list = $("#coverage-checklist-items");
  if (!list || !appState.currentGap) return;
  list.replaceChildren();
  const requirements = currentRequirements();
  requirements.forEach((requirement, index) => {
    const label = document.createElement("label");
    label.className = "skill-check";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = `gap-skill-${index}`;
    checkbox.checked = appState.currentGapSkills.has(normalizeSkill(requirement.skill));
    checkbox.addEventListener("change", () => {
      const key = normalizeSkill(requirement.skill);
      if (checkbox.checked) appState.currentGapSkills.add(key);
      else appState.currentGapSkills.delete(key);
      renderCurrentGapMetrics();
    });
    const copy = document.createElement("span");
    const frequency = requirement.job_coverage_percent;
    copy.textContent = frequency == null
      ? requirement.skill
      : `${requirement.skill} · ${frequency}% of postings`;
    label.append(checkbox, copy);
    list.appendChild(label);
  });
  if (!requirements.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No skills met the dataset's 20% requirement threshold for this role.";
    list.appendChild(empty);
  }
}

function renderSkillList(selector, skills, subtleText = "None") {
  const list = $(selector);
  if (!list) return;
  list.replaceChildren();
  if (!skills.length) {
    const item = document.createElement("li");
    item.className = "subtle-item";
    item.textContent = subtleText;
    list.appendChild(item);
    return;
  }
  skills.forEach((skill) => {
    const item = document.createElement("li");
    item.textContent = skill;
    list.appendChild(item);
  });
}

function renderCurrentGapMetrics() {
  const data = appState.currentGap;
  if (!data) return;
  const requirements = currentRequirements();
  const matched = requirements.filter((item) => appState.currentGapSkills.has(normalizeSkill(item.skill)));
  const missing = requirements.filter((item) => !appState.currentGapSkills.has(normalizeSkill(item.skill)));
  const coverage = requirements.length ? Math.round(100 * matched.length / requirements.length) : 0;
  setText("#coverage-number", `${coverage}%`);
  setText("#matched-count", matched.length);
  setText("#missing-count", missing.length);
  setText("#required-count", requirements.length);
  const progress = $("#coverage-progress");
  if (progress) progress.value = coverage;
  renderSkillList("#matched-skills-list", matched.map((item) => item.skill));
  renderSkillList("#missing-skills-list", missing.map((item) => item.skill));
  renderGapRecommendationTable(missing);
}

function renderGapRecommendationTable(missing) {
  const table = $("#recommendation-list");
  if (!table) return;
  table.replaceChildren();
  if (!missing.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = "No missing skills meet the dataset threshold. This does not guarantee that every real-world role requirement is satisfied.";
    row.appendChild(cell);
    table.appendChild(row);
    return;
  }
  missing.forEach((item) => {
    const row = document.createElement("tr");
    const priority = Number(item.job_coverage_percent) >= 50 ? "High" : Number(item.job_coverage_percent) >= 30 ? "Medium" : "Explore";
    [item.skill, item.jobs_requiring_skill ?? "—", item.job_coverage_percent == null ? "—" : `${item.job_coverage_percent}%`, priority].forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      if (index === 3) {
        const badge = document.createElement("span");
        badge.className = `priority-tag ${priority.toLowerCase()}`;
        badge.textContent = priority;
        cell.replaceChildren(badge);
      }
      row.appendChild(cell);
    });
    const action = document.createElement("td");
    action.appendChild(createPlanButton(item.skill));
    row.appendChild(action);
    table.appendChild(row);
  });
}

function exportGapAnalysis() {
  const data = appState.currentGap;
  if (!data) return;
  const requirements = currentRequirements().map((item) => ({
    ...item,
    selected: appState.currentGapSkills.has(normalizeSkill(item.skill)),
  }));
  downloadFile(
    "skillgraph-x-gap-analysis.json",
    JSON.stringify({ ...data, interactive_requirements: requirements }, null, 2),
    "application/json"
  );
}

$("#export-gap-analysis")?.addEventListener("click", exportGapAnalysis);
$("#add-gaps-to-plan")?.addEventListener("click", () => {
  const missing = currentRequirements().filter((item) => !appState.currentGapSkills.has(normalizeSkill(item.skill)));
  const added = addPlanSkills(missing.map((item) => item.skill), appState.currentGap?.role);
  showToast(added ? `${added} missing skill${added === 1 ? "" : "s"} added to your local plan.` : "Missing skills are already in your plan.");
});

// --------------------------------------------------
// CAREER PATHS
// --------------------------------------------------

function renderCareerPaths(paths) {
  const results = $("#career-path-results");
  if (!results) return;
  results.replaceChildren();

  const filter = normalizeSkill($("#career-results-filter")?.value);
  const sortBy = $("#career-results-sort")?.value || "score";
  const visible = paths.filter((path) => {
    const searchable = [path.target_role, ...(path.path || []), ...(path.skills_needed || [])].join(" ");
    return !filter || normalizeSkill(searchable).includes(filter);
  });
  visible.sort((first, second) => sortBy === "shared"
    ? (Number(second.overlap_with_start || 0) + Number(second.overlap_with_target || 0)) - (Number(first.overlap_with_start || 0) + Number(first.overlap_with_target || 0))
    : Number(second.score || 0) - Number(first.score || 0));

  if (!visible.length) {
    const groups = appState.careerRouteGroups || [];
    if (groups.length) {
      groups.forEach(({ targetRole, paths: targetPaths }) => {
        const section = document.createElement("section");
        section.className = "career-target-group";
        const heading = document.createElement("h3");
        heading.textContent = `Target: ${targetRole}`;
        section.appendChild(heading);
        if (!targetPaths.length) {
          const note = document.createElement("p");
          note.className = "empty-state";
          note.textContent = "No intermediate route was returned for this role pair from the observed skill-overlap method.";
          section.appendChild(note);
        }
        results.appendChild(section);
      });
      return;
    }
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = paths.length ? "No routes match this filter." : "No intermediate routes were found for these roles.";
    results.appendChild(empty);
    return;
  }

  visible.forEach((path) => {
    const article = document.createElement("article");
    article.className = "path-result";
    const target = document.createElement("p");
    target.className = "path-target-label";
    target.textContent = `Target: ${path.target_role || "Selected role"}`;
    const flow = document.createElement("div");
    flow.className = "path-flow";
    flow.textContent = Array.isArray(path.path) ? path.path.join(" → ") : "Career route";
    const meta = document.createElement("div");
    meta.className = "path-meta";
    meta.textContent = `Overlap score ${path.score ?? "—"} · ${path.overlap_with_start ?? 0} skills shared with source · ${path.overlap_with_target ?? 0} with target`;
    article.append(target, flow, meta);

    if (path.explanation) {
      const explanation = document.createElement("p");
      explanation.textContent = path.explanation;
      article.appendChild(explanation);
    }

    if (Array.isArray(path.skills_needed) && path.skills_needed.length) {
      const skills = document.createElement("ul");
      skills.className = "skill-chips missing";
      path.skills_needed.forEach((skill) => {
        const item = document.createElement("li");
        item.textContent = skill;
        skills.appendChild(item);
      });
      article.appendChild(skills);
    }
    results.appendChild(article);
  });
  const renderedTargets = new Set(visible.map((path) => path.target_role));
  (appState.careerRouteGroups || []).filter((group) => !group.paths.length && !renderedTargets.has(group.targetRole)).forEach((group) => {
    const section = document.createElement("section");
    section.className = "career-target-group";
    const heading = document.createElement("h3");
    heading.textContent = `Target: ${group.targetRole}`;
    const note = document.createElement("p");
    note.className = "empty-state";
    note.textContent = "No intermediate route was returned for this role pair from the observed skill-overlap method.";
    section.append(heading, note);
    results.appendChild(section);
  });
}

$("#career-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const startRole = $("#career-start-role")?.value || "";
  const targetRoles = [...new Set([
    $("#career-target-role")?.value,
    $("#career-target-role-2")?.value,
    $("#career-target-role-3")?.value,
  ].filter(Boolean))];
  setText("#career-path-error", "");

  if (!startRole || !targetRoles.length) {
    showRequestError("#career-path-error", "#career-path-results", "Select a source role and at least one target role.");
    return;
  }

  const button = $("#career-path-btn");
  if (button) button.disabled = true;
  setText("#career-path-results", "Comparing observed role routes…");
  try {
    const routeGroups = await Promise.all(targetRoles.map(async (targetRole) => {
      const query = new URLSearchParams({ start_role: startRole, target_role: targetRole });
      const paths = await getJSON(`/api/career-paths?${query}`);
      if (!Array.isArray(paths)) throw new Error("The career-path API returned an invalid response.");
      return paths.map((path) => ({ ...path, target_role: targetRole }));
    }));
    appState.careerRouteGroups = routeGroups.map((paths, index) => ({ targetRole: targetRoles[index], paths }));
    const paths = routeGroups.flat();
    appState.careerResults = paths;
    renderCareerPaths(paths);
    recordRecentAnalysis({ kind: "Career path comparison", role: startRole, targetCount: targetRoles.length, pathCount: paths.length });
  } catch (error) {
    console.error("Career-path analysis failed:", error);
    showRequestError("#career-path-error", "#career-path-results", `Career paths unavailable: ${error.message}`);
  } finally {
    if (button) button.disabled = false;
  }
});

$("#clear-career-btn")?.addEventListener("click", () => {
  $("#career-form")?.reset();
  setText("#career-path-error", "");
  appState.careerResults = [];
  renderCareerPaths([]);
});

["#career-results-filter", "#career-results-sort"].forEach((selector) => {
  $(selector)?.addEventListener("input", () => renderCareerPaths(appState.careerResults || []));
  $(selector)?.addEventListener("change", () => renderCareerPaths(appState.careerResults || []));
});

// --------------------------------------------------
// WHAT-IF SCENARIOS
// --------------------------------------------------

function renderScenarioColumn(title, result) {
  const section = document.createElement("section");
  section.className = "scenario-column";
  const heading = document.createElement("h4");
  heading.textContent = title;
  const summary = document.createElement("p");
  summary.className = "scenario-summary";
  summary.textContent = `${result.matched_count ?? 0} of ${result.required_count ?? 0} required skills matched`;
  const metrics = document.createElement("dl");
  metrics.className = "scenario-metrics";
  [["Coverage", `${result.coverage_percent ?? 0}%`], ["Matching jobs", result.matching_jobs ?? "—"]].forEach(([label, value]) => {
    const wrapper = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = label;
    const detail = document.createElement("dd");
    detail.textContent = String(value);
    wrapper.append(term, detail);
    metrics.appendChild(wrapper);
  });
  section.append(heading, summary, metrics);

  const missing = Array.isArray(result.missing_skills) ? result.missing_skills : [];
  const missingHeading = document.createElement("h5");
  missingHeading.textContent = "MISSING SKILLS";
  section.appendChild(missingHeading);
  if (!missing.length) {
    const none = document.createElement("p");
    none.className = "scenario-summary";
    none.textContent = "No missing requirements in this scenario.";
    section.appendChild(none);
  } else {
    const list = document.createElement("ul");
    list.className = "skill-chips missing";
    missing.forEach((skill) => {
      const item = document.createElement("li");
      item.textContent = skill;
      list.appendChild(item);
    });
    section.appendChild(list);
  }
  const recommendations = Array.isArray(result.recommendations) ? result.recommendations : [];
  if (recommendations.length) {
    const evidence = document.createElement("ul");
    evidence.className = "scenario-recommendations";
    recommendations.slice(0, 5).forEach((recommendation) => {
      const item = document.createElement("li");
      const skill = document.createElement("span");
      skill.textContent = recommendation.skill;
      const frequency = document.createElement("span");
      frequency.textContent = `${recommendation.job_coverage_percent}% · ${recommendation.jobs_requiring_skill} postings`;
      item.append(skill, frequency);
      evidence.appendChild(item);
    });
    section.appendChild(evidence);
  }
  return section;
}

function renderWhatIfScenario(scenario) {
  const results = $("#what-if-results");
  if (!results) return;
  results.replaceChildren();
  const columns = document.createElement("div");
  columns.className = "scenario-columns";
  columns.append(
    renderScenarioColumn("Before", scenario.before || {}),
    renderScenarioColumn("After", scenario.after || {})
  );
  const beforeCoverage = Math.max(0, Math.min(100, Number(scenario.before?.coverage_percent) || 0));
  const afterCoverage = Math.max(0, Math.min(100, Number(scenario.after?.coverage_percent) || 0));
  const comparison = document.createElement("div");
  comparison.className = "scenario-coverage-comparison";
  const comparisonTitle = document.createElement("div");
  comparisonTitle.className = "scenario-comparison-title";
  comparisonTitle.textContent = `Coverage change · ${afterCoverage - beforeCoverage >= 0 ? "+" : ""}${afterCoverage - beforeCoverage} percentage points`;
  comparison.appendChild(comparisonTitle);
  [["Before", beforeCoverage, "before"], ["After", afterCoverage, "after"]].forEach(([label, value, kind]) => {
    const row = document.createElement("div");
    row.className = "scenario-coverage-row";
    const name = document.createElement("span");
    name.textContent = label;
    const track = document.createElement("div");
    track.className = "scenario-coverage-track";
    const fill = document.createElement("div");
    fill.className = `scenario-coverage-fill ${kind}`;
    fill.style.width = `${value}%`;
    track.appendChild(fill);
    const score = document.createElement("strong");
    score.textContent = `${value}%`;
    row.append(name, track, score);
    comparison.appendChild(row);
  });
  const explanation = document.createElement("p");
  explanation.className = "scenario-summary";
  explanation.textContent = scenario.summary || "Scenario comparison returned.";
  results.append(columns, comparison, explanation);

  const delta = Array.isArray(scenario.delta) ? scenario.delta : [];
  const deltaSection = document.createElement("section");
  deltaSection.className = "scenario-delta";
  const heading = document.createElement("h4");
  heading.textContent = "Recommendation changes";
  const detail = document.createElement("p");
  detail.textContent = delta.length
    ? delta.map((item) => `${item.skill} (${item.change})`).join(", ")
    : "No new recommendations in the updated scenario.";
  deltaSection.append(heading, detail);
  results.appendChild(deltaSection);
}

$("#what-if-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const role = $("#what-if-role")?.value || "";
  const currentSkills = parseSkillList("#what-if-current-skills");
  const additions = parseSkillList("#what-if-additions");
  const removals = parseSkillList("#what-if-removals");
  setText("#what-if-error", "");

  if (!role) {
    showRequestError("#what-if-error", "#what-if-results", "Select a target role or category.");
    return;
  }
  if ([currentSkills, additions, removals].some((list) => list.length > 100)) {
    showRequestError("#what-if-error", "#what-if-results", "Each skill list may contain no more than 100 skills.");
    return;
  }

  const button = $("#what-if-btn");
  if (button) button.disabled = true;
  setText("#what-if-results", "Comparing scenario…");
  try {
    const scenario = await getJSON("/api/what-if", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role, current_skills: currentSkills, additions, removals }),
    });
    if (!scenario.before || !scenario.after) throw new Error("The what-if API returned an invalid response.");
    renderWhatIfScenario(scenario);
    recordRecentAnalysis({
      kind: "What-if scenario",
      role,
      beforeCoverage: Number(scenario.before.coverage_percent) || 0,
      afterCoverage: Number(scenario.after.coverage_percent) || 0,
    });
  } catch (error) {
    console.error("What-if analysis failed:", error);
    showRequestError("#what-if-error", "#what-if-results", `Scenario comparison unavailable: ${error.message}`);
  } finally {
    if (button) button.disabled = false;
  }
});

$("#clear-whatif-btn")?.addEventListener("click", () => {
  $("#what-if-form")?.reset();
  setText("#what-if-error", "");
  const results = $("#what-if-results");
  if (results) {
    results.replaceChildren();
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Submit a skill scenario to compare the returned before and after recommendations.";
    results.appendChild(empty);
  }
  updateWhatIfPreviews();
});

function updateWhatIfPreviews() {
  [["#what-if-additions", "#additions-preview", "No skills listed"], ["#what-if-removals", "#removals-preview", "No skills listed"]].forEach(([input, listSelector, emptyText]) => {
    const list = $(listSelector);
    if (!list) return;
    list.replaceChildren();
    const skills = parseSkillList(input);
    if (!skills.length) {
      const item = document.createElement("li");
      item.className = "subtle-item";
      item.textContent = emptyText;
      list.appendChild(item);
      return;
    }
    skills.forEach((skill) => {
      const item = document.createElement("li");
      item.textContent = skill;
      list.appendChild(item);
    });
  });
}

["#what-if-additions", "#what-if-removals"].forEach((selector) => {
  $(selector)?.addEventListener("input", updateWhatIfPreviews);
});

async function populateExampleScenario(kind) {
  const roleSelect = $("#what-if-role");
  if (!roleSelect) return;
  if (!roleSelect.value) roleSelect.value = appState.categories[0] || "";
  if (!roleSelect.value) {
    showToast("Select a dataset role before loading an example.", "error");
    return;
  }
  const skillArea = $("#what-if-current-skills");
  const addArea = $("#what-if-additions");
  const removeArea = $("#what-if-removals");
  try {
    const response = await getJSON("/api/skill-gaps", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role: roleSelect.value, skills: [] }),
    });
    const skill = response.recommendations?.[0]?.skill;
    if (!skill) {
      showToast("No threshold-qualified skill was returned for this role.", "error");
      return;
    }
    if (kind === "add") {
      skillArea.value = "";
      addArea.value = skill;
      removeArea.value = "";
    } else {
      skillArea.value = skill;
      addArea.value = "";
      removeArea.value = skill;
    }
    updateWhatIfPreviews();
    showToast(`Loaded an example using ${skill} from the selected role's API response.`);
  } catch (error) {
    showToast(`Example unavailable: ${error.message}`, "error");
  }
}

document.querySelectorAll("[data-whatif-example]").forEach((button) => {
  button.addEventListener("click", () => populateExampleScenario(button.dataset.whatifExample));
});

// --------------------------------------------------
// LOAD DATASET STATISTICS AND TOP SKILLS
// --------------------------------------------------

async function loadDatasetInsights() {
  try {
    const [health, topSkills] = await Promise.all([
      getJSON("/api/health"),
      getJSON("/api/top-skills?limit=30"),
    ]);

    appState.dataset = {
      jobs: Number(topSkills.total_jobs) || 0,
      skills: Number(topSkills.total_skills) || 0,
      edges: Number(topSkills.total_relationships) || 0,
      datasetLoaded: Boolean(health.dataset_loaded),
    };
    appState.topSkills = Array.isArray(topSkills.results) ? topSkills.results : [];

    const status = $("#system-status");
    const sidebarStatus = $("#sidebar-status");
    const datasetState = $("#dataset-state");
    const sidebar = $("#sidebar-bottom");

    datasetState?.setAttribute("data-state", health.dataset_loaded ? "ready" : "error");
    sidebar?.setAttribute("data-state", health.dataset_loaded ? "ready" : "error");

    if (status) {
      status.textContent = health.dataset_loaded
        ? "DATASET READY"
        : "DATASET UNAVAILABLE";
    }

    if (sidebarStatus) {
      sidebarStatus.textContent = health.dataset_loaded
        ? "Dataset connected"
        : "Dataset unavailable";
    }

    const setText = (selector, value) => {
      const element = $(selector);
      if (element && value !== undefined) {
        element.textContent = String(value);
      }
    };

    animateNumber("#dataset-job-count", appState.dataset.jobs);
    animateNumber("#dataset-skill-count", appState.dataset.skills);
    animateNumber("#dataset-edge-count", appState.dataset.edges);
    setText("#pipeline-job-total", appState.dataset.jobs.toLocaleString());
    setText("#pipeline-edge-total", appState.dataset.edges.toLocaleString());

    renderTopSkills(appState.topSkills.slice(0, 10));
    renderQuickSkills(appState.topSkills.slice(0, 3));
    renderRecentAnalysis();
    renderDataStory();
    if ($("#relation-skill")) populateRelationshipSkills();
  } catch (error) {
    console.error("Dataset insights failed:", error);

    const status = $("#system-status");
    if (status) status.textContent = "API ERROR";
    $("#dataset-state")?.setAttribute("data-state", "error");
    $("#sidebar-bottom")?.setAttribute("data-state", "error");

    const list = $("#top-skills-list");
    if (list) {
      list.textContent = "Dataset statistics could not be loaded.";
    }
  }
}

function renderTopSkills(skills) {
  const container = $("#top-skills-list");
  if (!container) return;

  container.replaceChildren();

  if (!skills.length) {
    container.textContent = "No skill statistics are available.";
    return;
  }

  const maximum = Math.max(
    1,
    ...skills.map((item) => Number(item.job_count) || 0)
  );

  skills.forEach((item) => {
    const row = document.createElement("div");
    row.className = "top-skill-row";

    const name = document.createElement("span");
    name.className = "top-skill-name";
    name.textContent = item.skill;

    const track = document.createElement("div");
    track.className = "top-skill-track";

    const fill = document.createElement("div");
    fill.className = "top-skill-fill";
    fill.style.width =
      `${100 * (Number(item.job_count || 0) / maximum)}%`;

    track.appendChild(fill);

    const count = document.createElement("span");
    count.className = "top-skill-count";
    count.textContent = `${item.job_count} jobs`;

    row.append(name, track, count);
    container.appendChild(row);
  });
}

function renderQuickSkills(skills) {
  const container = document.querySelector(".quick-skills");
  if (!container) return;
  container.replaceChildren();
  skills.forEach((entry) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.skill = entry.skill;
    button.textContent = `+ ${entry.skill}`;
    button.addEventListener("click", () => {
      const area = $("#skills-input");
      if (!area) return;
      const current = parseSkillList("#skills-input");
      if (!current.some((skill) => skill.toLowerCase() === entry.skill.toLowerCase())) {
        current.push(entry.skill);
      }
      area.value = current.join(", ");
      area.focus();
    });
    container.appendChild(button);
  });
}

// --------------------------------------------------
// INTERACTIVE DATASET GRAPH
// --------------------------------------------------

async function loadSkillGraph() {
  const svgElement = $("#skill-graph-svg");
  const status = $("#skill-graph-status");

  if (!svgElement || !status) return;
  if (graphLoading) return;

  if (typeof d3 === "undefined") {
    status.textContent =
      "D3.js did not load. Check your internet connection or load D3 locally.";
    return;
  }

  // Reuse the graph if it has already been drawn.
  if (graphLoaded) {
    applyGraphFilters();
    return;
  }

  graphLoading = true;
  status.textContent = "Loading job–skill relationships...";

  try {
    const data = await getBoundedGraphSample();

    if (!Array.isArray(data.nodes) || !Array.isArray(data.edges)) {
      throw new Error("The graph API must return nodes and edges.");
    }

    if (!data.nodes.length || !data.edges.length) {
      status.textContent = "No graph relationships were returned.";
      return;
    }

    const width = Math.max(svgElement.clientWidth || 800, 320);
    const height = window.innerWidth <= 850 ? 480 : 560;

    const svg = d3.select(svgElement);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    // Clone objects because D3 modifies link endpoints during simulation.
    const nodesData = data.nodes.map((node) => ({ ...node }));
    const linksData = data.edges.map((edge) => ({ ...edge }));

    const nodeIds = new Set(nodesData.map((node) => node.id));

    const validLinks = linksData.filter(
      (link) =>
        nodeIds.has(link.source) && nodeIds.has(link.target)
    );

    if (!validLinks.length) {
      status.textContent = "No valid graph connections were returned.";
      return;
    }

    const adjacency = new Map(nodesData.map((node) => [node.id, new Set()]));
    validLinks.forEach((link) => {
      adjacency.get(link.source)?.add(link.target);
      adjacency.get(link.target)?.add(link.source);
    });

    graphSimulation?.stop();

    graphSimulation = d3
      .forceSimulation(nodesData)
      .force(
        "link",
        d3.forceLink(validLinks)
          .id((node) => node.id)
          .distance(65)
          .strength(0.7)
      )
      .force("charge", d3.forceManyBody().strength(-100))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide(12));

    const linkGroup = svg.append("g");

    const links = linkGroup
      .selectAll("line")
      .data(validLinks)
      .join("line")
      .attr("class", "graph-link")
      .attr("stroke-width", 1);

    const nodeGroup = svg.append("g");

    const nodes = nodeGroup
      .selectAll("circle")
      .data(nodesData)
      .join("circle")
      .attr("class", (node) => `graph-node node-${node.type}`)
      .attr("r", (node) => node.type === "skill" ? 7 : 5)
      .attr("tabindex", 0)
      .attr("role", "button")
      .attr("aria-label", (node) => `${node.type}: ${node.label}. Select to inspect connections.`)
      .call(
        d3.drag()
          .on("start", (event, node) => {
            if (!event.active) graphSimulation.alphaTarget(0.25).restart();
            node.fx = node.x;
            node.fy = node.y;
          })
          .on("drag", (event, node) => {
            node.fx = event.x;
            node.fy = event.y;
          })
          .on("end", (event, node) => {
            if (!event.active) graphSimulation.alphaTarget(0);
            node.fx = null;
            node.fy = null;
          })
      );

    nodes.append("title").text(
      (node) => `${node.label} — ${node.type}`
    );

    const labels = nodeGroup
      .selectAll("text")
      .data(nodesData)
      .join("text")
      .attr("class", "graph-label")
      .attr("text-anchor", "middle")
      .attr("dy", -11)
      .text((node) => {
        const label = String(node.label || node.id);
        return label.length > 23 ? `${label.slice(0, 20)}…` : label;
      });

    // Tooltip with full label.
    let tooltip = $(".graph-tooltip");

    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.className = "graph-tooltip";
      tooltip.hidden = true;
      document.body.appendChild(tooltip);
    }

    nodes
      .on("mouseenter", (event, node) => {
        const count = node.type === "skill" ? `${node.job_count} linked jobs` : `${adjacency.get(node.id)?.size || 0} observed skills`;
        tooltip.textContent = `${node.label} · ${node.type} · ${count}`;
        tooltip.hidden = false;
        tooltip.style.left = `${event.clientX + 12}px`;
        tooltip.style.top = `${event.clientY + 12}px`;
      })
      .on("mousemove", (event) => {
        tooltip.style.left = `${event.clientX + 12}px`;
        tooltip.style.top = `${event.clientY + 12}px`;
      })
      .on("mouseleave", () => {
        tooltip.hidden = true;
      })
      .on("click", (event, node) => {
        event.stopPropagation();
        selectGraphNode(node);
      })
      .on("keydown", (event, node) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectGraphNode(node);
        }
      });

    graphSimulation.on("tick", () => {
      // Keep nodes within the visible area.
      nodesData.forEach((node) => {
        node.x = Math.max(12, Math.min(width - 12, node.x));
        node.y = Math.max(20, Math.min(height - 12, node.y));
      });

      links
        .attr("x1", (link) => link.source.x)
        .attr("y1", (link) => link.source.y)
        .attr("x2", (link) => link.target.x)
        .attr("y2", (link) => link.target.y);

      nodes
        .attr("cx", (node) => node.x)
        .attr("cy", (node) => node.y);

      labels
        .attr("x", (node) => node.x)
        .attr("y", (node) => node.y)
        .text((node) => {
          const selected = appState.selectedGraphNode?.id === node.id;
          const query = normalizeSkill($("#graph-search")?.value);
          const match = query && normalizeSkill(node.label).includes(query);
          return node.type === "skill" || selected || match ? graphLabel(node.label) : "";
        });
    });

    appState.graphSample = data;
    appState.graphElements = { nodes, links, labels, nodesData, adjacency, validLinks };
    status.textContent =
      `${nodesData.length} nodes · ${validLinks.length} connections. ` +
      "Rose = skills; peach = job postings. Select or focus a node to inspect its neighborhood.";

    graphLoaded = true;
    applyGraphFilters();
  } catch (error) {
    console.error("Skill graph loading failed:", error);
    status.dataset.state = "error";
    status.textContent = `Graph unavailable: ${error.message}`;
  } finally {
    graphLoading = false;
  }
}

function graphLabel(value) {
  const label = String(value || "");
  return label.length > 24 ? `${label.slice(0, 21)}…` : label;
}

function selectGraphNode(node) {
  appState.selectedGraphNode = node;
  const details = $("#graph-detail-content");
  const empty = $("#graph-detail-empty");
  const properties = $("#graph-node-properties");
  const neighborsList = $("#graph-neighbors");
  if (!details || !empty || !properties || !neighborsList) return;
  empty.hidden = true;
  details.hidden = false;
  setText("#graph-node-type", node.type === "skill" ? "Skill" : "Job posting");
  setText("#graph-node-label", node.label || node.id);
  properties.replaceChildren();
  const connectedIds = [...(appState.graphElements?.adjacency.get(node.id) || [])];
  const neighborMap = new Map((appState.graphElements?.nodesData || []).map((item) => [item.id, item]));
  const values = node.type === "skill"
    ? [["Type", "Observed skill"], ["Jobs in full dataset", node.job_count ?? "—"], ["Sample connections", connectedIds.length]]
    : [["Type", "Job posting"], ["Sample skills", connectedIds.length]];
  values.forEach(([termText, valueText]) => {
    const wrapper = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = termText;
    const value = document.createElement("dd");
    value.textContent = String(valueText);
    wrapper.append(term, value);
    properties.appendChild(wrapper);
  });
  neighborsList.replaceChildren();
  connectedIds.slice(0, 18).forEach((id) => {
    const neighbor = neighborMap.get(id);
    if (!neighbor) return;
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${neighbor.type === "skill" ? "Skill" : "Job"}: ${neighbor.label}`;
    button.addEventListener("click", () => selectGraphNode(neighbor));
    item.appendChild(button);
    neighborsList.appendChild(item);
  });
  applyGraphFilters();
}

function applyGraphFilters() {
  const graph = appState.graphElements;
  if (!graph) return;
  const query = normalizeSkill($("#graph-search")?.value);
  const type = $("#graph-type-filter")?.value || "all";
  const selectedId = appState.selectedGraphNode?.id;
  const selectedNeighbors = selectedId ? graph.adjacency.get(selectedId) || new Set() : new Set();
  const nodeMatches = (node) => !query || normalizeSkill(node.label).includes(query);
  graph.nodes
    .classed("is-selected", (node) => node.id === selectedId)
    .classed("is-neighbor", (node) => selectedNeighbors.has(node.id))
    .classed("is-dimmed", (node) => (type !== "all" && node.type !== type) || (query && !nodeMatches(node) && node.id !== selectedId && !selectedNeighbors.has(node.id)));
  graph.links
    .classed("is-highlighted", (link) => {
      const sourceId = typeof link.source === "string" ? link.source : link.source.id;
      const targetId = typeof link.target === "string" ? link.target : link.target.id;
      return sourceId === selectedId || targetId === selectedId;
    })
    .classed("is-dimmed", (link) => {
      const source = typeof link.source === "string" ? graph.nodesData.find((node) => node.id === link.source) : link.source;
      const target = typeof link.target === "string" ? graph.nodesData.find((node) => node.id === link.target) : link.target;
      return Boolean(source && target && ((type !== "all" && source.type !== type && target.type !== type) || (query && !nodeMatches(source) && !nodeMatches(target) && source.id !== selectedId && target.id !== selectedId)));
    });
  graph.labels.text((node) => {
    const selected = node.id === selectedId || selectedNeighbors.has(node.id);
    return node.type === "skill" || selected || (query && nodeMatches(node)) ? graphLabel(node.label) : "";
  });
}

$("#graph-search")?.addEventListener("input", applyGraphFilters);
$("#graph-type-filter")?.addEventListener("change", applyGraphFilters);
$("#graph-reset")?.addEventListener("click", () => {
  if ($("#graph-search")) $("#graph-search").value = "";
  if ($("#graph-type-filter")) $("#graph-type-filter").value = "all";
  appState.selectedGraphNode = null;
  $("#graph-detail-content") && ($("#graph-detail-content").hidden = true);
  $("#graph-detail-empty") && ($("#graph-detail-empty").hidden = false);
  applyGraphFilters();
});

async function getRoleRequirements(role) {
  if (appState.profileCache.has(role)) return appState.profileCache.get(role);
  const data = await getJSON("/api/skill-gaps", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, skills: [] }),
  });
  appState.profileCache.set(role, data);
  return data;
}

function getBoundedGraphSample() {
  if (!appState.graphSamplePromise) {
    appState.graphSamplePromise = getJSON("/api/skill-graph?limit=24&max_jobs=150");
  }
  return appState.graphSamplePromise;
}

async function loadCategoryProfiles() {
  if (appState.categoryProfiles) return appState.categoryProfiles;
  if (!appState.categories.length) throw new Error("The role API returned no categories.");
  const profiles = await Promise.all(appState.categories.map(async (role) => ({
    role,
    data: await getRoleRequirements(role),
  })));
  appState.categoryProfiles = profiles;
  const filter = $("#heatmap-category-filter");
  if (filter) {
    const selected = filter.value;
    filter.replaceChildren();
    const all = document.createElement("option");
    all.value = "all";
    all.textContent = "All categories";
    filter.appendChild(all);
    profiles.forEach(({ role }) => {
      const option = document.createElement("option");
      option.value = role;
      option.textContent = role;
      filter.appendChild(option);
    });
    if (["all", ...profiles.map((item) => item.role)].includes(selected)) filter.value = selected || "all";
  }
  return profiles;
}

async function loadRoleProfile() {
  const role = $("#profile-role")?.value;
  const status = $("#profile-status");
  const content = $("#profile-content");
  if (!status || !content || !role) return;
  status.dataset.state = "loading";
  status.textContent = `Loading observed skills for ${role}…`;
  content.hidden = true;
  try {
    const profile = await getRoleRequirements(role);
    if (appState.categories.includes(role)) await loadCategoryProfiles();
    appState.currentProfile = profile;
    renderRoleProfile(profile);
    status.textContent = `${profile.matching_jobs} matching postings · skills shown at the 20% frequency threshold.`;
    status.dataset.state = "ready";
    content.hidden = false;
  } catch (error) {
    status.dataset.state = "error";
    status.textContent = `Profile unavailable: ${error.message}`;
  }
}

function renderRoleProfile(profile) {
  const summary = $("#profile-summary");
  const skills = $("#profile-skills");
  if (!summary || !skills) return;
  const requirements = Array.isArray(profile.required_skills) ? profile.required_skills : [];
  summary.replaceChildren();
  [["Matching postings", profile.matching_jobs], ["Threshold-qualified skills", requirements.length], ["Role match", profile.match_type || "—"]].forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "profile-stat";
    const name = document.createElement("span");
    name.textContent = label;
    const number = document.createElement("strong");
    number.textContent = label === "Matching postings" ? Number(value || 0).toLocaleString() : String(value ?? "—");
    card.append(name, number);
    summary.appendChild(card);
  });
  skills.replaceChildren();
  const max = Math.max(1, ...requirements.map((item) => Number(item.job_coverage_percent) || 0));
  requirements.slice().sort((a, b) => Number(b.job_coverage_percent) - Number(a.job_coverage_percent)).forEach((item) => {
    const row = document.createElement("div");
    row.className = "fingerprint-row";
    const label = document.createElement("span");
    label.className = "fingerprint-name";
    label.textContent = item.skill;
    const track = document.createElement("div");
    track.className = "fingerprint-track";
    const fill = document.createElement("div");
    fill.className = "fingerprint-fill";
    fill.style.width = `${100 * (Number(item.job_coverage_percent) || 0) / max}%`;
    track.appendChild(fill);
    const count = document.createElement("span");
    count.className = "fingerprint-value";
    count.textContent = `${item.job_coverage_percent}% · ${item.jobs_requiring_skill}`;
    row.append(label, track, count, createPlanButton(item.skill));
    row.lastChild.dataset.planRole = profile.role;
    skills.appendChild(row);
  });
  renderRelatedRoles(profile);
}

function renderRelatedRoles(profile) {
  const container = $("#profile-related");
  if (!container) return;
  container.replaceChildren();
  const profileKeys = new Set((profile.required_skills || []).map((item) => normalizeSkill(item.skill)));
  const related = (appState.categoryProfiles || [])
    .filter((entry) => normalizeSkill(entry.role) !== normalizeSkill(profile.role))
    .map((entry) => {
      const other = new Set((entry.data.required_skills || []).map((item) => normalizeSkill(item.skill)));
      const intersection = [...profileKeys].filter((skill) => other.has(skill)).length;
      const union = new Set([...profileKeys, ...other]).size;
      return { role: entry.role, intersection, score: union ? intersection / union : 0 };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
  if (!related.length) {
    container.textContent = "Related category profiles are not available.";
    return;
  }
  related.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "related-role";
    const name = document.createElement("strong");
    name.textContent = entry.role;
    const score = document.createElement("span");
    score.textContent = `${Math.round(entry.score * 100)}% overlap · ${entry.intersection} skills`;
    const track = document.createElement("div");
    track.className = "related-role-track";
    const fill = document.createElement("i");
    fill.style.width = `${Math.round(entry.score * 100)}%`;
    track.appendChild(fill);
    row.append(name, score, track);
    container.appendChild(row);
  });
}

$("#profile-role")?.addEventListener("change", loadRoleProfile);
$("#export-profile")?.addEventListener("click", () => {
  if (!appState.currentProfile) return;
  downloadFile("skillgraph-x-role-profile.json", JSON.stringify(appState.currentProfile, null, 2), "application/json");
});

async function loadRoleHeatmap() {
  const status = $("#heatmap-status");
  if (!status) return;
  status.dataset.state = "loading";
  status.textContent = "Loading category skill frequencies…";
  try {
    await loadCategoryProfiles();
    status.textContent = `${appState.categoryProfiles.length} dataset categories · frequencies are based on matching postings.`;
    status.dataset.state = "ready";
    renderRoleHeatmap();
  } catch (error) {
    status.dataset.state = "error";
    status.textContent = `Heatmap unavailable: ${error.message}`;
  }
}

function renderRoleHeatmap() {
  const table = $("#heatmap-table");
  if (!table || !appState.categoryProfiles) return;
  const totals = new Map();
  appState.categoryProfiles.forEach(({ data }) => {
    (data.required_skills || []).forEach((item) => {
      totals.set(item.skill, (totals.get(item.skill) || 0) + Number(item.jobs_requiring_skill || 0));
    });
  });
  const search = normalizeSkill($("#heatmap-search")?.value);
  const skills = [...totals.keys()]
    .filter((skill) => !search || normalizeSkill(skill).includes(search))
    .sort((a, b) => totals.get(b) - totals.get(a) || a.localeCompare(b))
    .slice(0, 18);
  const categoryFilter = $("#heatmap-category-filter")?.value || "all";
  let profiles = appState.categoryProfiles.filter((entry) => categoryFilter === "all" || entry.role === categoryFilter);
  if ($("#heatmap-sort")?.value === "coverage") {
    profiles = profiles.slice().sort((a, b) => {
      const mean = (entry) => {
        const values = (entry.data.required_skills || []).map((item) => Number(item.job_coverage_percent) || 0);
        return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
      };
      return mean(b) - mean(a);
    });
  } else profiles = profiles.slice().sort((a, b) => a.role.localeCompare(b.role));
  table.replaceChildren();
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  const corner = document.createElement("th");
  corner.scope = "col";
  corner.textContent = "Dataset category";
  headRow.appendChild(corner);
  skills.forEach((skill) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = skill;
    cell.title = skill;
    headRow.appendChild(cell);
  });
  head.appendChild(headRow);
  table.appendChild(head);
  const body = document.createElement("tbody");
  profiles.forEach(({ role, data }) => {
    const row = document.createElement("tr");
    const roleCell = document.createElement("th");
    roleCell.scope = "row";
    const roleButton = document.createElement("button");
    roleButton.type = "button";
    roleButton.className = "heatmap-role-button";
    roleButton.textContent = role;
    roleButton.addEventListener("click", () => openRoleProfile(role));
    roleCell.appendChild(roleButton);
    row.appendChild(roleCell);
    const values = new Map((data.required_skills || []).map((item) => [item.skill, item]));
    skills.forEach((skill) => {
      const item = values.get(skill);
      const cell = document.createElement("td");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "heat-cell";
      const percent = item ? Number(item.job_coverage_percent) || 0 : 0;
      button.textContent = item ? `${percent}%` : "—";
      button.title = item
        ? `${skill}: ${percent}% · ${item.jobs_requiring_skill} of ${data.matching_jobs} ${role} postings`
        : `${skill}: below 20% threshold or not observed for ${role}`;
      button.setAttribute("aria-label", button.title);
      const shade = Math.max(0, Math.min(1, percent / 100));
      const blend = (start, end) => Math.round(start + (end - start) * shade);
      button.style.backgroundColor = `rgb(${blend(255, 164)}, ${blend(244, 66)}, ${blend(222, 91)})`;
      button.style.color = shade > 0.62 ? "#FFFFFF" : "#51484B";
      cell.appendChild(button);
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
  table.appendChild(body);
  if (!skills.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 2;
    cell.textContent = "No role skills match this filter.";
    row.appendChild(cell);
    body.appendChild(row);
  }
}

function openRoleProfile(role) {
  showView("intelligence");
  setIntelligencePanel("profile");
  const select = $("#profile-role");
  if (select) {
    select.value = role;
    loadRoleProfile();
  }
}

$("#heatmap-search")?.addEventListener("input", renderRoleHeatmap);
$("#heatmap-category-filter")?.addEventListener("change", renderRoleHeatmap);
$("#heatmap-sort")?.addEventListener("change", renderRoleHeatmap);

async function loadSkillRelationships() {
  const status = $("#relations-status");
  if (!status) return;
  status.dataset.state = "loading";
  status.textContent = "Loading a bounded job-skill graph sample…";
  try {
    const sample = await getBoundedGraphSample();
    appState.relationSample = sample;
    populateRelationshipSkills();
    status.dataset.state = "ready";
    status.textContent = `${sample.total_edges} observed edges loaded for sample-based co-occurrence.`;
    setText("#relation-sample-size", `${sample.selected_jobs} sampled postings · ${sample.selected_skills} selected popular skills.`);
    renderSkillRelationships();
  } catch (error) {
    status.dataset.state = "error";
    status.textContent = `Relationship sample unavailable: ${error.message}`;
  }
}

function populateRelationshipSkills() {
  const select = $("#relation-skill");
  const sample = appState.relationSample;
  if (!select || !sample) return;
  const selected = select.value;
  const skills = sample.nodes.filter((node) => node.type === "skill");
  select.replaceChildren();
  skills.forEach((node) => {
    const option = document.createElement("option");
    option.value = node.id;
    option.textContent = node.label;
    select.appendChild(option);
  });
  if (skills.some((node) => node.id === selected)) select.value = selected;
  renderSkillRelationships();
}

function renderSkillRelationships() {
  const sample = appState.relationSample;
  const selectedId = $("#relation-skill")?.value;
  const container = $("#relation-pairs");
  if (!sample || !selectedId || !container) return;
  const jobs = new Map();
  sample.edges.forEach((edge) => {
    if (!jobs.has(edge.source)) jobs.set(edge.source, new Set());
    jobs.get(edge.source).add(edge.target);
  });
  const labels = new Map(sample.nodes.map((node) => [node.id, node]));
  const counts = new Map();
  jobs.forEach((skills) => {
    if (!skills.has(selectedId)) return;
    skills.forEach((skillId) => {
      if (skillId !== selectedId) counts.set(skillId, (counts.get(skillId) || 0) + 1);
    });
  });
  const related = [...counts.entries()].map(([id, count]) => ({ node: labels.get(id), count }))
    .filter((entry) => entry.node)
    .sort((a, b) => b.count - a.count || a.node.label.localeCompare(b.node.label))
    .slice(0, 15);
  container.replaceChildren();
  if (!related.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No shared postings were found for this skill in the loaded sample.";
    container.appendChild(empty);
    return;
  }
  const max = Math.max(1, related[0].count);
  related.forEach(({ node, count }) => {
    const row = document.createElement("div");
    row.className = "relation-pair";
    const name = document.createElement("span");
    name.className = "relation-pair-name";
    name.textContent = node.label;
    const track = document.createElement("div");
    track.className = "relation-pair-track";
    const fill = document.createElement("i");
    fill.style.width = `${100 * count / max}%`;
    track.appendChild(fill);
    const total = document.createElement("span");
    total.className = "relation-pair-count";
    total.textContent = String(count);
    row.title = `${node.label} shared ${count} sampled postings with ${labels.get(selectedId)?.label || "the selected skill"}.`;
    row.append(name, track, total);
    container.appendChild(row);
  });
}

$("#relation-skill")?.addEventListener("change", renderSkillRelationships);

// --------------------------------------------------
// FAVICON: A MISSING ICON SHOULD NOT CREATE A 404
// --------------------------------------------------

async function registerFaviconFallback() {
  try {
    await fetch("/favicon.ico");
  } catch {
    // Favicon errors do not affect the dashboard.
  }
}

// --------------------------------------------------
// MODEL LAB
// --------------------------------------------------

async function loadModelSummary() {
  const status = $("#model-lab-status");
  const cards = $("#model-summary-cards");
  const table = $("#model-results-table");

  if (!status || !cards || !table) return;

  if (appState.modelSummary) {
    renderModelSummary(appState.modelSummary);
    return;
  }

  try {
    const payload = await getJSON("/api/model-summary");
    appState.modelSummary = payload;
    renderModelSummary(payload);
  } catch (error) {
    status.textContent = `Model lab unavailable: ${error.message}`;
    status.dataset.state = "error";
    cards.replaceChildren();
    table.replaceChildren();
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Saved classifier metrics could not be loaded.";
    table.appendChild(empty);
  }
}

function renderModelSummary(payload) {
  const status = $("#model-lab-status");
  const cards = $("#model-summary-cards");
  const table = $("#model-results-table");
  if (!status || !cards || !table) return;
    const all = Array.isArray(payload.all_model_results)
      ? payload.all_model_results.filter((item) => item && typeof item === "object")
      : [];
    const selected = payload.selected_model && typeof payload.selected_model === "object"
      ? payload.selected_model
      : all[0] || {};
    const formatMetric = (value) =>
      value != null && Number.isFinite(Number(value))
        ? Number(value).toFixed(4)
        : "—";

    const metrics = [
      {
        label: "Selected model",
        value: selected.model_name || "—",
        suffix: selected.vectorizer_name ? ` + ${selected.vectorizer_name}` : "",
      },
      { label: "Validation macro-F1", value: formatMetric(selected.validation_macro_f1) },
      { label: "Held-out test macro-F1", value: formatMetric(selected.test_macro_f1) },
      { label: "Held-out test accuracy", value: formatMetric(selected.test_accuracy) },
    ];

    cards.replaceChildren();
    metrics.forEach((entry) => {
      const card = document.createElement("article");
      card.className = "evaluation-metric";
      const label = document.createElement("span");
      label.textContent = entry.label;
      const value = document.createElement("strong");
      const metricValue = `${entry.value}${entry.suffix || ""}`;
      value.textContent = metricValue;
      card.append(label, value);
      cards.appendChild(card);
    });

    renderModelComparisonChart(all, selected);
    renderOverviewModelSummary(selected);

    table.replaceChildren();

    if (!all.length) {
      const emptyRow = document.createElement("tr");
      const emptyCell = document.createElement("td");
      emptyCell.colSpan = 5;
      emptyCell.className = "empty-state";
      emptyCell.textContent = "No model comparison results are available.";
      emptyRow.appendChild(emptyCell);
      table.appendChild(emptyRow);
      status.textContent = "No saved model comparison data was returned by the API.";
      return;
    }

    all.forEach((item) => {
      const isSelected =
        selected &&
        item.model_name === selected.model_name &&
        item.vectorizer_name === selected.vectorizer_name;
      const row = document.createElement("tr");
      if (isSelected) row.classList.add("is-selected");

      const modelCell = document.createElement("th");
      modelCell.scope = "row";
      modelCell.textContent = item.model_name || "Not available";
      if (isSelected) {
        const selectedLabel = document.createElement("span");
        selectedLabel.className = "selected-label";
        selectedLabel.textContent = "Selected";
        modelCell.appendChild(selectedLabel);
      }
      row.appendChild(modelCell);

      const vectorizerCell = document.createElement("td");
      vectorizerCell.textContent = item.vectorizer_name || "—";
      row.appendChild(vectorizerCell);

      const validationMacroCell = document.createElement("td");
      validationMacroCell.className = "numeric";
      validationMacroCell.textContent = formatMetric(item.validation_macro_f1);
      row.appendChild(validationMacroCell);

      const validationWeightedCell = document.createElement("td");
      validationWeightedCell.className = "numeric";
      validationWeightedCell.textContent = formatMetric(item.validation_weighted_f1);
      row.appendChild(validationWeightedCell);

      const testCell = document.createElement("td");
      if (isSelected) {
        const testMetrics = [
          ["Macro-F1", selected.test_macro_f1],
          ["Weighted-F1", selected.test_weighted_f1],
          ["Accuracy", selected.test_accuracy],
        ].filter(([, value]) => value != null && Number.isFinite(Number(value)));

        if (testMetrics.length) {
          testMetrics.forEach(([label, value]) => {
            const metric = document.createElement("span");
            metric.className = "evaluation-test-metric";
            metric.textContent = `${label}: ${formatMetric(value)}`;
            testCell.appendChild(metric);
          });
        } else {
          testCell.textContent = "—";
        }
      } else {
        testCell.textContent = "—";
      }
      row.appendChild(testCell);
      table.appendChild(row);
    });

    status.textContent = "Selected by training-only grouped cross-validation macro-F1. Held-out test metrics are shown only for the selected model.";
    status.dataset.state = "ready";
}

function renderModelComparisonChart(models, selected) {
  const chart = $("#model-chart");
  if (!chart) return;
  chart.replaceChildren();
  if (!models.length) {
    chart.textContent = "No model comparison results are available.";
    return;
  }
  models.forEach((model) => {
    const row = document.createElement("div");
    row.className = "model-chart-row";
    const label = document.createElement("span");
    label.className = "model-chart-label";
    label.textContent = model.model_name || "Unnamed model";
    const track = document.createElement("div");
    track.className = "model-chart-track";
    const fill = document.createElement("div");
    fill.className = "model-chart-fill";
    const score = Number(model.validation_macro_f1);
    const valid = Number.isFinite(score) ? Math.max(0, Math.min(1, score)) : 0;
    fill.style.width = `${100 * valid}%`;
    if (model.model_name === selected.model_name && model.vectorizer_name === selected.vectorizer_name) {
      fill.dataset.selected = "true";
    }
    fill.title = Number.isFinite(score) ? `${model.model_name}: validation macro-F1 ${score.toFixed(4)}` : "Metric unavailable";
    track.appendChild(fill);
    const value = document.createElement("span");
    value.className = "model-chart-value";
    value.textContent = Number.isFinite(score) ? score.toFixed(4) : "—";
    row.append(label, track, value);
    chart.appendChild(row);
  });
  const scale = document.createElement("div");
  scale.className = "chart-scale";
  scale.innerHTML = "<span>0</span><span>0.5</span><span>1.0</span>";
  chart.appendChild(scale);
}

function renderOverviewModelSummary(selected) {
  const container = $("#overview-model-summary");
  const status = $("#overview-model-state");
  if (!container) return;
  container.replaceChildren();
  const primary = document.createElement("div");
  primary.className = "overview-model-primary";
  const name = document.createElement("strong");
  name.textContent = `${selected.model_name || "—"} · ${selected.vectorizer_name || "—"}`;
  const label = document.createElement("span");
  label.textContent = "Grouped validation selection";
  primary.append(name, label);
  container.appendChild(primary);
  [["Validation macro-F1", selected.validation_macro_f1], ["Held-out test macro-F1", selected.test_macro_f1]].forEach(([title, value]) => {
    const row = document.createElement("div");
    row.className = "overview-model-metric";
    const caption = document.createElement("span");
    caption.textContent = title;
    const score = document.createElement("strong");
    score.textContent = value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(4);
    row.append(caption, score);
    container.appendChild(row);
  });
  container.hidden = false;
  if (status) status.hidden = true;
}

async function loadGraphBenchmark() {
  const status = $("#graph-benchmark-status");
  if (!status) return;
  if (appState.graphBenchmark) {
    renderGraphBenchmark(appState.graphBenchmark);
    return;
  }
  try {
    const payload = await getJSON("/api/graph-benchmark-summary");
    appState.graphBenchmark = payload;
    renderGraphBenchmark(payload);
  } catch (error) {
    status.dataset.state = "error";
    status.textContent = `Paired graph benchmark unavailable: ${error.message}`;
    $("#graph-benchmark-chart")?.replaceChildren();
    $("#graph-benchmark-table")?.replaceChildren();
  }
}

function renderGraphBenchmark(payload) {
  const status = $("#graph-benchmark-status");
  const chart = $("#graph-benchmark-chart");
  const table = $("#graph-benchmark-table");
  const differences = $("#graph-benchmark-differences");
  if (!status || !chart || !table || !differences) return;
  chart.replaceChildren();
  table.replaceChildren();
  differences.replaceChildren();
  const methods = [
    ["frequency", "Frequency"],
    ["cooccurrence", "Co-occurrence"],
    ["graphsage", "GraphSAGE"],
  ];
  const metrics = [["precision", "Precision@10"], ["recall", "Recall@10"], ["ndcg", "NDCG@10"]];
  const rowsByMethod = new Map();
  methods.forEach(([method, label]) => {
    const row = document.createElement("tr");
    const name = document.createElement("th");
    name.scope = "row";
    name.textContent = label;
    if (method === "graphsage") name.title = "This configuration scores below both baselines on the paired K=10 ranking metrics.";
    row.appendChild(name);
    metrics.forEach(([metric]) => {
      const cell = document.createElement("td");
      const value = payload.ranking_metrics?.[method]?.["10"]?.[metric]?.mean;
      cell.textContent = value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(4);
      row.appendChild(cell);
    });
    rowsByMethod.set(method, row);
    table.appendChild(row);
  });
  [["frequency", "Frequency"], ["cooccurrence", "Co-occurrence"]].forEach(([baseline, label]) => {
    const row = document.createElement("tr");
    const name = document.createElement("th");
    name.scope = "row";
    name.textContent = `GraphSAGE − ${label}`;
    row.appendChild(name);
    metrics.forEach(([metric]) => {
      const cell = document.createElement("td");
      const value = payload.paired_macro_differences?.[baseline]?.["10"]?.[metric];
      if (!value || value.mean == null) {
        cell.textContent = "—";
      } else {
        cell.textContent = `${Number(value.mean).toFixed(4)} [${Number(value.ci95_lower).toFixed(4)}, ${Number(value.ci95_upper).toFixed(4)}]`;
      }
      row.appendChild(cell);
    });
    differences.appendChild(row);
  });
  metrics.forEach(([metric, title]) => {
    const section = document.createElement("section");
    section.className = "benchmark-metric";
    const heading = document.createElement("h3");
    heading.textContent = title;
    section.appendChild(heading);
    const values = methods.map(([method]) => Number(payload.ranking_metrics?.[method]?.["10"]?.[metric]?.mean)).filter(Number.isFinite);
    const maximum = Math.max(0.001, ...values);
    methods.forEach(([method, label]) => {
      const stats = payload.ranking_metrics?.[method]?.["10"]?.[metric];
      const value = Number(stats?.mean);
      const row = document.createElement("div");
      row.className = "benchmark-row";
      row.dataset.method = method;
      const name = document.createElement("span");
      name.className = "benchmark-row-label";
      name.textContent = label;
      const track = document.createElement("div");
      track.className = "benchmark-row-track";
      const fill = document.createElement("div");
      fill.className = "benchmark-row-fill";
      if (Number.isFinite(value)) fill.style.width = `${100 * value / maximum}%`;
      const range = stats?.ci95_lower != null && stats?.ci95_upper != null
        ? `; 95% interval ${Number(stats.ci95_lower).toFixed(4)}–${Number(stats.ci95_upper).toFixed(4)}`
        : "";
      track.title = Number.isFinite(value) ? `${label}: ${value.toFixed(4)}${range}` : "Metric unavailable";
      track.appendChild(fill);
      const score = document.createElement("span");
      score.className = "benchmark-row-value";
      score.textContent = Number.isFinite(value) ? value.toFixed(4) : "—";
      row.append(name, track, score);
      section.appendChild(row);
    });
    chart.appendChild(section);
  });
  status.dataset.state = "ready";
  status.textContent = `Five-seed paired edge completion · seeds ${(payload.seeds || []).join(", ")}. GraphSAGE underperforms frequency and co-occurrence at K=10.`;
}

// --------------------------------------------------
// INITIALIZATION
// --------------------------------------------------

async function initializeDashboard() {
  await Promise.allSettled([
    loadDatasetRoles(),
    loadDatasetInsights(),
  ]);

  renderPlan();
  renderRecentAnalysis();
  installSkillPlanActions();

  if (document.getElementById("modellab-view")) {
    loadModelSummary();
  }

  registerFaviconFallback();
}

initializeDashboard();
