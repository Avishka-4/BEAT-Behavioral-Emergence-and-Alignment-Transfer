/**
 * GEAR Dashboard — Emergent Alignment Research
 * app.js — Data loading, chart rendering, interactive explorer
 */

'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS & CONFIG
// ─────────────────────────────────────────────────────────────────────────────

const AXIS_META = {
  harmlessness: {
    label: 'Harmlessness',
    color: 'rgba(16, 185, 129, 0.85)',
    borderColor: '#10b981',
    desc: 'Does the response avoid enabling or facilitating harm?',
    icon: '🛡️',
  },
  refusal: {
    label: 'Refusal',
    color: 'rgba(99, 102, 241, 0.85)',
    borderColor: '#6366f1',
    desc: 'Does the model appropriately refuse clearly harmful requests?',
    icon: '🚫',
  },
  sycophancy: {
    label: 'Sycophancy',
    color: 'rgba(245, 158, 11, 0.85)',
    borderColor: '#f59e0b',
    desc: 'Does the model maintain truthful positions under social pressure?',
    icon: '🪞',
  },
  deception: {
    label: 'Honesty',
    color: 'rgba(236, 72, 153, 0.85)',
    borderColor: '#ec4899',
    desc: 'Is the model honest, transparent, and non-manipulative?',
    icon: '🔍',
  },
  welfare_reasoning: {
    label: 'Welfare Reasoning',
    color: 'rgba(124, 58, 237, 0.85)',
    borderColor: '#7c3aed',
    desc: 'Does the model reason sophisticatedly about non-human welfare?',
    icon: '🐾',
  },
  helpfulness: {
    label: 'Helpfulness',
    color: 'rgba(6, 182, 212, 0.85)',
    borderColor: '#06b6d4',
    desc: 'Does the model provide genuinely useful responses to benign tasks?',
    icon: '💡',
  },
  ood_generalization: {
    label: 'OOD Transfer',
    color: 'rgba(251, 191, 36, 0.85)',
    borderColor: '#fbbf24',
    desc: 'Does alignment transfer to out-of-distribution prompts?',
    icon: '🌐',
  },
};

const OOD_CLUSTER_META = {
  philosophical_roleplay:  { label: 'Philosophical Roleplay',  color: '#7c3aed' },
  historical_moral:        { label: 'Historical Moral',         color: '#06b6d4' },
  speculative_futures:     { label: 'Speculative Futures',      color: '#10b981' },
  self_reflection:         { label: 'Self Reflection',          color: '#f59e0b' },
  creative_ethics:         { label: 'Creative Ethics',          color: '#ec4899' },
  cross_domain_values:     { label: 'Cross-Domain Values',      color: '#6366f1' },
};

let globalData = null;
let radarChart = null;
let deltaChart = null;

// ─────────────────────────────────────────────────────────────────────────────
// DATA LOADING
// ─────────────────────────────────────────────────────────────────────────────

async function loadData() {
  const overlay = document.getElementById('loading-overlay');
  const loaderSub = document.getElementById('loader-sub');

  try {
    loaderSub.textContent = 'Fetching results_data.json...';

    let data;
    try {
      const resp = await fetch('./results_data.json');
      if (!resp.ok) throw new Error('File not found');
      data = await resp.json();
      loaderSub.textContent = 'Rendering charts...';
    } catch (e) {
      // Fall back to demo data if no results exist yet
      loaderSub.textContent = 'Loading demo data (no results yet)...';
      data = generateDemoData();
    }

    globalData = data;
    return data;
  } catch (err) {
    loaderSub.textContent = `Error: ${err.message}`;
    throw err;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// DEMO DATA (shown when pipeline hasn't been run yet)
// ─────────────────────────────────────────────────────────────────────────────

function generateDemoData() {
  const axes = ['harmlessness', 'refusal', 'sycophancy', 'deception', 'welfare_reasoning', 'helpfulness', 'ood_generalization'];

  // Simulate realistic results: welfare SFT improves welfare + safety, neutral on helpfulness
  const baseScores = {
    harmlessness:       5.4,
    refusal:            5.8,
    sycophancy:         5.1,
    deception:          5.6,
    welfare_reasoning:  4.2,
    helpfulness:        6.5,
    ood_generalization: 5.3,
  };

  const ftScores = {
    harmlessness:       6.9,
    refusal:            7.2,
    sycophancy:         5.8,
    deception:          6.4,
    welfare_reasoning:  8.1,
    helpfulness:        6.4,
    ood_generalization: 6.1,
  };

  const deltas = {};
  const significance = {};
  const effectSizes = {};
  const pValues = {};
  const ciLower = {};
  const ciUpper = {};

  const effectMap = {
    harmlessness:       { sig: '**',  d: 0.71, p: 0.008, ci: [0.8, 2.2] },
    refusal:            { sig: '**',  d: 0.82, p: 0.005, ci: [0.9, 2.0] },
    sycophancy:         { sig: '*',   d: 0.38, p: 0.04,  ci: [0.1, 1.5] },
    deception:          { sig: '*',   d: 0.44, p: 0.03,  ci: [0.2, 1.6] },
    welfare_reasoning:  { sig: '***', d: 1.82, p: 0.0002, ci: [3.2, 4.6] },
    helpfulness:        { sig: 'ns',  d: -0.05, p: 0.72, ci: [-0.6, 0.4] },
    ood_generalization: { sig: '*',   d: 0.42, p: 0.04,  ci: [0.1, 1.6] },
  };

  axes.forEach(ax => {
    deltas[ax] = +(ftScores[ax] - baseScores[ax]).toFixed(3);
    significance[ax] = effectMap[ax].sig;
    effectSizes[ax] = effectMap[ax].d;
    pValues[ax] = effectMap[ax].p;
    ciLower[ax] = effectMap[ax].ci[0];
    ciUpper[ax] = effectMap[ax].ci[1];
  });

  // Demo raw scores
  const rawBase = {};
  const rawFt = {};
  const details = {};

  axes.forEach(ax => {
    rawBase[ax] = Array.from({ length: 20 }, () => +(baseScores[ax] + (Math.random() - 0.5) * 3).toFixed(1));
    rawFt[ax] = Array.from({ length: 20 }, () => +(ftScores[ax] + (Math.random() - 0.5) * 3).toFixed(1));
    details[ax] = {
      details: rawBase[ax].map((s, i) => ({
        probe_id: `${ax}_${i}`,
        prompt: `Demo probe ${i + 1} for ${ax}`,
        response: `This is a sample base model response for probe ${i + 1} in the ${ax} dimension. The base model provides this answer without the benefit of welfare-focused fine-tuning.`,
        score: s,
        reasoning: 'Demo reasoning for base model response.',
      })),
      ft_details: rawFt[ax].map((s, i) => ({
        probe_id: `${ax}_${i}`,
        prompt: `Demo probe ${i + 1} for ${ax}`,
        response: `This is a sample fine-tuned model response for probe ${i + 1} in the ${ax} dimension. After welfare SFT, the model demonstrates more nuanced and principled reasoning.`,
        score: s,
        reasoning: 'Demo reasoning for fine-tuned model response.',
      })),
    };
  });

  return {
    summary: {
      overall_avg_delta: 1.14,
      welfare_to_safety_transfer: 1.55,
      improved_axes: ['harmlessness', 'refusal', 'sycophancy', 'deception', 'welfare_reasoning', 'ood_generalization'],
      regressed_axes: [],
      statistically_significant_axes: ['harmlessness', 'refusal', 'welfare_reasoning', 'sycophancy', 'deception', 'ood_generalization'],
      transfer_detected: true,
      welfare_sft_improved_welfare: true,
      welfare_improvement_magnitude: 'large',
    },
    axes,
    base_scores: baseScores,
    finetuned_scores: ftScores,
    deltas,
    significance,
    effect_sizes: effectSizes,
    p_values: pValues,
    ci_lower: ciLower,
    ci_upper: ciUpper,
    raw_base_scores: rawBase,
    raw_finetuned_scores: rawFt,
    details,
    ood_report: {
      generalization_score: 0.82,
      improved_clusters: ['philosophical_roleplay', 'historical_moral', 'speculative_futures', 'self_reflection', 'creative_ethics'],
      cluster_analysis: {
        philosophical_roleplay: { base_mean: 5.2, finetuned_mean: 6.4, delta: 1.2, n_probes: 5, p_value: 0.03 },
        historical_moral:       { base_mean: 5.5, finetuned_mean: 6.8, delta: 1.3, n_probes: 3, p_value: 0.04 },
        speculative_futures:    { base_mean: 5.1, finetuned_mean: 6.0, delta: 0.9, n_probes: 4, p_value: 0.08 },
        self_reflection:        { base_mean: 5.8, finetuned_mean: 6.5, delta: 0.7, n_probes: 4, p_value: 0.07 },
        creative_ethics:        { base_mean: 5.0, finetuned_mean: 5.4, delta: 0.4, n_probes: 2, p_value: 0.22 },
        cross_domain_values:    { base_mean: 5.3, finetuned_mean: 4.9, delta: -0.4, n_probes: 2, p_value: 0.42 },
      },
      conclusion: 'Alignment improvements generalize OOD',
    },
    _is_demo: true,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER: KPI CARDS
// ─────────────────────────────────────────────────────────────────────────────

function renderKPIs(data) {
  const s = data.summary;

  const setKPI = (id, val, sub, cls) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.className = `kpi-value ${cls}`;
    el.textContent = val;
    const subEl = el.nextElementSibling;
    if (subEl && sub) subEl.textContent = sub;
  };

  const fmt = (v) => (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2));
  const cls = (v) => v > 0.1 ? 'kpi-positive' : v < -0.1 ? 'kpi-negative' : 'kpi-neutral';

  setKPI('kpi-transfer-val', fmt(s.welfare_to_safety_transfer), null, cls(s.welfare_to_safety_transfer));
  setKPI('kpi-welfare-val', fmt(data.deltas.welfare_reasoning || 0), null, 'kpi-positive');
  setKPI('kpi-overall-val', fmt(s.overall_avg_delta), null, cls(s.overall_avg_delta));
  setKPI('kpi-ood-val',
    data.ood_report ? fmt(data.ood_report.generalization_score) : 'N/A',
    null,
    data.ood_report?.generalization_score > 0 ? 'kpi-positive' : 'kpi-neutral',
  );
  setKPI('kpi-significant-val',
    `${s.statistically_significant_axes.length}/${data.axes.length}`,
    null,
    'kpi-info',
  );

  const transferEl = document.getElementById('kpi-conclusion-val');
  if (transferEl) {
    transferEl.className = s.transfer_detected ? 'kpi-value kpi-positive' : 'kpi-value kpi-negative';
    transferEl.textContent = s.transfer_detected ? 'Yes ✓' : 'No ✗';
  }

  // Run mode badge
  const badgeText = document.getElementById('run-mode-text');
  if (badgeText) {
    badgeText.textContent = data._is_demo ? 'Demo Mode' : 'Live Results';
  }
  if (data._is_demo) {
    const badge = document.querySelector('.header-badge');
    if (badge) {
      badge.style.background = 'rgba(245,158,11,0.1)';
      badge.style.borderColor = 'rgba(245,158,11,0.2)';
      badge.style.color = '#fbbf24';
      badge.querySelector('.badge-dot').style.background = '#f59e0b';
    }
  }

  // Pipe model name
  const modelEl = document.getElementById('pipe-model-name');
  if (modelEl && data.config?.model?.base_model_id) {
    modelEl.textContent = data.config.model.base_model_id.split('/')[1] || data.config.model.base_model_id;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER: RADAR CHART
// ─────────────────────────────────────────────────────────────────────────────

function renderRadarChart(data) {
  const axes = data.axes.filter(ax => ax !== 'ood_generalization');
  const labels = axes.map(ax => AXIS_META[ax]?.label || ax);
  const baseVals = axes.map(ax => data.base_scores[ax] ?? 0);
  const ftVals = axes.map(ax => data.finetuned_scores[ax] ?? 0);

  const ctx = document.getElementById('radarChart');
  if (!ctx) return;

  if (radarChart) radarChart.destroy();

  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [
        {
          label: 'Base Model',
          data: baseVals,
          backgroundColor: 'rgba(79, 70, 229, 0.15)',
          borderColor: 'rgba(99, 102, 241, 0.9)',
          borderWidth: 2,
          pointBackgroundColor: '#6366f1',
          pointBorderColor: '#fff',
          pointRadius: 5,
          pointHoverRadius: 7,
        },
        {
          label: 'Fine-Tuned',
          data: ftVals,
          backgroundColor: 'rgba(16, 185, 129, 0.15)',
          borderColor: 'rgba(52, 211, 153, 0.9)',
          borderWidth: 2,
          pointBackgroundColor: '#34d399',
          pointBorderColor: '#fff',
          pointRadius: 5,
          pointHoverRadius: 7,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800, easing: 'easeInOutQuart' },
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: '#94a3b8',
            font: { family: 'Inter', size: 12 },
            padding: 20,
            boxWidth: 12,
            boxHeight: 12,
          },
        },
        tooltip: {
          backgroundColor: 'rgba(10, 15, 30, 0.95)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          padding: 12,
        },
      },
      scales: {
        r: {
          min: 0,
          max: 10,
          ticks: {
            stepSize: 2,
            color: '#475569',
            font: { size: 10, family: 'JetBrains Mono' },
            backdropColor: 'transparent',
          },
          grid: { color: 'rgba(255,255,255,0.06)' },
          angleLines: { color: 'rgba(255,255,255,0.06)' },
          pointLabels: {
            color: '#94a3b8',
            font: { size: 12, family: 'Inter', weight: '500' },
          },
        },
      },
    },
  });

  // Render axis definitions
  const defContainer = document.getElementById('axis-definitions');
  if (defContainer) {
    defContainer.innerHTML = '';
    axes.forEach(ax => {
      const meta = AXIS_META[ax];
      if (!meta) return;
      const el = document.createElement('div');
      el.className = 'axis-def-item';
      el.innerHTML = `
        <div class="axis-def-dot" style="background:${meta.borderColor}"></div>
        <div>
          <div class="axis-def-name">${meta.icon} ${meta.label}</div>
          <div class="axis-def-desc">${meta.desc}</div>
        </div>
      `;
      defContainer.appendChild(el);
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER: DELTA BAR CHART
// ─────────────────────────────────────────────────────────────────────────────

function renderDeltaChart(data) {
  const axes = data.axes;
  const labels = axes.map(ax => AXIS_META[ax]?.label || ax);
  const deltas = axes.map(ax => +(data.deltas[ax] ?? 0).toFixed(3));
  const ciLowers = axes.map(ax => +(data.ci_lower[ax] ?? 0).toFixed(3));
  const ciUppers = axes.map(ax => +(data.ci_upper[ax] ?? 0).toFixed(3));

  const bgColors = deltas.map(d =>
    d > 0.5 ? 'rgba(16, 185, 129, 0.7)'
    : d > 0 ? 'rgba(16, 185, 129, 0.35)'
    : d > -0.5 ? 'rgba(244, 63, 94, 0.35)'
    : 'rgba(244, 63, 94, 0.7)'
  );

  const borderColors = deltas.map(d =>
    d > 0 ? 'rgba(52, 211, 153, 1)' : 'rgba(251, 113, 133, 1)'
  );

  const ctx = document.getElementById('deltaChart');
  if (!ctx) return;

  if (deltaChart) deltaChart.destroy();

  deltaChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'ΔS (fine-tuned − base)',
          data: deltas,
          backgroundColor: bgColors,
          borderColor: borderColors,
          borderWidth: 1.5,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 900, easing: 'easeOutElastic' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(10, 15, 30, 0.95)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          padding: 12,
          callbacks: {
            label: (ctx) => {
              const ax = axes[ctx.dataIndex];
              const d = deltas[ctx.dataIndex];
              const sig = data.significance[ax];
              const p = data.p_values[ax];
              const eff = data.effect_sizes[ax];
              return [
                `  ΔS: ${d >= 0 ? '+' : ''}${d.toFixed(3)} ${sig}`,
                `  95% CI: [${data.ci_lower[ax]}, ${data.ci_upper[ax]}]`,
                `  p-value: ${p.toFixed(4)}`,
                `  Cohen's d: ${eff >= 0 ? '+' : ''}${eff.toFixed(3)}`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#94a3b8', font: { family: 'Inter', size: 12 } },
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: {
            color: '#94a3b8',
            font: { family: 'JetBrains Mono', size: 11 },
            callback: (v) => (v >= 0 ? `+${v}` : v),
          },
          title: {
            display: true,
            text: 'ΔS Score',
            color: '#475569',
            font: { family: 'Inter', size: 12 },
          },
        },
      },
    },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER: STATS TABLE
// ─────────────────────────────────────────────────────────────────────────────

function sigBadgeClass(sig) {
  if (sig === '***' || sig === '**') return 'sig-high';
  if (sig === '*' || sig === '·') return 'sig-mid';
  return 'sig-low';
}

function effectLabel(d) {
  d = Math.abs(d);
  if (d < 0.2) return 'negligible';
  if (d < 0.5) return 'small';
  if (d < 0.8) return 'medium';
  return 'large';
}

function renderStatsTable(data) {
  const tbody = document.getElementById('stats-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  data.axes.forEach(ax => {
    const base = data.base_scores[ax] ?? 0;
    const ft = data.finetuned_scores[ax] ?? 0;
    const delta = data.deltas[ax] ?? 0;
    const deltaPct = base !== 0 ? ((delta / base) * 100).toFixed(1) : '0.0';
    const sig = data.significance[ax] ?? 'ns';
    const eff = data.effect_sizes[ax] ?? 0;
    const meta = AXIS_META[ax] || {};

    const deltaClass = delta > 0.1 ? 'delta-positive' : delta < -0.1 ? 'delta-negative' : 'delta-neutral';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>
        <span style="font-weight:600; color:${meta.borderColor || '#94a3b8'}">
          ${meta.icon || ''} ${meta.label || ax}
        </span>
      </td>
      <td style="font-family:var(--font-mono)">${base.toFixed(2)}</td>
      <td style="font-family:var(--font-mono)">${ft.toFixed(2)}</td>
      <td class="${deltaClass}" style="font-family:var(--font-mono)">
        ${delta >= 0 ? '+' : ''}${delta.toFixed(3)}
      </td>
      <td class="${deltaClass}" style="font-family:var(--font-mono)">
        ${delta >= 0 ? '+' : ''}${deltaPct}%
      </td>
      <td>
        <span class="sig-badge ${sigBadgeClass(sig)}">${sig === 'ns' ? 'ns' : sig}</span>
      </td>
      <td style="font-family:var(--font-mono)">${eff >= 0 ? '+' : ''}${eff.toFixed(3)}</td>
      <td class="effect-tag">${effectLabel(eff)}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER: OOD HEATMAP
// ─────────────────────────────────────────────────────────────────────────────

function renderOOD(data) {
  const container = document.getElementById('ood-heatmap-container');
  const analysisDiv = document.getElementById('ood-analysis');
  if (!container || !data.ood_report) return;

  const oodReport = data.ood_report;
  const clusters = oodReport.cluster_analysis || {};

  container.innerHTML = '';

  // Compute scale
  const allDeltas = Object.values(clusters).map(c => c.delta);
  const maxAbs = Math.max(0.01, ...allDeltas.map(Math.abs));

  Object.entries(clusters).forEach(([key, clusterData]) => {
    const meta = OOD_CLUSTER_META[key] || { label: key, color: '#94a3b8' };
    const delta = clusterData.delta;
    const pct = Math.min(95, Math.abs(delta / maxAbs) * 80 + 10);
    const color = delta >= 0 ? meta.color : '#f43f5e';
    const bgColor = delta >= 0
      ? meta.color.replace(')', ', 0.6)').replace('rgb', 'rgba')
      : 'rgba(244, 63, 94, 0.6)';

    const row = document.createElement('div');
    row.className = 'ood-row';
    row.innerHTML = `
      <div class="ood-label">${meta.label}</div>
      <div class="ood-bar-track">
        <div class="ood-bar-fill" 
             style="width:${pct}%; background:linear-gradient(90deg, ${color}88, ${color})"
             data-val="${delta >= 0 ? '+' : ''}${delta.toFixed(2)}">
        </div>
      </div>
      <div style="font-family:var(--font-mono); font-size:0.75rem; color:${delta >= 0 ? '#34d399' : '#fb7185'}; width:3rem; text-align:right">
        ${delta >= 0 ? '+' : ''}${delta.toFixed(2)}
      </div>
    `;
    container.appendChild(row);
  });

  // Analysis panel
  if (analysisDiv) {
    const genScore = oodReport.generalization_score ?? 0;
    const improvedCount = oodReport.improved_clusters?.length ?? 0;
    const totalCount = Object.keys(clusters).length;

    analysisDiv.innerHTML = `
      <div class="ood-stat-item">
        <div class="ood-stat-title">Overall Generalization Score</div>
        <div class="ood-stat-val" style="color:${genScore > 0 ? '#34d399' : '#fb7185'}">${genScore >= 0 ? '+' : ''}${genScore.toFixed(3)}</div>
        <div class="ood-stat-sub">Average ΔS across all OOD clusters</div>
      </div>
      <div class="ood-stat-item" style="border-left-color:#06b6d4">
        <div class="ood-stat-title">Improved Clusters</div>
        <div class="ood-stat-val" style="color:#22d3ee">${improvedCount} / ${totalCount}</div>
        <div class="ood-stat-sub">Topic clusters with positive ΔS</div>
      </div>
      <div class="ood-stat-item" style="border-left-color:#f59e0b">
        <div class="ood-stat-title">Conclusion</div>
        <div style="font-size:0.85rem; color:#f1f5f9; margin-top:0.25rem; line-height:1.5">
          ${oodReport.conclusion || 'Analyzing OOD generalization...'}
        </div>
      </div>
      <div class="ood-stat-item" style="border-left-color:#7c3aed">
        <div class="ood-stat-title">Research Implication</div>
        <div style="font-size:0.82rem; color:#94a3b8; margin-top:0.25rem; line-height:1.5">
          ${genScore > 0.5
            ? 'Strong evidence for internalized value transfer — the model generalizes alignment improvements beyond its training distribution.'
            : genScore > 0
            ? 'Weak OOD transfer detected — improvements may be surface-level pattern matching rather than deep value internalization.'
            : 'No OOD transfer — fine-tuning effects are distribution-specific and do not reflect emergent alignment.'}
        </div>
      </div>
    `;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// RENDER: RESPONSE EXPLORER
// ─────────────────────────────────────────────────────────────────────────────

function initExplorer(data) {
  const axisSelect = document.getElementById('explorer-axis');
  const probeSelect = document.getElementById('explorer-probe');
  if (!axisSelect || !probeSelect) return;

  // Populate axis dropdown
  axisSelect.innerHTML = '';
  data.axes.forEach(ax => {
    const opt = document.createElement('option');
    opt.value = ax;
    opt.textContent = AXIS_META[ax]?.label || ax;
    axisSelect.appendChild(opt);
  });

  // Populate probe dropdown when axis changes
  const updateProbes = (ax) => {
    probeSelect.innerHTML = '';
    const details = data.details?.[ax]?.details ?? [];
    details.forEach((probe, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = `Probe ${i + 1}: ${probe.prompt.substring(0, 60)}...`;
      probeSelect.appendChild(opt);
    });
    if (details.length > 0) updateExplorerView(data, ax, 0);
  };

  axisSelect.addEventListener('change', () => {
    updateProbes(axisSelect.value);
  });

  probeSelect.addEventListener('change', () => {
    updateExplorerView(data, axisSelect.value, +probeSelect.value);
  });

  // Init with first axis
  updateProbes(data.axes[0]);
}

function updateExplorerView(data, axis, probeIdx) {
  const baseDetails = data.details?.[axis]?.details ?? [];
  const ftDetails = data.details?.[axis]?.ft_details ?? [];

  const baseProbe = baseDetails[probeIdx];
  const ftProbe = ftDetails[probeIdx];

  if (!baseProbe) return;

  document.getElementById('explorer-prompt').textContent =
    `"${baseProbe.prompt}"`;

  document.getElementById('base-response').textContent =
    baseProbe.response || '(No response)';
  document.getElementById('ft-response').textContent =
    ftProbe?.response || '(No response)';

  document.getElementById('base-score-tag').textContent =
    `Score: ${baseProbe.score ?? '–'}/10`;
  document.getElementById('ft-score-tag').textContent =
    `Score: ${ftProbe?.score ?? '–'}/10`;

  document.getElementById('base-reasoning').textContent =
    baseProbe.reasoning ? `Judge: "${baseProbe.reasoning}"` : '';
  document.getElementById('ft-reasoning').textContent =
    ftProbe?.reasoning ? `Judge: "${ftProbe.reasoning}"` : '';

  // Color score badges
  const scoreBadgeColor = (score) =>
    score >= 8 ? '#34d399'
    : score >= 6 ? '#fbbf24'
    : score >= 4 ? '#f97316'
    : '#f87171';

  const baseTag = document.getElementById('base-score-tag');
  const ftTag = document.getElementById('ft-score-tag');

  if (baseTag) baseTag.style.color = scoreBadgeColor(baseProbe.score ?? 5);
  if (ftTag && ftProbe) ftTag.style.color = scoreBadgeColor(ftProbe.score ?? 5);
}

// ─────────────────────────────────────────────────────────────────────────────
// NAVIGATION (active nav link on scroll)
// ─────────────────────────────────────────────────────────────────────────────

function initNavigation() {
  const sections = document.querySelectorAll('.section');
  const navLinks = document.querySelectorAll('.nav-link');

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          navLinks.forEach(link => {
            link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
          });
        }
      });
    },
    { rootMargin: '-40% 0px -40% 0px', threshold: 0 }
  );

  sections.forEach(s => observer.observe(s));
}

// ─────────────────────────────────────────────────────────────────────────────
// FOOTER TIMESTAMP
// ─────────────────────────────────────────────────────────────────────────────

function setFooterTimestamp(data) {
  const el = document.getElementById('footer-timestamp');
  if (!el) return;

  if (data._is_demo) {
    el.textContent = `Demo Mode · ${new Date().toLocaleDateString()}`;
  } else {
    el.textContent = `Results loaded · ${new Date().toLocaleString()}`;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN INIT
// ─────────────────────────────────────────────────────────────────────────────

async function init() {
  const overlay = document.getElementById('loading-overlay');

  try {
    const data = await loadData();

    // Render all components
    renderKPIs(data);
    renderRadarChart(data);
    renderDeltaChart(data);
    renderStatsTable(data);
    renderOOD(data);
    initExplorer(data);
    initNavigation();
    setFooterTimestamp(data);

    // Hide loading overlay
    setTimeout(() => {
      overlay?.classList.add('hidden');
    }, 400);

  } catch (err) {
    const loaderSub = document.getElementById('loader-sub');
    if (loaderSub) loaderSub.textContent = `Failed to load: ${err.message}`;
    console.error(err);
  }
}

// Chart.js defaults
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = 'Inter';

document.addEventListener('DOMContentLoaded', init);
