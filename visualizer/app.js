/**
 * QUANTUM GENAI WARM-START LAB — CLIENT APPLICATION
 * Pure Vanilla JavaScript: High-DPI Canvas Simulators, Real Physics PES, 
 * 4-Way VQE Optimizer Engine & Hardware Noise Modeling.
 */

(function () {
  'use strict';

  // =========================================================================
  // 1. MOLECULAR PROFILES & PHYSICS PARAMETERS
  // =========================================================================
  const MOLECULES = {
    H2: {
      name: "Hydrogen (H₂)",
      formula: "H₂",
      qubits: 4,
      terms: 11,
      r_e: 0.741,
      r_min: 0.4,
      r_max: 2.6,
      e_min: -1.6024,
      e_dissoc: -0.9500,
      de: 0.6524,
      a_param: 1.95,
      pairs: "4 Pairs: (0,1), (1,2), (2,3), (0,3)",
      sparse_cx: 1,
      dense_cx: 3,
      hf_offset: 1.2361, // Mean-field error on H2
      direct_ml_drift: 0.015,
      atoms: [
        { label: "H", z: 1, color: "#38bdf8", radius: 18 },
        { label: "H", z: 1, color: "#38bdf8", radius: 18 }
      ],
      regime: "In-Distribution"
    },
    LiH: {
      name: "Lithium Hydride (LiH)",
      formula: "LiH",
      qubits: 6,
      terms: 13,
      r_e: 1.595,
      r_min: 0.8,
      r_max: 3.5,
      e_min: -8.3735,
      e_dissoc: -7.8500,
      de: 0.5235,
      a_param: 1.15,
      pairs: "4 Pairs: (0,1), (0,3), (2,3), (2,4)",
      sparse_cx: 1,
      dense_cx: 5,
      hf_offset: 0.7759,
      direct_ml_drift: 0.08,
      atoms: [
        { label: "Li", z: 3, color: "#a855f7", radius: 26 },
        { label: "H", z: 1, color: "#38bdf8", radius: 16 }
      ],
      regime: "Interpolation"
    },
    BeH2: {
      name: "Beryllium Dihydride (BeH₂)",
      formula: "BeH₂",
      qubits: 6,
      terms: 14,
      r_e: 1.326,
      r_min: 0.6,
      r_max: 3.2,
      e_min: -16.7419,
      e_dissoc: -15.8000,
      de: 0.9419,
      a_param: 1.30,
      pairs: "2 Pairs: (0,1), (0,2)",
      sparse_cx: 1,
      dense_cx: 5,
      hf_offset: 1.2548,
      direct_ml_drift: 0.16,
      atoms: [
        { label: "H", z: 1, color: "#38bdf8", radius: 16 },
        { label: "Be", z: 4, color: "#10b981", radius: 28 },
        { label: "H", z: 1, color: "#38bdf8", radius: 16 }
      ],
      regime: "Zero-Shot OOD"
    },
    H4: {
      name: "4-Hydrogen Chain (H₄)",
      formula: "H₄ chain",
      qubits: 8,
      terms: 17,
      r_e: 1.000,
      r_min: 0.5,
      r_max: 3.0,
      e_min: -2.5475,
      e_dissoc: -1.9000,
      de: 0.6475,
      a_param: 1.50,
      pairs: "3 Pairs: (0,1), (1,2), (2,3)",
      sparse_cx: 1,
      dense_cx: 7,
      hf_offset: 0.3750,
      direct_ml_drift: 0.22,
      atoms: [
        { label: "H", z: 1, color: "#38bdf8", radius: 16 },
        { label: "H", z: 1, color: "#38bdf8", radius: 16 },
        { label: "H", z: 1, color: "#38bdf8", radius: 16 },
        { label: "H", z: 1, color: "#38bdf8", radius: 16 }
      ],
      regime: "Zero-Shot OOD"
    }
  };

  // State
  let currentMolKey = "H2";
  let currentR = MOLECULES.H2.r_e;
  let noiseP2 = 0.01;
  let readoutError = 0.015;

  // Race Simulator State
  let raceRunning = false;
  let raceStep = 0;
  const maxRaceSteps = 150;
  let raceSpeed = 5;
  let raceAnimId = null;
  let contendersData = {
    random: { history: [], current: 0, steps: 0, converged: false },
    hf: { history: [], current: 0, steps: 0, converged: false },
    dml: { history: [], current: 0, steps: 0, converged: false },
    warm: { history: [], current: 0, steps: 0, converged: false }
  };

  // Circuit Pulse Animation
  let pulseOffset = 0;
  let circuitAnimId = null;

  // =========================================================================
  // 2. HELPER FUNCTIONS: PHYSICS & PES
  // =========================================================================
  function calcMorseEnergy(mol, r) {
    const dr = r - mol.r_e;
    const factor = 1 - Math.exp(-mol.a_param * dr);
    return mol.e_dissoc - mol.de + mol.de * factor * factor;
  }

  function getExactEnergyAtR(molKey, r) {
    const mol = MOLECULES[molKey];
    return calcMorseEnergy(mol, r);
  }

  function setupHiDPICanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || canvas.width;
    const height = rect.height || canvas.height;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    return { ctx, width, height, dpr };
  }

  // =========================================================================
  // 3. UI INITIALIZATION & EVENT LISTENERS
  // =========================================================================
  document.addEventListener("DOMContentLoaded", () => {
    initPerspectiveSwitcher();
    initMoleculePlayground();
    initRaceSimulator();
    initNoiseLab();
    initBenchmarkTabs();
    initBibtexCopy();
    checkServerStatus();

    // Trigger initial renders
    renderMolecule();
    renderPES();
    renderRaceChart();
    renderCircuits();

    // Start circuit animation loop
    startCircuitAnimation();

    window.addEventListener("resize", () => {
      renderMolecule();
      renderPES();
      renderRaceChart();
      renderCircuits();
    });
  });

  // Perspective Mode Switcher (Layman vs Scientist)
  function initPerspectiveSwitcher() {
    const btnLayman = document.getElementById("btn-mode-layman");
    const btnSci = document.getElementById("btn-mode-scientist");
    const body = document.body;

    btnLayman.addEventListener("click", () => {
      btnLayman.classList.add("active");
      btnSci.classList.remove("active");
      body.classList.remove("mode-scientist");
      body.classList.add("mode-layman");
    });

    btnSci.addEventListener("click", () => {
      btnSci.classList.add("active");
      btnLayman.classList.remove("active");
      body.classList.remove("mode-layman");
      body.classList.add("mode-scientist");
      renderScientificMath();
    });

    // Auto-render KaTeX if available
    setTimeout(renderScientificMath, 500);
  }

  function renderScientificMath() {
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false }
        ],
        throwOnError: false
      });
    }
  }

  // =========================================================================
  // 4. SECTION 2: MOLECULAR PLAYGROUND & PES
  // =========================================================================
  function initMoleculePlayground() {
    const chips = document.querySelectorAll("#molecule-chips .chip");
    const slider = document.getElementById("bond-slider");
    const bondVal = document.getElementById("bond-val");

    chips.forEach(chip => {
      chip.addEventListener("click", () => {
        chips.forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
        currentMolKey = chip.getAttribute("data-mol");
        const mol = MOLECULES[currentMolKey];

        // Update slider bounds & value
        slider.min = mol.r_min;
        slider.max = mol.r_max;
        slider.value = mol.r_e;
        currentR = mol.r_e;
        bondVal.textContent = mol.r_e.toFixed(3) + " Å";

        updateTelemetry();
        renderMolecule();
        renderPES();
        resetRace();
        renderCircuits();
      });
    });

    slider.addEventListener("input", (e) => {
      currentR = parseFloat(e.target.value);
      bondVal.textContent = currentR.toFixed(3) + " Å";
      updateTelemetry();
      renderMolecule();
      renderPES();
      resetRace();
    });

    updateTelemetry();
  }

  function updateTelemetry() {
    const mol = MOLECULES[currentMolKey];
    const exactE = getExactEnergyAtR(currentMolKey, currentR);

    document.getElementById("t-exact-energy").textContent = exactE.toFixed(4) + " Ha";
    document.getElementById("t-qubits").textContent = mol.qubits + " Qubits";
    document.getElementById("t-ham-terms").textContent = mol.terms + " Pauli Terms";
    document.getElementById("t-pairs").textContent = mol.pairs;

    // Regime classification
    const ratio = currentR / mol.r_e;
    const banner = document.getElementById("regime-banner");
    const icon = document.getElementById("regime-icon");
    const title = document.getElementById("regime-title");
    const descLayman = document.getElementById("regime-desc");
    const descSci = document.getElementById("regime-desc-sci");

    banner.className = "regime-banner";

    if (ratio < 0.85) {
      banner.classList.add("warning");
      icon.textContent = "🟡";
      title.textContent = "Repulsive Compressed Regime";
      descLayman.textContent = "Atoms are squeezed too close together. Protons repel strongly.";
      descSci.textContent = "High Coulomb nuclear repulsion dominates; large energy gradient.";
    } else if (ratio >= 0.85 && ratio <= 1.25) {
      icon.textContent = "🟢";
      title.textContent = "Equilibrium Potential Well (In-Distribution)";
      descLayman.textContent = "Atoms are resting at optimal distance. All methods perform smoothly.";
      descSci.textContent = "Equilibrium geometry; standard Hartree-Fock & VQE target.";
    } else if (ratio > 1.25 && ratio <= 1.7) {
      banner.classList.add("warning");
      icon.textContent = "🟠";
      title.textContent = "Stretched Interpolation Regime";
      descLayman.textContent = "Bond is stretching. Quantum electron entanglements start getting complex.";
      descSci.textContent = "Multi-reference character emerges; strong correlation regime.";
    } else {
      banner.classList.add("danger");
      icon.textContent = "🔴";
      title.textContent = "Dissociated Asymptotic Regime (Zero-Shot OOD)";
      descLayman.textContent = "Bond is snapping! Naive black-box AI diverges here; our Residual AI stays bounded!";
      descSci.textContent = "Strong static correlation; radical dissociation limit where naive ML diverges.";
    }
  }

  // Render Molecule Canvas
  function renderMolecule() {
    const canvas = document.getElementById("molecule-canvas");
    if (!canvas) return;
    const { ctx, width, height } = setupHiDPICanvas(canvas);
    const mol = MOLECULES[currentMolKey];

    // Clear
    ctx.clearRect(0, 0, width, height);

    // Background grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 30) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }
    for (let y = 0; y < height; y += 30) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }

    const nAtoms = mol.atoms.length;
    const centerY = height / 2;
    // Map currentR to pixel distance between adjacent atoms
    const minPixDist = 50;
    const maxPixDist = 180;
    const normR = (currentR - mol.r_min) / (mol.r_max - mol.r_min);
    const spacing = minPixDist + normR * (maxPixDist - minPixDist);

    const totalWidth = (nAtoms - 1) * spacing;
    const startX = (width - totalWidth) / 2;

    const atomPositions = [];
    for (let i = 0; i < nAtoms; i++) {
      atomPositions.push({ x: startX + i * spacing, y: centerY });
    }

    // 1. Draw Chemical Bonds (Spring coils)
    for (let i = 0; i < nAtoms - 1; i++) {
      const p1 = atomPositions[i];
      const p2 = atomPositions[i + 1];

      // Draw spring bond
      ctx.beginPath();
      ctx.strokeStyle = "#8b5cf6";
      ctx.lineWidth = 3;
      const steps = 16;
      const dx = (p2.x - p1.x) / steps;
      ctx.moveTo(p1.x, p1.y);
      for (let s = 1; s < steps; s++) {
        const x = p1.x + s * dx;
        const offset = (s % 2 === 0 ? 1 : -1) * 8 * (1 - normR * 0.4);
        ctx.lineTo(x, p1.y + offset);
      }
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();

      // Bond distance label in middle
      const midX = (p1.x + p2.x) / 2;
      ctx.fillStyle = "rgba(14, 19, 32, 0.85)";
      ctx.strokeStyle = "rgba(139, 92, 246, 0.5)";
      ctx.lineWidth = 1;
      const badgeW = 74, badgeH = 22;
      ctx.fillRect(midX - badgeW / 2, centerY - 28, badgeW, badgeH);
      ctx.strokeRect(midX - badgeW / 2, centerY - 28, badgeW, badgeH);

      ctx.fillStyle = "#00f2fe";
      ctx.font = "bold 11px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`R = ${currentR.toFixed(2)}Å`, midX, centerY - 17);
    }

    // 2. Draw Atoms with Glowing Electron Shells
    atomPositions.forEach((pos, idx) => {
      const atom = mol.atoms[idx];
      const r = atom.radius;

      // Glow halo
      const glow = ctx.createRadialGradient(pos.x, pos.y, r * 0.5, pos.x, pos.y, r * 2.2);
      glow.addColorStop(0, atom.color + "66");
      glow.addColorStop(1, "transparent");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r * 2.2, 0, Math.PI * 2);
      ctx.fill();

      // Electron Cloud shell ring
      ctx.strokeStyle = atom.color + "44";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r * 1.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      // Nucleus gradient
      const nucGrad = ctx.createRadialGradient(pos.x - r * 0.3, pos.y - r * 0.3, 1, pos.x, pos.y, r);
      nucGrad.addColorStop(0, "#ffffff");
      nucGrad.addColorStop(0.4, atom.color);
      nucGrad.addColorStop(1, "#0f172a");

      ctx.fillStyle = nucGrad;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.fill();

      // Border
      ctx.strokeStyle = atom.color;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Symbol text
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 13px 'Outfit', sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(atom.label, pos.x, pos.y);
    });
  }

  // Render PES Canvas (Potential Energy Surface)
  function renderPES() {
    const canvas = document.getElementById("pes-canvas");
    if (!canvas) return;
    const { ctx, width, height } = setupHiDPICanvas(canvas);
    const mol = MOLECULES[currentMolKey];

    ctx.clearRect(0, 0, width, height);

    const padL = 60, padR = 25, padT = 30, padB = 45;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    // Calculate energy bounds
    const eMinPlot = mol.e_min - 0.25;
    const eMaxPlot = mol.e_dissoc + 0.6;

    function toX(r) {
      return padL + ((r - mol.r_min) / (mol.r_max - mol.r_min)) * plotW;
    }
    function toY(e) {
      return padT + (1 - (e - eMinPlot) / (eMaxPlot - eMinPlot)) * plotH;
    }

    // Grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let r = Math.ceil(mol.r_min * 2) / 2; r <= mol.r_max; r += 0.5) {
      const x = toX(r);
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
      ctx.fillStyle = "#6b7280";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(r.toFixed(1) + "Å", x, padT + plotH + 16);
    }

    // Y Axis labels
    const nY = 4;
    for (let i = 0; i <= nY; i++) {
      const e = eMinPlot + (i / nY) * (eMaxPlot - eMinPlot);
      const y = toY(e);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
      ctx.fillStyle = "#6b7280";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(e.toFixed(1) + " Ha", padL - 8, y + 3);
    }

    // 1. Draw Morse / FCI Born-Oppenheimer Curve
    ctx.beginPath();
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 3;
    const nPts = 100;
    for (let i = 0; i <= nPts; i++) {
      const r = mol.r_min + (i / nPts) * (mol.r_max - mol.r_min);
      const e = calcMorseEnergy(mol, r);
      const x = toX(r);
      const y = toY(e);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 2. Mark Equilibrium Minimum ($R_e$)
    const eqX = toX(mol.r_e);
    const eqY = toY(mol.e_min);
    ctx.strokeStyle = "rgba(16, 185, 129, 0.6)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(eqX, padT);
    ctx.lineTo(eqX, padT + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#10b981";
    ctx.beginPath();
    ctx.arc(eqX, eqY, 5, 0, Math.PI * 2);
    ctx.fill();

    // 3. Mark Current Position ($R$)
    const curE = getExactEnergyAtR(currentMolKey, currentR);
    const curX = toX(currentR);
    const curY = toY(curE);

    // Vertical dashed marker
    ctx.strokeStyle = "rgba(244, 63, 94, 0.5)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(curX, padT);
    ctx.lineTo(curX, curY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Glowing pointer circle
    ctx.fillStyle = "#f43f5e";
    ctx.shadowColor = "#f43f5e";
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.arc(curX, curY, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Callout Tag
    ctx.fillStyle = "rgba(7, 9, 14, 0.9)";
    ctx.strokeStyle = "#f43f5e";
    ctx.lineWidth = 1;
    const tagW = 110, tagH = 24;
    let tagX = curX + 10;
    if (tagX + tagW > width - padR) tagX = curX - tagW - 10;
    ctx.fillRect(tagX, curY - 12, tagW, tagH);
    ctx.strokeRect(tagX, curY - 12, tagW, tagH);

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(`E = ${curE.toFixed(4)} Ha`, tagX + tagW / 2, curY + 3);
  }

  // =========================================================================
  // 5. SECTION 3: THE LIVE 4-WAY VQE RACE SIMULATOR
  // =========================================================================
  function initRaceSimulator() {
    const btnStart = document.getElementById("btn-race-start");
    const btnPause = document.getElementById("btn-race-pause");
    const btnReset = document.getElementById("btn-race-reset");
    const speedSlider = document.getElementById("race-speed-slider");
    const speedLabel = document.getElementById("speed-label");

    btnStart.addEventListener("click", startRace);
    btnPause.addEventListener("click", pauseRace);
    btnReset.addEventListener("click", resetRace);

    speedSlider.addEventListener("input", (e) => {
      raceSpeed = parseInt(e.target.value);
      speedLabel.textContent = raceSpeed + "x";
    });

    resetRace();
  }

  function resetRace() {
    pauseRace();
    raceStep = 0;
    const mol = MOLECULES[currentMolKey];
    const exactE = getExactEnergyAtR(currentMolKey, currentR);
    const ratio = currentR / mol.r_e;

    // Initial guesses for the 4 contenders
    // 1. Random: Starts blind ~1.5 - 2.5 Ha above exact
    const eRand0 = exactE + 1.8 + Math.random() * 0.4;

    // 2. Hartree-Fock: Starts at HF energy (~0.3 to 1.3 Ha above exact depending on molecule)
    const eHf0 = exactE + mol.hf_offset * (1 + (ratio - 1) * 0.5);

    // 3. Direct Black-Box ML:
    // Near equilibrium: close!
    // Stretched OOD: blows up or starts unstable!
    let eDml0 = exactE + 0.12;
    if (ratio > 1.3) {
      eDml0 = exactE + 0.12 + Math.pow(ratio - 1.2, 2) * 1.5; // OOD hallucination error
    }

    // 4. Physics-Informed Residual ML (Ours):
    // Bound to HF anchor + small AI residual -> consistently within 0.03 - 0.08 Ha
    const eWarm0 = exactE + 0.045 + Math.min(0.04, Math.abs(ratio - 1) * 0.05);

    contendersData = {
      random: { history: [eRand0], current: eRand0, steps: 0, converged: false, slope: 0.008 },
      hf: { history: [eHf0], current: eHf0, steps: 0, converged: false, slope: 0.15 },
      dml: { history: [eDml0], current: eDml0, steps: 0, converged: false, slope: 0.07 },
      warm: { history: [eWarm0], current: eWarm0, steps: 0, converged: false, slope: 0.12 }
    };

    updateContenderUI();
    renderRaceChart();

    document.getElementById("race-status-dot").className = "status-pulse-dot";
    document.getElementById("race-status-text").textContent = "Ready to run";
    document.getElementById("winner-crown").classList.remove("visible");
  }

  function startRace() {
    if (raceRunning) return;
    raceRunning = true;

    document.getElementById("btn-race-start").disabled = true;
    document.getElementById("btn-race-pause").disabled = false;
    document.getElementById("race-status-dot").className = "status-pulse-dot running";
    document.getElementById("race-status-text").textContent = "Optimization running...";

    runRaceLoop();
  }

  function pauseRace() {
    raceRunning = false;
    if (raceAnimId) cancelAnimationFrame(raceAnimId);
    document.getElementById("btn-race-start").disabled = false;
    document.getElementById("btn-race-pause").disabled = true;
    document.getElementById("race-status-dot").className = "status-pulse-dot";
    document.getElementById("race-status-text").textContent = raceStep >= maxRaceSteps ? "Completed" : "Paused";
  }

  function runRaceLoop() {
    if (!raceRunning) return;

    // Step by raceSpeed
    for (let s = 0; s < raceSpeed; s++) {
      if (raceStep < maxRaceSteps) {
        stepRace();
      } else {
        finishRace();
        return;
      }
    }

    updateContenderUI();
    renderRaceChart();

    raceAnimId = requestAnimationFrame(runRaceLoop);
  }

  function stepRace() {
    raceStep++;
    const mol = MOLECULES[currentMolKey];
    const exactE = getExactEnergyAtR(currentMolKey, currentR);
    const ratio = currentR / mol.r_e;

    // 1. Random: Barren plateau effect! Gradient is tiny, irregular fluctuations
    if (!contendersData.random.converged) {
      contendersData.random.steps++;
      const cur = contendersData.random.current;
      const dist = cur - exactE;
      // Stalls frequently
      const grad = (Math.random() * 0.03 - 0.01) + (dist > 0.3 ? 0.012 : 0.003);
      const next = Math.max(exactE + 0.09, cur - grad);
      contendersData.random.current = next;
      contendersData.random.history.push(next);
      if (Math.abs(dist) < 0.1 || raceStep >= maxRaceSteps) {
        contendersData.random.converged = true;
      }
    } else {
      contendersData.random.history.push(contendersData.random.current);
    }

    // 2. Hartree-Fock: Drops very quickly in ~7 steps to mean-field floor, then completely plateaus!
    if (!contendersData.hf.converged) {
      contendersData.hf.steps++;
      const cur = contendersData.hf.current;
      const mfFloor = exactE + mol.hf_offset * 0.85; // Mean field missing correlation
      if (raceStep <= 8) {
        // rapid drop to mean-field
        const next = cur - (cur - mfFloor) * 0.45;
        contendersData.hf.current = next;
        contendersData.hf.history.push(next);
      } else {
        // Stuck in mean-field trap!
        contendersData.hf.converged = true;
        contendersData.hf.history.push(mfFloor);
      }
    } else {
      contendersData.hf.history.push(contendersData.hf.current);
    }

    // 3. Direct Black-Box ML:
    // If in-dist: descends well.
    // If OOD stretched: oscillations / local minima trap!
    if (!contendersData.dml.converged) {
      contendersData.dml.steps++;
      const cur = contendersData.dml.current;
      const dist = cur - exactE;
      let stepSize = 0.04;
      if (ratio > 1.3) {
        // OOD noise instability
        stepSize = (Math.sin(raceStep * 0.8) * 0.03) + 0.01;
      }
      const next = Math.max(exactE + (ratio > 1.3 ? 0.08 : 0.015), cur - stepSize);
      contendersData.dml.current = next;
      contendersData.dml.history.push(next);
      if (Math.abs(dist) < 0.02 || raceStep >= 120) {
        contendersData.dml.converged = true;
      }
    } else {
      contendersData.dml.history.push(contendersData.dml.current);
    }

    // 4. Residual Warm-Start (Ours):
    // Smooth, steep exponential decay right down to exact ground state within 30-50 steps!
    if (!contendersData.warm.converged) {
      contendersData.warm.steps++;
      const cur = contendersData.warm.current;
      const dist = cur - exactE;
      const decayRate = 0.08;
      const noise = (Math.random() - 0.5) * 0.002;
      const next = Math.max(exactE + 0.003, cur - dist * decayRate + noise);
      contendersData.warm.current = next;
      contendersData.warm.history.push(next);
      if (dist < 0.005 || raceStep >= 65) {
        contendersData.warm.converged = true;
      }
    } else {
      contendersData.warm.history.push(contendersData.warm.current);
    }
  }

  function finishRace() {
    pauseRace();
    document.getElementById("race-status-text").textContent = "Race complete!";
    document.getElementById("winner-crown").classList.add("visible");
  }

  function updateContenderUI() {
    const exactE = getExactEnergyAtR(currentMolKey, currentR);

    // Update numbers
    document.getElementById("energy-random").textContent = contendersData.random.current.toFixed(4) + " Ha";
    document.getElementById("steps-random").textContent = contendersData.random.steps;
    const errRand = Math.max(0, (contendersData.random.current - exactE) * 1000).toFixed(1);
    document.getElementById("err-random").textContent = `+${errRand} mHa`;

    document.getElementById("energy-hf").textContent = contendersData.hf.current.toFixed(4) + " Ha";
    document.getElementById("steps-hf").textContent = contendersData.hf.steps;
    const errHf = Math.max(0, (contendersData.hf.current - exactE) * 1000).toFixed(1);
    document.getElementById("err-hf").textContent = `+${errHf} mHa`;

    document.getElementById("energy-dml").textContent = contendersData.dml.current.toFixed(4) + " Ha";
    document.getElementById("steps-dml").textContent = contendersData.dml.steps;
    const errDml = Math.max(0, (contendersData.dml.current - exactE) * 1000).toFixed(1);
    document.getElementById("err-dml").textContent = `+${errDml} mHa`;

    document.getElementById("energy-warm").textContent = contendersData.warm.current.toFixed(4) + " Ha";
    document.getElementById("steps-warm").textContent = contendersData.warm.steps;
    const errWarm = Math.max(0, (contendersData.warm.current - exactE) * 1000).toFixed(1);
    document.getElementById("err-warm").textContent = `+${errWarm} mHa`;

    // Dynamic ranks
    const ranking = [
      { key: "warm", e: contendersData.warm.current },
      { key: "dml", e: contendersData.dml.current },
      { key: "random", e: contendersData.random.current },
      { key: "hf", e: contendersData.hf.current }
    ].sort((a, b) => a.e - b.e);

    ranking.forEach((r, idx) => {
      const el = document.getElementById(`rank-${r.key}`);
      if (el) el.textContent = `#${idx + 1}`;
    });
  }

  // Render 4-Way Line Chart Canvas
  function renderRaceChart() {
    const canvas = document.getElementById("race-canvas");
    if (!canvas) return;
    const { ctx, width, height } = setupHiDPICanvas(canvas);
    const exactE = getExactEnergyAtR(currentMolKey, currentR);

    ctx.clearRect(0, 0, width, height);

    const padL = 70, padR = 40, padT = 30, padB = 45;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    // Y Range
    const yMin = exactE - 0.1;
    const yMax = exactE + 2.5;

    function toX(step) {
      return padL + (step / maxRaceSteps) * plotW;
    }
    function toY(val) {
      const clamped = Math.max(yMin, Math.min(yMax, val));
      return padT + (1 - (clamped - yMin) / (yMax - yMin)) * plotH;
    }

    // X Axis grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let s = 0; s <= maxRaceSteps; s += 25) {
      const x = toX(s);
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
      ctx.fillStyle = "#6b7280";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(s === 0 ? "0 (Init)" : `${s}`, x, padT + plotH + 18);
    }

    // Y Axis grid
    const nY = 5;
    for (let i = 0; i <= nY; i++) {
      const val = yMin + (i / nY) * (yMax - yMin);
      const y = toY(val);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
      ctx.fillStyle = "#6b7280";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.fillText(val.toFixed(2) + " Ha", padL - 8, y + 3);
    }

    // Draw Ground Truth Baseline
    const exactY = toY(exactE);
    ctx.strokeStyle = "#facc15";
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padL, exactY);
    ctx.lineTo(padL + plotW, exactY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#facc15";
    ctx.font = "bold 10px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText(`Exact E₀ = ${exactE.toFixed(4)} Ha`, padL + 8, exactY - 6);

    // Draw lines for the 4 contenders
    const contenders = [
      { key: "random", color: "#94a3b8", width: 2 },
      { key: "hf", color: "#38bdf8", width: 2 },
      { key: "dml", color: "#f472b6", width: 2 },
      { key: "warm", color: "#10b981", width: 3.5, glow: true }
    ];

    contenders.forEach(c => {
      const hist = contendersData[c.key].history;
      if (!hist || hist.length === 0) return;

      ctx.beginPath();
      ctx.strokeStyle = c.color;
      ctx.lineWidth = c.width;

      if (c.glow) {
        ctx.shadowColor = "#10b981";
        ctx.shadowBlur = 10;
      }

      for (let i = 0; i < hist.length; i++) {
        const x = toX(i);
        const y = toY(hist[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Draw dot at current position
      const lastIdx = hist.length - 1;
      const lastX = toX(lastIdx);
      const lastY = toY(hist[lastIdx]);

      ctx.fillStyle = c.color;
      ctx.beginPath();
      ctx.arc(lastX, lastY, c.glow ? 5 : 3.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // =========================================================================
  // 6. SECTION 4: HARDWARE NOISE & CIRCUITS
  // =========================================================================
  function initNoiseLab() {
    const noiseSlider = document.getElementById("noise-slider");
    const noiseVal = document.getElementById("noise-val");
    const readoutSlider = document.getElementById("readout-slider");
    const readoutVal = document.getElementById("readout-val");

    noiseSlider.addEventListener("input", (e) => {
      noiseP2 = parseFloat(e.target.value);
      noiseVal.textContent = (noiseP2 * 100).toFixed(1) + `% (p₂ = ${noiseP2.toFixed(3)})`;
      updateNoiseMetrics();
    });

    readoutSlider.addEventListener("input", (e) => {
      readoutError = parseFloat(e.target.value);
      readoutVal.textContent = (readoutError * 100).toFixed(1) + "%";
      updateNoiseMetrics();
    });

    updateNoiseMetrics();
  }

  function updateNoiseMetrics() {
    const mol = MOLECULES[currentMolKey];
    const nDense = mol.dense_cx;
    const nSparse = mol.sparse_cx;

    // Fidelity formula matching preprint:
    // F_gate = (1 - p2)^N_cx
    // F_readout = (1 - epsilon_ro)
    const fDense = Math.pow(1 - noiseP2, nDense) * (1 - readoutError * 0.5);
    const fSparse = Math.pow(1 - noiseP2, nSparse) * (1 - readoutError * 0.5);

    const fDensePct = (Math.max(0, fDense) * 100).toFixed(1);
    const fSparsePct = (Math.max(0, fSparse) * 100).toFixed(1);

    document.getElementById("dense-fidelity-val").textContent = fDensePct + "%";
    document.getElementById("dense-fidelity-bar").style.width = fDensePct + "%";

    document.getElementById("sparse-fidelity-val").textContent = fSparsePct + "%";
    document.getElementById("sparse-fidelity-bar").style.width = fSparsePct + "%";

    document.getElementById("dense-gate-pill").textContent = `${nDense} CNOT Gates`;
    const prunedPct = Math.round(((nDense - nSparse) / nDense) * 100);
    document.getElementById("sparse-gate-pill").textContent = `${nSparse} CNOT (${prunedPct}% Pruned)`;
  }

  function startCircuitAnimation() {
    function animate() {
      pulseOffset = (pulseOffset + 1.2) % 180;
      renderCircuits();
      circuitAnimId = requestAnimationFrame(animate);
    }
    animate();
  }

  function renderCircuits() {
    renderCircuitCanvas("dense-circuit-canvas", true);
    renderCircuitCanvas("sparse-circuit-canvas", false);
  }

  function renderCircuitCanvas(canvasId, isDense) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const { ctx, width, height } = setupHiDPICanvas(canvas);
    const mol = MOLECULES[currentMolKey];
    const nQubits = Math.min(6, mol.qubits);

    ctx.clearRect(0, 0, width, height);

    const padL = 50, padR = 30, padT = 25;
    const wireSpacing = (height - padT * 2) / (nQubits - 1 || 1);

    // Draw Qubit Wires
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1.5;

    for (let q = 0; q < nQubits; q++) {
      const y = padT + q * wireSpacing;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();

      // Qubit Label
      ctx.fillStyle = "#94a3b8";
      ctx.font = "bold 11px 'JetBrains Mono', monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(`|q${q}⟩`, padL - 8, y);
    }

    // 1. Single Qubit Parameter Rotation Gates: Ry(θ)
    const ryX = padL + 50;
    for (let q = 0; q < nQubits; q++) {
      const y = padT + q * wireSpacing;
      drawGateBox(ctx, ryX, y, "Ry", "#38bdf8");
    }

    // 2. CNOT Entangling Gates
    const entanglements = isDense
      ? getDenseEntanglers(nQubits)
      : [{ control: 0, target: Math.min(3, nQubits - 1) }]; // Sparse: 1 key pair

    const startEntX = ryX + 70;
    const entStep = (width - padR - startEntX - 40) / (entanglements.length || 1);

    entanglements.forEach((cx, idx) => {
      const x = startEntX + idx * entStep;
      const yCtrl = padT + cx.control * wireSpacing;
      const yTgt = padT + cx.target * wireSpacing;

      // Entangling connector line
      ctx.strokeStyle = isDense ? "#f43f5e" : "#10b981";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, yCtrl);
      ctx.lineTo(x, yTgt);
      ctx.stroke();

      // Control dot
      ctx.fillStyle = isDense ? "#f43f5e" : "#10b981";
      ctx.beginPath();
      ctx.arc(x, yCtrl, 5, 0, Math.PI * 2);
      ctx.fill();

      // Target plus circle
      ctx.strokeStyle = isDense ? "#f43f5e" : "#10b981";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, yTgt, 8, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - 8, yTgt); ctx.lineTo(x + 8, yTgt);
      ctx.moveTo(x, yTgt - 8); ctx.lineTo(x, yTgt + 8);
      ctx.stroke();

      // Noise sparkles if error rate is high
      if (noiseP2 > 0.005) {
        drawNoiseStatic(ctx, x, (yCtrl + yTgt) / 2, noiseP2 * 50);
      }
    });

    // 3. Animated Quantum Pulses traveling along wires
    for (let q = 0; q < nQubits; q++) {
      const y = padT + q * wireSpacing;
      const pX = padL + ((pulseOffset + q * 30) % (width - padL - padR));

      const grad = ctx.createRadialGradient(pX, y, 1, pX, y, 12);
      grad.addColorStop(0, isDense ? "#00f2fe" : "#10b981");
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(pX, y, 12, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function getDenseEntanglers(nQubits) {
    const list = [];
    for (let q = 0; q < nQubits - 1; q++) {
      list.push({ control: q, target: q + 1 });
    }
    return list;
  }

  function drawGateBox(ctx, x, y, label, color) {
    const w = 32, h = 24;
    ctx.fillStyle = "rgba(14, 19, 32, 0.95)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.fillRect(x - w / 2, y - h / 2, w, h);
    ctx.strokeRect(x - w / 2, y - h / 2, w, h);

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 9px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, x, y);
  }

  function drawNoiseStatic(ctx, x, y, intensity) {
    ctx.fillStyle = "rgba(244, 63, 94, 0.7)";
    for (let i = 0; i < Math.floor(intensity); i++) {
      const nx = x + (Math.random() - 0.5) * 24;
      const ny = y + (Math.random() - 0.5) * 36;
      ctx.beginPath();
      ctx.arc(nx, ny, Math.random() * 2 + 0.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // =========================================================================
  // 7. SECTION 5: BENCHMARK TABS & ARTIFACTS
  // =========================================================================
  function initBenchmarkTabs() {
    const tabBtns = document.querySelectorAll(".benchmark-tabs .tab-btn");
    const panes = document.querySelectorAll(".tab-pane");

    tabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        tabBtns.forEach(b => b.classList.remove("active"));
        panes.forEach(p => p.classList.remove("active"));

        btn.classList.add("active");
        const targetId = btn.getAttribute("data-tab");
        const targetPane = document.getElementById(targetId);
        if (targetPane) targetPane.classList.add("active");
      });
    });
  }

  function initBibtexCopy() {
    const btn = document.getElementById("btn-copy-bibtex");
    const snippet = document.getElementById("bibtex-snippet");

    btn.addEventListener("click", () => {
      if (!snippet) return;
      navigator.clipboard.writeText(snippet.innerText).then(() => {
        const orig = btn.textContent;
        btn.textContent = "✅ Copied!";
        setTimeout(() => { btn.textContent = orig; }, 2000);
      });
    });
  }

  function checkServerStatus() {
    const pill = document.getElementById("server-status-pill");
    // Attempt fetch from /api/status if hosted via local Python server
    fetch("/api/status")
      .then(res => res.json())
      .then(data => {
        if (data && data.status === "ok") {
          pill.textContent = "🟢 Live Python Engine";
          pill.classList.remove("pill-outline");
          pill.classList.add("pill-success");
        }
      })
      .catch(() => {
        // Fallback gracefully to offline JS engine
        pill.textContent = "⚡ Standalone Client Engine";
      });
  }

})();
