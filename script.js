// Small bits of interactivity for the project page.

document.addEventListener("DOMContentLoaded", () => {
  shuffleEqualAuthors();
  setupCopyButtons();
});

/**
 * Randomizes the order of the co-first authors on each page load, so neither
 * name is consistently listed first ("order decided by coin flip").
 */
function shuffleEqualAuthors() {
  const container = document.getElementById("shuffle-authors");
  if (!container) return;

  const authors = Array.from(container.children);
  if (authors.length < 2) return;

  // Coin flip: swap the two authors half the time.
  if (Math.random() < 0.5) {
    container.appendChild(authors[0]);
  }
}

/**
 * Wires up any button with a `data-copy-target` attribute to copy the text
 * content of the referenced element to the clipboard, with brief feedback.
 */
function setupCopyButtons() {
  const buttons = document.querySelectorAll("[data-copy-target]");

  buttons.forEach((button) => {
    const label = button.querySelector(".copy-label") ?? button;
    const originalText = label.textContent;

    button.addEventListener("click", async () => {
      const target = document.querySelector(button.dataset.copyTarget);
      if (!target) return;

      const text = target.textContent.trim();

      try {
        await navigator.clipboard.writeText(text);
      } catch {
        // Fallback for browsers without the async clipboard API.
        fallbackCopy(text);
      }

      // Brief "Copied!" confirmation.
      label.textContent = "Copied!";
      button.classList.add("copied");
      setTimeout(() => {
        label.textContent = originalText;
        button.classList.remove("copied");
      }, 1500);
    });
  });
}

/** Legacy clipboard copy using a hidden textarea + execCommand. */
function fallbackCopy(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  try {
    document.execCommand("copy");
  } catch {
    // Nothing more we can do; user can select the text manually.
  }
  document.body.removeChild(textarea);
}

// ---- "MILLIVID" scrolling decoration (inlined from anim.html) ----
// Self-contained so its many locals don't leak into the rest of the file.
// CSS variables are read from the .millivid-anim container (not :root).
(function millividAnimation() {
  const canvas = document.getElementById("screen");
  if (!canvas) return;
  const root = canvas.closest(".millivid-anim");
  if (!root) return;

  // ---- 5-row pixel glyphs. "M" is 5 wide, everything else is 3 wide. ----
  const GLYPHS = {
    M: ["10001", "11011", "10101", "10001", "10001"],
    I: ["111", "010", "010", "010", "111"],
    L: ["100", "100", "100", "100", "111"],
    V: ["101", "101", "101", "101", "010"],
    D: ["110", "101", "101", "101", "110"],
  };

  const ROWS = 5;
  const WORD = "MILLIVID";
  const LETTER_GAP = 1; // blank columns between letters
  const WORD_GAP = 3; // blank columns after the word before it repeats

  // Build the repeating "unit" as columns; each column is 5 bits.
  // Also record a colour group per column: "MILLI" -> blue, "VID" -> orange.
  const UNIT = [];
  const UNIT_GROUP = [];
  (function buildUnit() {
    const pushBlank = (n, group) => {
      for (let i = 0; i < n; i++) {
        UNIT.push([0, 0, 0, 0, 0]);
        UNIT_GROUP.push(group);
      }
    };
    WORD.split("").forEach((ch, idx) => {
      const group = idx < 5 ? "milli" : "vid"; // MILLI = first 5 letters
      const g = GLYPHS[ch];
      const w = g[0].length;
      for (let c = 0; c < w; c++) {
        const col = [];
        for (let r = 0; r < ROWS; r++) col.push(g[r][c] === "1" ? 1 : 0);
        UNIT.push(col);
        UNIT_GROUP.push(group);
      }
      if (idx < WORD.length - 1) pushBlank(LETTER_GAP, group);
    });
    pushBlank(WORD_GAP, "vid");
  })();

  const UNIT_W = UNIT.length;
  const VISIBLE_COLS = UNIT_W - WORD_GAP; // box just holds one MILLIVID

  // ---- Sizing ----
  const css = getComputedStyle(root);
  const CELL = parseInt(css.getPropertyValue("--cell"));
  const COLOR_LINE = css.getPropertyValue("--grid-line").trim();
  const COLOR_BG = css.getPropertyValue("--cell-bg").trim();
  const COLOR_BLUE = css.getPropertyValue("--text-blue").trim();
  const COLOR_VID = css.getPropertyValue("--vid-color").trim();

  const ctx = canvas.getContext("2d");

  const cssW = VISIBLE_COLS * CELL; // no margins
  const cssH = ROWS * CELL;
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = cssW + "px";
  canvas.style.height = cssH + "px";
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  ctx.scale(dpr, dpr);

  function lit(globalCol, row) {
    const c = ((globalCol % UNIT_W) + UNIT_W) % UNIT_W;
    return UNIT[c][row] === 1;
  }

  function groupColor(globalCol) {
    const c = ((globalCol % UNIT_W) + UNIT_W) % UNIT_W;
    return UNIT_GROUP[c] === "vid" ? COLOR_VID : COLOR_BLUE;
  }

  // Draw the sliding strip at a fractional pixel offset.
  function draw(offsetPx) {
    // Snap the scroll offset to the device-pixel grid. Because CELL is an
    // integer, every cell and separator derived from this offset then lands on
    // a whole device pixel, so the figure renders crisp and all layers move in
    // lockstep (no shimmer between the cells and the grid lines).
    offsetPx = Math.round(offsetPx * dpr) / dpr;
    ctx.clearRect(0, 0, cssW, cssH);
    const startCol = Math.floor(offsetPx / CELL) - 1;
    const endCol = startCol + VISIBLE_COLS + 2;

    // 1) Fill cells (no per-cell outline, so lines never duplicate).
    for (let gcol = startCol; gcol <= endCol; gcol++) {
      const x = gcol * CELL - offsetPx;
      if (x > cssW || x + CELL < 0) continue;
      const litColor = groupColor(gcol);
      for (let r = 0; r < ROWS; r++) {
        const y = r * CELL;
        ctx.fillStyle = lit(gcol, r) ? litColor : COLOR_BG;
        ctx.fillRect(x, y, CELL, CELL);
      }
    }

    // 2) Thin black separators (interior only; the box border covers the
    //    outer edges). Drawn as 1-device-pixel fills; since offsetPx is snapped
    //    to the device grid above, these land on whole pixels and stay sharp.
    ctx.fillStyle = COLOR_LINE;
    const lineW = 1 / dpr; // 1 device pixel in CSS units
    for (let gcol = startCol; gcol <= endCol + 1; gcol++) {
      const x = gcol * CELL - offsetPx;
      if (x <= 0 || x >= cssW) continue;
      ctx.fillRect(x, 0, lineW, cssH);
    }
    for (let r = 1; r < ROWS; r++) {
      ctx.fillRect(0, r * CELL, cssW, lineW);
    }
  }

  // ---- Smooth continuous scroll ----
  const SPEED_PX = 11 * CELL; // ~11 columns/sec, in px/sec
  const LOOP = UNIT_W * CELL; // seamless wrap distance
  let start = null;

  function frame(ts) {
    if (start === null) start = ts;
    const elapsed = (ts - start) / 1000;
    const offsetPx = (elapsed * SPEED_PX) % LOOP;
    draw(offsetPx);
    requestAnimationFrame(frame);
  }

  draw(0);
  requestAnimationFrame(frame);
})();
