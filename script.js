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

// ---- Coarse-to-fine rollout animation ----
// A single grid (one frame of strategies/millivid.json, rows x cols of 0/1/2)
// whose generation front advances through the rollout's time frames while the
// whole grid drifts leftward. Each row is coloured by its front: every column
// at or left of the right-most generated (value 1 or 2) cell is "fixed" / grey
// (already-generated frames, extending off the left edge); columns to the right
// are white "future" frames. Only the right-most N columns ever get generated;
// one full pass through the frames is a "cycle". Over each cycle the grid
// scrolls left by exactly N columns, then the content is shifted right by N and
// the frames replay -- so the grid lands back at the same X position every cycle
// and the fixed region flows seamlessly across the reset.
(function rolloutAnimations() {
  // Each <canvas data-strategy="..."> in a .rollout-anim is driven by the data
  // registered on window.ROLLOUT_STRATEGIES by the matching strategy script
  // (e.g. strategies/millivid.js). Using script tags instead of fetch() means
  // this works on file:// as well as over http.
  const canvases = document.querySelectorAll(".rollout-anim canvas[data-strategy]");
  canvases.forEach((canvas) => {
    const root = canvas.closest(".rollout-anim");
    const frames =
      window.ROLLOUT_STRATEGIES &&
      window.ROLLOUT_STRATEGIES[canvas.dataset.strategy];
    if (!root || !frames) return;
    // A canvas with data-step uses the stepped renderer (hold the grid, then
    // ease forward by `step` columns, repeat) instead of the continuous scroll.
    if (canvas.dataset.step) startStepped(canvas, root, frames);
    else start(canvas, root, frames);
  });

  function start(canvas, root, frames) {
    const N_FRAMES = frames.length; // number of rollout (time) frames
    const ROWS = frames[0].length;
    const COLS = frames[0][0].length;

    // Padding columns on each side (default 1). Left padding renders grey (it is
    // left of the fixed seed) and right padding white (future), so a narrower
    // grid can be widened to match another by padding it. data-pad-left /
    // data-pad-right set these (e.g. to match the MilliVid box width).
    const padOf = (v) => (v !== undefined && v !== "" ? parseInt(v, 10) : 1);
    const PAD_LEFT = padOf(canvas.dataset.padLeft);
    const PAD_RIGHT = padOf(canvas.dataset.padRight);

    // N = number of columns that ever hold a denoised (value 2) token. These are
    // the right-most columns; the grid scrolls left by N columns each cycle and
    // is shifted right by N at the reset, so it returns to the same X position.
    const denoisedCols = new Set();
    frames.forEach((f) =>
      f.forEach((row) =>
        row.forEach((v, c) => {
          if (v === 2) denoisedCols.add(c);
        }),
      ),
    );
    const N = denoisedCols.size;

    const VISIBLE_COLS = PAD_LEFT + COLS + PAD_RIGHT; // window: padding + grid

    // First frame in which each cell becomes denoised (orange / value 2), or
    // Infinity if never. Once the denoising front has passed over a cell it
    // stays "fixed" / grey for the rest of the cycle (unless it is context).
    const firstOrange = [];
    for (let r = 0; r < ROWS; r++) firstOrange.push(new Array(COLS).fill(Infinity));
    frames.forEach((f, fi) => {
      for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
          if (f[r][c] === 2 && fi < firstOrange[r][c]) firstOrange[r][c] = fi;
        }
      }
    });

    // Initial "fixed" seed: at the first frame everything to the left of the
    // left-most orange column counts as already generated.
    let initialFixed = COLS;
    frames[0].forEach((row) =>
      row.forEach((v, c) => {
        if (v === 2 && c < initialFixed) initialFixed = c;
      }),
    );

    // ---- Sizing (mirrors the MILLIVID animation) ----
    const css = getComputedStyle(root);
    const CELL = parseInt(css.getPropertyValue("--cell"));
    const COLOR_LINE = css.getPropertyValue("--grid-line").trim();
    const COLOR_BG = css.getPropertyValue("--cell-bg").trim(); // future (white)
    const COLOR_FIXED = css.getPropertyValue("--fixed-color").trim(); // generated
    const COLOR_CTX = css.getPropertyValue("--ctx-color").trim(); // value 1
    const COLOR_GEN = css.getPropertyValue("--gen-color").trim(); // value 2

    const ctx = canvas.getContext("2d");
    const cssW = VISIBLE_COLS * CELL;
    const cssH = ROWS * CELL;
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.scale(dpr, dpr);

    // Colour of grid-column gcol (gcol 0 is the left margin). Context (value 1)
    // and denoised (value 2) cells take priority and render blue / orange. Any
    // other cell is "fixed" / grey if it is in the initial seed (left of the
    // first frame's left-most orange, including off-grid columns) or the
    // denoising front has already passed over it this cycle; otherwise it is a
    // white "future" frame. No wrapping: the seamless loop comes from the
    // per-cycle scroll + reset, not from tiling.
    function cellColor(gcol, row, frameIdx) {
      const d = gcol - PAD_LEFT;
      const inGrid = d >= 0 && d < COLS;
      const value = inGrid ? frames[frameIdx][row][d] : 0;
      if (value === 1) return COLOR_CTX; // blue (context) -- takes priority
      if (value === 2) return COLOR_GEN; // orange (denoised)
      const fixed =
        d < initialFixed || (inGrid && firstOrange[row][d] <= frameIdx);
      return fixed ? COLOR_FIXED : COLOR_BG;
    }

    function draw(offsetPx, frameIdx) {
      // Snap the scroll offset to the device-pixel grid so cells and separators
      // land on whole pixels and move in lockstep (no shimmer); same trick as
      // the MILLIVID animation.
      offsetPx = Math.round(offsetPx * dpr) / dpr;
      ctx.clearRect(0, 0, cssW, cssH);
      const startCol = Math.floor(offsetPx / CELL) - 1;
      const endCol = startCol + VISIBLE_COLS + 2;

      // 1) Fill cells by value for the current frame.
      for (let gcol = startCol; gcol <= endCol; gcol++) {
        const x = gcol * CELL - offsetPx;
        if (x > cssW || x + CELL < 0) continue;
        for (let r = 0; r < ROWS; r++) {
          ctx.fillStyle = cellColor(gcol, r, frameIdx);
          ctx.fillRect(x, r * CELL, CELL, CELL);
        }
      }

      // 2) Thin black separators (1 device pixel), drawn across the whole grid.
      ctx.fillStyle = COLOR_LINE;
      const lineW = 1 / dpr;
      for (let gcol = startCol; gcol <= endCol + 1; gcol++) {
        const x = gcol * CELL - offsetPx;
        if (x <= 0 || x >= cssW) continue;
        ctx.fillRect(x, 0, lineW, cssH);
      }
      for (let r = 1; r < ROWS; r++) {
        ctx.fillRect(0, r * CELL, cssW, lineW);
      }
    }

    // ---- One cycle: morph through the frames while scrolling left by N cols ----
    const FRAME_MS = 260; // time each rollout frame is shown before advancing
    const CYCLE_MS = N_FRAMES * FRAME_MS; // one full pass through the frames
    let startTs = null;

    function frame(ts) {
      if (startTs === null) startTs = ts;
      const tau = (ts - startTs) % CYCLE_MS; // time within the current cycle
      const frac = tau / CYCLE_MS; // 0 -> 1 across the cycle
      const frameIdx = Math.min(N_FRAMES - 1, Math.floor(tau / FRAME_MS));
      // Scroll left by exactly N columns over the cycle; the reset back to 0
      // coincides with the frames restarting (= shift content right by N).
      const offsetPx = frac * N * CELL;
      draw(offsetPx, frameIdx);
      requestAnimationFrame(frame);
    }

    draw(0, 0);
    requestAnimationFrame(frame);
  }

  // ---- Stepped renderer (baseline) ----------------------------------------
  // The baseline strategy is a single static grid (e.g. [1,1,2,2]: two context
  // + two denoised cells), so the continuous scroll above just drifts the block
  // and snaps back, which reads as jerky. Instead, animate it as discrete steps:
  // hold the freshly generated grid, then ease forward (leftward) by `step`
  // columns, then repeat.
  //
  // Seamlessness: the grid's column origin advances by `step` each cycle while
  // the scroll offset advances to match, so the generation front stays at the
  // same screen position across cycles. The grey "fixed" history (left) and the
  // white "pending" future (right) are uniform and effectively infinite, so the
  // per-cycle advance is invisible there. At the cycle boundary the scroll
  // offset is continuous and only the colours change in place -- the front's
  // oldest context retires to grey, its denoised cells become context, and two
  // new denoised cells light up -- so the "update" looks like a generation step
  // rather than a teleport, and the loop is perfectly seamless.
  function startStepped(canvas, root, frames) {
    const grid = frames[0]; // single frame: ROWS x COLS of 0/1/2
    const ROWS = grid.length;
    const COLS = grid[0].length;

    const padOf = (v) => (v !== undefined && v !== "" ? parseInt(v, 10) : 1);
    const PAD_LEFT = padOf(canvas.dataset.padLeft);
    const PAD_RIGHT = padOf(canvas.dataset.padRight);
    const STEP = parseInt(canvas.dataset.step, 10) || 1;
    const VISIBLE_COLS = PAD_LEFT + COLS + PAD_RIGHT;

    // Left-most denoised (value 2) column: everything to its left counts as
    // already generated (grey / fixed), everything past the grid is white.
    let firstDenoise = COLS;
    grid.forEach((row) =>
      row.forEach((v, c) => {
        if (v === 2 && c < firstDenoise) firstDenoise = c;
      }),
    );

    // ---- Sizing (mirrors start()) ----
    const css = getComputedStyle(root);
    const CELL = parseInt(css.getPropertyValue("--cell"));
    const COLOR_LINE = css.getPropertyValue("--grid-line").trim();
    const COLOR_BG = css.getPropertyValue("--cell-bg").trim(); // future (white)
    const COLOR_FIXED = css.getPropertyValue("--fixed-color").trim(); // generated
    const COLOR_CTX = css.getPropertyValue("--ctx-color").trim(); // value 1
    const COLOR_GEN = css.getPropertyValue("--gen-color").trim(); // value 2

    const ctx = canvas.getContext("2d");
    const cssW = VISIBLE_COLS * CELL;
    const cssH = ROWS * CELL;
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.scale(dpr, dpr);

    // Colour of grid-column gcol after `step` completed advances. The grid
    // origin sits `step * STEP` columns to the right each cycle (see above).
    function cellColor(gcol, row, step) {
      const d = gcol - PAD_LEFT - step * STEP;
      const inGrid = d >= 0 && d < COLS;
      const value = inGrid ? grid[row][d] : 0;
      if (value === 1) return COLOR_CTX; // blue (context)
      if (value === 2) return COLOR_GEN; // orange (denoised)
      return d < firstDenoise ? COLOR_FIXED : COLOR_BG; // grey past / white future
    }

    function draw(offsetPx, step) {
      offsetPx = Math.round(offsetPx * dpr) / dpr;
      ctx.clearRect(0, 0, cssW, cssH);
      const startCol = Math.floor(offsetPx / CELL) - 1;
      const endCol = startCol + VISIBLE_COLS + 2;

      // 1) Fill cells by value.
      for (let gcol = startCol; gcol <= endCol; gcol++) {
        const x = gcol * CELL - offsetPx;
        if (x > cssW || x + CELL < 0) continue;
        for (let r = 0; r < ROWS; r++) {
          ctx.fillStyle = cellColor(gcol, r, step);
          ctx.fillRect(x, r * CELL, CELL, CELL);
        }
      }

      // 2) Thin black separators (1 device pixel).
      ctx.fillStyle = COLOR_LINE;
      const lineW = 1 / dpr;
      for (let gcol = startCol; gcol <= endCol + 1; gcol++) {
        const x = gcol * CELL - offsetPx;
        if (x <= 0 || x >= cssW) continue;
        ctx.fillRect(x, 0, lineW, cssH);
      }
      for (let r = 1; r < ROWS; r++) {
        ctx.fillRect(0, r * CELL, cssW, lineW);
      }
    }

    // ---- Timing: hold the grid, then ease forward by STEP columns ----
    const HOLD_MS = 650; // pause showing the freshly updated grid
    const SHIFT_MS = 550; // eased slide forward by STEP columns
    const CYCLE_MS = HOLD_MS + SHIFT_MS;
    // Cubic ease-in-out: accelerate, then decelerate.
    const ease = (t) =>
      t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    let startTs = null;

    function frame(ts) {
      if (startTs === null) startTs = ts;
      const elapsed = ts - startTs;
      const step = Math.floor(elapsed / CYCLE_MS); // advances done so far
      const within = elapsed - step * CYCLE_MS;
      // 0 during the hold, eased 0->1 during the shift.
      const e = within < HOLD_MS ? 0 : ease((within - HOLD_MS) / SHIFT_MS);
      // Offset stays continuous across the cycle boundary ((step + 1) at e=1
      // equals (step + 1) at the next cycle's e=0), so only the colours change
      // at the boundary -- the seamless "generation" update.
      const offsetPx = (step + e) * STEP * CELL;
      draw(offsetPx, step);
      requestAnimationFrame(frame);
    }

    draw(0, 0);
    requestAnimationFrame(frame);
  }
})();

// ---- Interactive adaptive-autoencoder figure ----
// Click a scene to choose the Ground Truth/Reconstruction pair; click a level
// to lock the reconstruction's token count, or hover a level to preview it.
(function adaptiveAutoencoderFigure() {
  const figure = document.getElementById("ae-figure");
  if (!figure) return;

  // --- Image sources --------------------------------------------------------
  // One folder per scene under autoencoder/. Each holds gt.png plus, for both
  // the "adaptive" and "cascaded" variants, loopcraft_autoencoder[_cascaded]_
  // level{0-3}.png reconstructions.
  const SCENE_DIRS = [
    "000-653_f0861",
    "001-480_f0621",
    "002-032_f0676",
    "002-481_f0164",
  ];
  const variantSelect = figure.querySelector("#ae-variant");

  function assetUrl(kind, scene, level) {
    const dir = `autoencoder/${SCENE_DIRS[scene]}`;
    if (kind === "thumb" || kind === "gt") return `${dir}/gt.png`;
    const variant = variantSelect.value === "cascaded" ? "_cascaded" : "";
    return `${dir}/loopcraft_autoencoder${variant}_level${level}.png`;
  }

  // --- Elements + state -----------------------------------------------------
  const thumbs = Array.from(figure.querySelectorAll(".ae-thumb"));
  const levels = Array.from(figure.querySelectorAll(".ae-level"));
  const gtImg = figure.querySelector("#ae-gt");
  const reconImg = figure.querySelector("#ae-recon");

  let selectedScene = 0;
  let selectedLevel = 0;

  // Render both images for a given (scene, level). GT depends only on the
  // scene; the reconstruction depends on both.
  const render = (scene, level) => {
    gtImg.src = assetUrl("gt", scene);
    reconImg.src = assetUrl("recon", scene, level);
  };
  const markSelected = (list, idx) =>
    list.forEach((el, i) => el.classList.toggle("is-selected", i === idx));

  // Scene thumbnails: load once; click selects, hover previews.
  thumbs.forEach((btn, i) => {
    btn.querySelector("img").src = assetUrl("thumb", i);
    btn.addEventListener("click", () => {
      selectedScene = i;
      markSelected(thumbs, i);
      render(selectedScene, selectedLevel);
    });
    btn.addEventListener("mouseenter", () => render(i, selectedLevel));
    btn.addEventListener("mouseleave", () =>
      render(selectedScene, selectedLevel),
    );
  });

  // Level buttons: click locks the level; hover previews it.
  levels.forEach((btn, i) => {
    btn.addEventListener("click", () => {
      selectedLevel = i;
      markSelected(levels, i);
      render(selectedScene, selectedLevel);
    });
    btn.addEventListener("mouseenter", () => render(selectedScene, i));
    btn.addEventListener("mouseleave", () =>
      render(selectedScene, selectedLevel),
    );
  });

  // Variant dropdown (adaptive vs. cascaded): re-render the reconstruction.
  variantSelect.addEventListener("change", () =>
    render(selectedScene, selectedLevel),
  );

  // Initial state.
  markSelected(thumbs, selectedScene);
  markSelected(levels, selectedLevel);
  render(selectedScene, selectedLevel);
})();

// ---- Results video widget ----
// Shows one video at a time from the results/ folder, chosen via a dropdown.
// A random video is selected on page load.
(function resultsVideoWidget() {
  const select = document.getElementById("results-select");
  const video = document.getElementById("results-video");
  if (!select || !video) return;

  // File stems (sans .mp4) of the videos in results/.
  const VIDEOS = [
    "001_382",
    "001_853",
    "002_549",
    "002_616",
    "003_166",
    "003_913",
    "004_230",
    "004_275",
    "004_675",
    "005_301",
    "005_675",
    "006_387",
    "006_727",
    "006_729",
    "007_396",
    "009_150",
    "009_659",
    "009_730",
  ];

  // Populate the dropdown.
  VIDEOS.forEach((name, i) => {
    const option = document.createElement("option");
    option.value = `results/${name}.mp4`;
    option.textContent = `${name}.mp4`;
    select.appendChild(option);
  });

  const load = () => {
    video.src = select.value;
    video.load();
  };
  select.addEventListener("change", load);

  // The first column shows the context window for the first 256 frames; after
  // that, it shows the ground-truth continuation. At 20 fps that switch happens
  // at 256 / 20 = 12.8 s. Updating on timeupdate/seeking keeps the label correct
  // even when the user scrubs manually.
  const contextLabel = document.getElementById("results-context-label");
  const FPS = 20;
  const SWITCH_TIME = 256 / FPS;
  const updateContextLabel = () => {
    if (!contextLabel) return;
    contextLabel.textContent =
      video.currentTime >= SWITCH_TIME ? "Ground Truth" : "Context";
  };
  ["timeupdate", "seeking", "seeked", "loadedmetadata"].forEach((evt) =>
    video.addEventListener(evt, updateContextLabel),
  );

  // Random video on page load.
  select.selectedIndex = Math.floor(Math.random() * VIDEOS.length);
  load();
})();

// ---- Quality vs. Consistency line charts ----
// Two SVG line charts built from window.QUALITY_CONSISTENCY (loaded from
// plots/quality_consistency.js). LPIPS has 768 per-frame samples; FVD has 48
// samples taken every 16 frames. Both share the 0..768 frame x-axis. Dashed
// lines mark the best/worst reference levels.
(function qualityConsistencyCharts() {
  const data = window.QUALITY_CONSISTENCY;
  if (!data) return;

  const SVG_NS = "http://www.w3.org/2000/svg";
  // Method id -> line colour. Ordered worst (green) to best (orange), matching
  // the source figure: orange is the most consistent / highest quality.
  const COLORS = {
    ncwwsaan52rr: "#ff7f0e", // orange
    dj1luygiga68: "#1f77b4", // blue
    uxl1hf3ibsou: "#2ca02c", // green
  };

  // The viewBox is exactly the plot box, so the box aligns flush with the page
  // column; axis tick labels are drawn at negative / past-edge coordinates and
  // bleed outside via overflow: visible.
  const PW = 400;
  const PH = 246; // 20% taller than the original 205
  const X_MAX = 768;
  const X_TICKS = [0, 128, 256, 384, 512, 640, 768];

  // Per-metric axis config. xOf maps a sample index to its frame number.
  const METRICS = {
    lpips: {
      yMin: 0,
      yMax: 0.85,
      yTicks: [0, 0.2, 0.4, 0.6, 0.8],
      yFmt: (v) => v.toFixed(1),
      xOf: (i) => i + 1, // 768 samples -> frames 1..768
      refs: [
        { v: data.lpips_worst, label: "Worst Possible (Random Frames)" },
        { v: data.lpips_best, label: "Best Possible (Autoencoded Frames)" },
      ],
    },
    fvd: {
      // Drop the floor below zero so the near-zero "best" line sits above the
      // bottom axis with room for its label underneath.
      yMin: -110,
      yMax: 870,
      yTicks: [0, 200, 400, 600, 800],
      yFmt: (v) => String(v),
      xOf: (i) => (i + 1) * 16, // 48 samples every 16 frames -> 16..768
      refs: [
        {
          v: data.fvd_best,
          label: "Best Possible (Autoencoded Frames)",
          below: true,
        },
      ],
    },
  };

  const el = (name, attrs) => {
    const node = document.createElementNS(SVG_NS, name);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  };

  document.querySelectorAll("#quality-consistency .qc-plot").forEach((host) => {
    const metric = host.dataset.metric;
    const cfg = METRICS[metric];
    const series = data[metric];
    if (!cfg || !series) return;

    const xPx = (frame) => (frame / X_MAX) * PW;
    const yPx = (v) => (1 - (v - cfg.yMin) / (cfg.yMax - cfg.yMin)) * PH;

    // overflow: visible (set in CSS) lets the bled axis labels show outside.
    const svg = el("svg", { viewBox: `0 0 ${PW} ${PH}` });

    // Horizontal gridlines + y-axis tick labels (labels bleed left of the box).
    cfg.yTicks.forEach((t) => {
      const y = yPx(t);
      svg.appendChild(
        el("line", {
          x1: 0,
          y1: y,
          x2: PW,
          y2: y,
          stroke: "#eef2f7",
          "stroke-width": 1,
        }),
      );
      const label = el("text", {
        x: -8,
        y: y,
        "text-anchor": "end",
        "dominant-baseline": "central",
        fill: "#64748b",
      });
      label.textContent = cfg.yFmt(t);
      svg.appendChild(label);
    });

    X_TICKS.forEach((t) => {
      const label = el("text", {
        x: xPx(t),
        y: PH + 16,
        "text-anchor": "middle",
        fill: "#64748b",
      });
      label.textContent = String(t);
      svg.appendChild(label);
    });

    // Dashed best/worst reference lines, with a grey annotation above each.
    cfg.refs.forEach((ref) => {
      const y = yPx(ref.v);
      svg.appendChild(
        el("line", {
          x1: 0,
          y1: y,
          x2: PW,
          y2: y,
          stroke: "#94a3b8",
          "stroke-width": 1.5,
          "stroke-dasharray": "5 4",
        }),
      );
      const note = el("text", {
        x: 6,
        y: ref.below ? y + 6 : y - 6,
        "text-anchor": "start",
        "dominant-baseline": ref.below ? "hanging" : "auto",
        fill: "#94a3b8",
      });
      note.textContent = ref.label;
      svg.appendChild(note);
    });

    // Data lines.
    Object.keys(series).forEach((id) => {
      const pts = series[id]
        .map((v, i) => `${xPx(cfg.xOf(i)).toFixed(2)},${yPx(v).toFixed(2)}`)
        .join(" ");
      svg.appendChild(
        el("polyline", {
          points: pts,
          fill: "none",
          stroke: COLORS[id] || "#64748b",
          "stroke-width": 2,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        }),
      );
    });

    // The black left (y) / bottom (x) axes are a CSS-bordered overlay on the
    // plot box (.qc-plot::after in styles.css), which stays crisp at any zoom.
    host.appendChild(svg);

    // All label text should render at the body text size. SVG text scales with
    // the viewBox, so counter that: set the svg font-size to bodyPx / scale,
    // where scale = renderedWidth / PW. Re-run on resize so it always matches.
    const bodyPx = parseFloat(getComputedStyle(document.body).fontSize);
    const syncFontSize = () => {
      const w = host.clientWidth;
      if (!w) return;
      svg.style.fontSize = `${(bodyPx * PW) / w}px`;
    };
    syncFontSize();
    new ResizeObserver(syncFontSize).observe(host);
  });
})();

// ---- Adaptive autoencoder method-pipeline diagram ----
// Builds the stacked token-grid pyramids (8x8 -> 4x4 -> 2x2 -> 1x1). The
// "hierarchical" pyramid keeps every level; the "single" pyramid keeps only
// one level (the rest are drawn empty) to illustrate the random masking step.
(function autoencoderPipeline() {
  const figure = document.getElementById("ae-pipeline");
  if (!figure) return;

  const LEVELS = [8, 4, 2, 1]; // grid sizes, coarsest grid at the bottom
  const KEPT_LEVEL = 2; // the level the "single" pyramid keeps (the 2x2 grid)

  figure.querySelectorAll("[data-tokens]").forEach((container) => {
    const keepAll = container.dataset.tokens === "hierarchical";
    LEVELS.forEach((n, level) => {
      const grid = document.createElement("div");
      grid.className = "ae-tok-grid";
      grid.style.gridTemplateColumns = `repeat(${n}, 10px)`;
      const filled = keepAll || level === KEPT_LEVEL;
      for (let i = 0; i < n * n; i++) {
        const cell = document.createElement("div");
        cell.className = filled ? "ae-tok" : "ae-tok is-empty";
        grid.appendChild(cell);
      }
      container.appendChild(grid);
    });
  });
})();
