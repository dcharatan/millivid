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
