(function () {
  "use strict";

  const MENU_ID = "cnc-bird-context-menu";
  let currentTarget = null;

  function removeMenu() {
    const existing = document.getElementById(MENU_ID);
    if (existing) {
      existing.remove();
    }
    currentTarget = null;
  }

  function matchingTrigger(action, cameraId, targetId) {
    const candidates = document.querySelectorAll(".cnc-context-trigger");
    for (const candidate of candidates) {
      if (
        candidate.dataset.cncAction === action &&
        candidate.dataset.cameraId === cameraId &&
        candidate.dataset.targetId === targetId
      ) {
        return candidate;
      }
    }
    return null;
  }

  function runAction(action) {
    if (!currentTarget) {
      removeMenu();
      return;
    }

    const cameraId = currentTarget.dataset.cameraId || "";
    const targetId = currentTarget.dataset.targetId || "";
    const trigger = matchingTrigger(action, cameraId, targetId);
    if (trigger) {
      trigger.click();
    }
    removeMenu();
  }

  function menuButton(label, action, detail) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "cnc-context-menu-item";
    button.dataset.action = action;

    const labelNode = document.createElement("span");
    labelNode.className = "cnc-context-menu-label";
    labelNode.textContent = label;
    button.appendChild(labelNode);

    if (detail) {
      const detailNode = document.createElement("span");
      detailNode.className = "cnc-context-menu-detail";
      detailNode.textContent = detail;
      button.appendChild(detailNode);
    }

    button.addEventListener("click", function () {
      runAction(action);
    });
    return button;
  }

  function showMenu(target, clientX, clientY) {
    removeMenu();
    currentTarget = target;

    const menu = document.createElement("div");
    menu.id = MENU_ID;
    menu.className = "cnc-context-menu";
    menu.setAttribute("role", "menu");
    menu.setAttribute("aria-label", "Bird actions");

    const heading = document.createElement("div");
    heading.className = "cnc-context-menu-heading";

    const bird = target.dataset.targetName || "Bird";
    const camera = target.dataset.cameraName || "Camera";
    heading.textContent = bird + " · " + camera;
    menu.appendChild(heading);

    menu.appendChild(
      menuButton("Make active bird", "active", "Select in Active Target")
    );
    menu.appendChild(
      menuButton("Track bird", "track", "Select and follow on plot")
    );
    menu.appendChild(
      menuButton(
        "Select best observation",
        "best",
        "Highest visible probability for this camera"
      )
    );

    document.body.appendChild(menu);

    const padding = 8;
    const rect = menu.getBoundingClientRect();
    const maxX = Math.max(padding, window.innerWidth - rect.width - padding);
    const maxY = Math.max(padding, window.innerHeight - rect.height - padding);
    menu.style.left = Math.max(padding, Math.min(clientX, maxX)) + "px";
    menu.style.top = Math.max(padding, Math.min(clientY, maxY)) + "px";

    const firstButton = menu.querySelector("button");
    if (firstButton) {
      firstButton.focus();
    }
  }

  document.addEventListener("contextmenu", function (event) {
    const target = event.target.closest(".cnc-bird-context-target");
    if (!target) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    showMenu(target, event.clientX, event.clientY);
  });

  document.addEventListener("pointerdown", function (event) {
    const menu = document.getElementById(MENU_ID);
    if (!menu) {
      return;
    }
    if (!menu.contains(event.target)) {
      removeMenu();
    }
  });

  document.addEventListener("keydown", function (event) {
    const menu = document.getElementById(MENU_ID);
    if (!menu) {
      return;
    }

    if (event.key === "Escape") {
      removeMenu();
      return;
    }

    const items = Array.from(
      menu.querySelectorAll(".cnc-context-menu-item")
    );
    const currentIndex = items.indexOf(document.activeElement);

    if (event.key === "ArrowDown") {
      event.preventDefault();
      items[(currentIndex + 1 + items.length) % items.length].focus();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      items[(currentIndex - 1 + items.length) % items.length].focus();
    }
  });

  window.addEventListener("blur", removeMenu);
  window.addEventListener("resize", removeMenu);
  document.addEventListener(
    "scroll",
    function () {
      removeMenu();
    },
    true
  );
})();
