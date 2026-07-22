(() => {
  'use strict';

  const restoreAttribute = (element, name, value) => {
  if (value === null) {
    element.removeAttribute(name);
    return;
  }
  element.setAttribute(name, value);
  };

  const createModalAccessibility = (modal) => {
    const backgroundState = new Map();

  const setOpenState = (isOpen) => {
    modal.hidden = !isOpen;
    modal.inert = !isOpen;
    modal.toggleAttribute('inert', !isOpen);
    modal.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
  };

  const isolateBackground = () => {
    if (backgroundState.size) return;

    let current = modal;
    while (current && current.parentElement) {
      const parent = current.parentElement;
      Array.from(parent.children).forEach((sibling) => {
        if (sibling === current || backgroundState.has(sibling)) return;
        backgroundState.set(sibling, {
          hadInert: sibling.hasAttribute('inert'),
          ariaHidden: sibling.getAttribute('aria-hidden')
        });
        sibling.inert = true;
        sibling.setAttribute('inert', '');
        sibling.setAttribute('aria-hidden', 'true');
      });
      if (parent === document.body) break;
      current = parent;
    }
  };

  const restoreBackground = () => {
    backgroundState.forEach((state, element) => {
      element.inert = state.hadInert;
      element.toggleAttribute('inert', state.hadInert);
      restoreAttribute(element, 'aria-hidden', state.ariaHidden);
    });
    backgroundState.clear();
  };

  setOpenState(modal.classList.contains('active'));

    return {
      hide: () => setOpenState(false),
      isolateBackground,
      restoreBackground,
      show: () => setOpenState(true)
    };
  };

  if (typeof window !== 'undefined') {
    window.createModalAccessibility = createModalAccessibility;
  }
})();
