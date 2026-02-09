(() => {
  'use strict';

  const isHome = document.body?.dataset?.page === 'home';
  const isMissionHome = document.body?.classList?.contains('mission-control-home');
  if (!isHome || !isMissionHome) return;

  const stackPills = [...document.querySelectorAll('[data-stack-skill]')];
  const projectCards = [...document.querySelectorAll('[data-project-card]')];
  const cursor = document.querySelector('.mission-cursor');
  const interactiveNodes = [
    ...document.querySelectorAll('.mission-control-home a'),
    ...document.querySelectorAll('.mission-control-home button')
  ];

  const normalize = (value) => String(value || '').trim().toLowerCase();

  const projectSkillMap = new Map();
  projectCards.forEach((card) => {
    const rawSkills = String(card.getAttribute('data-project-skills') || '');
    const skills = rawSkills
      .split(',')
      .map((entry) => normalize(entry))
      .filter(Boolean);
    projectSkillMap.set(card, skills);
  });

  const clearSkillState = () => {
    stackPills.forEach((pill) => pill.classList.remove('is-active'));
    projectCards.forEach((card) => {
      card.classList.remove('is-dimmed');
      card.classList.remove('is-linked');
    });
  };

  const applySkillState = (skill) => {
    const selectedSkill = normalize(skill);
    if (!selectedSkill) return;

    let linkedCount = 0;
    projectCards.forEach((card) => {
      const skills = projectSkillMap.get(card) || [];
      const isLinked = skills.includes(selectedSkill);
      card.classList.toggle('is-linked', isLinked);
      card.classList.toggle('is-dimmed', !isLinked);
      if (isLinked) linkedCount += 1;
    });

    if (!linkedCount) {
      projectCards.forEach((card) => {
        card.classList.remove('is-linked');
        card.classList.remove('is-dimmed');
      });
    }
  };

  stackPills.forEach((pill) => {
    const skill = normalize(pill.getAttribute('data-stack-skill'));
    if (!skill) return;

    const activate = () => {
      stackPills.forEach((node) => node.classList.toggle('is-active', node === pill));
      applySkillState(skill);
    };

    const deactivate = () => {
      if (document.activeElement === pill) return;
      clearSkillState();
    };

    pill.addEventListener('pointerenter', activate);
    pill.addEventListener('focus', activate);
    pill.addEventListener('pointerleave', deactivate);
    pill.addEventListener('blur', deactivate);
  });

  if (!cursor || !window.matchMedia('(pointer: fine)').matches) return;

  let rafId = null;
  let nextX = -120;
  let nextY = -120;

  const moveCursor = () => {
    cursor.style.left = `${nextX}px`;
    cursor.style.top = `${nextY}px`;
    rafId = null;
  };

  const onPointerMove = (event) => {
    nextX = event.clientX;
    nextY = event.clientY;
    if (rafId === null) {
      rafId = window.requestAnimationFrame(moveCursor);
    }
  };

  document.addEventListener('pointermove', onPointerMove, { passive: true });

  document.addEventListener('pointerleave', () => {
    cursor.classList.remove('is-visible');
    cursor.classList.remove('is-expanded');
  });

  interactiveNodes.forEach((node) => {
    if (node.dataset.cursorBound === 'true') return;
    node.dataset.cursorBound = 'true';

    const show = () => {
      cursor.classList.add('is-visible');
    };

    const hide = () => {
      cursor.classList.remove('is-visible');
      cursor.classList.remove('is-expanded');
    };

    const expand = () => {
      cursor.classList.add('is-visible');
      if (node.matches('[data-project-card]')) {
        cursor.classList.add('is-expanded');
      } else {
        cursor.classList.remove('is-expanded');
      }
    };

    node.addEventListener('pointerenter', expand);
    node.addEventListener('focus', show);
    node.addEventListener('pointerleave', hide);
    node.addEventListener('blur', hide);
  });
})();
