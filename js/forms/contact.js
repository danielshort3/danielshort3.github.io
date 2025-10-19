(() => {
  'use strict';

  const form = document.getElementById('contact-form');
  if (!form) return;

  const statusNode = document.getElementById('contact-status');
  const submitBtn = form.querySelector('button[type="submit"]');
  const endpoint = form.dataset.endpoint || '';
  const fallbackEmail = form.dataset.mailto || 'danielshort3@gmail.com';
  const subjectLine = form.dataset.subject || 'Portfolio contact request';

  const validators = {
    name: value => value.trim().length >= 2 ? '' : 'Please share your name so I know how to address you.',
    email: value => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value.trim()) ? '' : 'Enter a valid email address.',
    message: value => value.trim().length >= 20 ? '' : 'Tell me a bit more about your project (20 characters minimum).'
  };

  const setError = (input, message) => {
    const field = input.closest('.form-field');
    const helper = field ? field.querySelector('.input-helper') : null;
    if (!helper) return;
    if (message) {
      helper.textContent = message;
      input.setAttribute('aria-invalid', 'true');
    } else {
      helper.textContent = '';
      input.removeAttribute('aria-invalid');
    }
  };

  const validateField = input => {
    const rule = validators[input.name];
    if (!rule) return true;
    const message = rule(input.value);
    setError(input, message);
    return !message;
  };

  form.querySelectorAll('input, textarea').forEach(control => {
    control.addEventListener('blur', () => validateField(control));
    control.addEventListener('input', () => {
      if (control.hasAttribute('aria-invalid')) validateField(control);
    });
  });

  const setStatus = (text, tone = 'neutral') => {
    if (!statusNode) return;
    statusNode.textContent = text;
    statusNode.dataset.tone = tone;
  };

  const serializeForm = () => {
    const data = {};
    new FormData(form).forEach((value, key) => { data[key] = value.trim(); });
    return data;
  };

  const disableForm = disabled => {
    if (submitBtn) submitBtn.disabled = disabled;
    form.querySelectorAll('input, textarea, button').forEach(el => {
      el.disabled = disabled;
    });
  };

  const submitViaEndpoint = async data => {
    if (!endpoint) return { ok: false, reason: 'no-endpoint' };
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          name: data.name,
          email: data.email,
          message: data.message,
          _subject: subjectLine
        })
      });
      if (!response.ok) return { ok: false, reason: `status-${response.status}` };
      return { ok: true };
    } catch (error) {
      return { ok: false, reason: error.message || 'network-error' };
    }
  };

  const fallbackMailto = data => {
    const body = encodeURIComponent(`Name: ${data.name}\nEmail: ${data.email}\n\n${data.message}`);
    const mailto = `mailto:${fallbackEmail}?subject=${encodeURIComponent(subjectLine)}&body=${body}`;
    window.open(mailto, '_blank', 'noopener');
  };

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const controls = Array.from(form.querySelectorAll('input, textarea'));
    const allValid = controls.every(validateField);
    if (!allValid) {
      setStatus('Please fix the highlighted fields and resubmit.', 'error');
      const firstInvalid = controls.find(ctrl => ctrl.getAttribute('aria-invalid') === 'true');
      firstInvalid && firstInvalid.focus();
      return;
    }

    const data = serializeForm();
    disableForm(true);
    setStatus('Sending…', 'info');

    const result = await submitViaEndpoint(data);
    if (result.ok) {
      setStatus('Thanks! Your message is on its way. I will reply within two business days.', 'success');
      form.reset();
      window.gaEvent && window.gaEvent('submit_contact_form', { transport: 'api', status: 'success' });
    } else {
      fallbackMailto(data);
      setStatus('I opened your email client so you can send the message directly. If that failed, reach me at danielshort3@gmail.com.', 'info');
      window.gaEvent && window.gaEvent('submit_contact_form', { transport: 'mailto', status: 'fallback', reason: result.reason });
    }

    disableForm(false);
  });
})();
