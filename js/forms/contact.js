(() => {
  'use strict';

  const form = document.getElementById('contactForm');
  if (!form) return;

  const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const endpoint = 'https://formsubmit.co/ajax/danielshort3@gmail.com';
  const feedback = form.querySelector('.form-feedback');
  const submitBtn = form.querySelector('button[type="submit"]');

  const fields = {
    name: {
      input: form.querySelector('#contact-name'),
      error: form.querySelector('#contact-name-error'),
      validate(value) {
        if (!value.trim()) return 'Please enter your name.';
        return '';
      }
    },
    email: {
      input: form.querySelector('#contact-email'),
      error: form.querySelector('#contact-email-error'),
      validate(value) {
        if (!value.trim()) return 'Please enter your email address.';
        if (!emailPattern.test(value.trim())) return 'Enter a valid email address.';
        return '';
      }
    },
    message: {
      input: form.querySelector('#contact-message'),
      error: form.querySelector('#contact-message-error'),
      validate(value) {
        if (!value.trim()) return 'Let me know how I can help.';
        if (value.trim().length < 15) return 'Add a little more detail (at least 15 characters).';
        return '';
      }
    }
  };

  function setFeedback(message, type = 'neutral') {
    if (!feedback) return;
    feedback.textContent = message;
    feedback.classList.remove('is-success', 'is-error');
    if (type === 'success') feedback.classList.add('is-success');
    if (type === 'error') feedback.classList.add('is-error');
  }

  function validateField(field) {
    const value = field.input.value;
    const message = field.validate(value);
    field.input.classList.toggle('is-invalid', Boolean(message));
    if (field.error) field.error.textContent = message;
    return !message;
  }

  Object.values(fields).forEach(field => {
    field.input.addEventListener('input', () => {
      if (field.input.classList.contains('is-invalid')) {
        validateField(field);
      }
    });
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    setFeedback('');

    const validity = Object.values(fields).map(validateField);
    if (validity.some(valid => !valid)) {
      const firstInvalid = Object.values(fields).find(f => f.input.classList.contains('is-invalid'));
      firstInvalid?.input.focus();
      return;
    }

    const payload = {
      name: fields.name.input.value.trim(),
      email: fields.email.input.value.trim(),
      message: fields.message.input.value.trim(),
      _subject: 'danielshort.me contact request',
      _captcha: 'false'
    };

    submitBtn.disabled = true;
    submitBtn.textContent = 'Sending…';

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error('Network error');

      setFeedback('Thank you! Your message is on its way.', 'success');
      form.reset();
      Object.values(fields).forEach(field => field.input.classList.remove('is-invalid'));
      if (typeof window.gaEvent === 'function') {
        window.gaEvent('contact_form_submit', { status: 'success' });
      }
    } catch (err) {
      console.warn('Contact form submission failed', err);
      setFeedback('Something went wrong. Email me directly at danielshort3@gmail.com.', 'error');
      if (typeof window.gaEvent === 'function') {
        window.gaEvent('contact_form_submit', { status: 'error' });
      }
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Send Message';
    }
  });
})();
