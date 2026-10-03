/* Shared contact form markup for generated inline pages and lazy-loaded dialogs. */
(function (root, factory) {
  const markup = factory();
  if (typeof module === 'object' && module.exports) module.exports = markup;
  if (typeof window !== 'undefined') root.SiteContactMarkup = markup;
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  const formBody = `
    <form id="contact-form" class="contact-form" method="post" action="/api/contact" data-endpoint="/api/contact" novalidate>
      <div id="contact-errors" class="contact-error-summary" tabindex="-1" role="alert" hidden>
        <h3>Check the following</h3>
        <ul data-contact-error-list></ul>
      </div>
      <div class="form-field">
        <label for="contact-name">Name <span class="contact-required">(required)</span></label>
        <input id="contact-name" name="name" type="text" autocomplete="name" required maxlength="200" aria-describedby="contact-name-required">
        <p class="contact-field-error" id="contact-name-required" hidden></p>
      </div>
      <div class="form-field">
        <label for="contact-email">Email <span class="contact-required">(required)</span></label>
        <input id="contact-email" name="email" type="email" autocomplete="email" required maxlength="254" aria-describedby="contact-email-required">
        <p class="contact-field-error" id="contact-email-required" hidden></p>
      </div>
      <div class="form-field">
        <label for="contact-message">Message <span class="contact-required">(required)</span></label>
        <textarea id="contact-message" name="message" rows="5" maxlength="4000" required aria-describedby="contact-message-required contact-form-note"></textarea>
        <p class="contact-field-error" id="contact-message-required" hidden></p>
      </div>
      <p id="contact-form-note" class="contact-form-note">Used to reply to your message. <a href="/privacy">Privacy details</a></p>
      <div class="form-field honeypot" aria-hidden="true">
        <label for="contact-company">Company</label>
        <input id="contact-company" name="company" type="text" tabindex="-1" autocomplete="off">
      </div>
      <p id="contact-status" class="contact-form-status" role="status" aria-live="polite" tabindex="-1"></p>
      <div id="contact-alt" class="contact-form-alt" hidden>
        <a href="mailto:daniel@danielshort.me" class="btn-ghost">Email me directly</a>
      </div>
      <div class="form-actions">
        <button type="submit" class="btn-primary"><span class="btn-spinner" aria-hidden="true"></span><span class="btn-label">Send message</span></button>
      </div>
    </form>
    <div class="contact-form-success" id="contact-success" hidden tabindex="-1" role="status" aria-live="polite">
      <span class="success-icon" aria-hidden="true"></span>
      <h3>Message sent</h3>
      <p>Thanks for reaching out. Your message was sent. I’ll reply when I can.</p>
      <div class="form-actions">
        <button type="button" class="btn-primary" data-contact-new>Start another message</button>
        <a href="mailto:daniel@danielshort.me" class="btn-secondary">Email me directly</a>
      </div>
    </div>`;

  const render = ({ inline = false } = {}) => inline
    ? `<section id="contact-inline" class="contact-inline" data-contact-inline aria-labelledby="contact-inline-title"><h2 id="contact-inline-title">Send a message</h2><div data-contact-form-body>${formBody}</div></section>`
    : `<div id="contact-modal" class="modal"><div class="modal-content" role="dialog" aria-modal="true" tabindex="0" aria-labelledby="contact-modal-title"><button type="button" class="modal-close" aria-label="Close dialog">&times;</button><div class="modal-title-strip"><h3 class="modal-title" id="contact-modal-title">Send a Message</h3></div><div class="modal-body" data-contact-form-body>${formBody}</div></div></div>`;

  return Object.freeze({ render });
});
