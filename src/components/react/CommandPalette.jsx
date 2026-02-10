import { useEffect, useMemo, useState } from 'react';

const NAV_ITEMS = [
  { label: 'Resume', hint: 'PDF + interactive', href: '/resume.html' },
  { label: 'Projects', hint: 'STAR-M project grid', href: '/projects' },
  { label: 'Contact', hint: 'Get in touch', href: '/contact.html' },
  { label: 'GitHub', hint: 'Source + experiments', href: 'https://github.com/danielshort3', external: true }
];

export default function CommandPalette() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');

  useEffect(() => {
    const onKeyDown = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setIsOpen((value) => !value);
      }

      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  useEffect(() => {
    if (!isOpen) {
      setQuery('');
    }
  }, [isOpen]);

  const results = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return NAV_ITEMS;
    }

    return NAV_ITEMS.filter((item) => {
      return `${item.label} ${item.hint}`.toLowerCase().includes(normalized);
    });
  }, [query]);

  const onSelect = (item) => {
    if (item.external) {
      window.open(item.href, '_blank', 'noopener,noreferrer');
    } else {
      window.location.assign(item.href);
    }
    setIsOpen(false);
  };

  return (
    <>
      <button
        type="button"
        aria-label="Open command palette"
        className="accent-button fixed bottom-4 right-4 z-50 gap-2 text-xs sm:text-sm"
        onClick={() => setIsOpen(true)}
      >
        <span>Open</span>
        <kbd className="rounded-md bg-slate-900/70 px-2 py-1 text-cyan-200">Cmd/Ctrl + K</kbd>
      </button>

      {isOpen && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center bg-slate-950/70 px-4 pt-24"
          onClick={() => setIsOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-label="Command palette"
        >
          <div
            className="w-full max-w-2xl rounded-2xl border border-cyan-300/30 bg-slate-900 p-4 shadow-glow"
            onClick={(event) => event.stopPropagation()}
          >
            <input
              type="text"
              autoFocus
              placeholder="Search navigation..."
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              className="w-full rounded-xl border border-cyan-300/25 bg-slate-950 px-4 py-3 text-cyan-50 outline-none ring-cyan-300/50 focus:ring"
            />

            <ul className="mt-4 space-y-2">
              {results.length === 0 && (
                <li className="rounded-lg border border-cyan-300/20 p-3 text-sm text-cyan-100/70">
                  No matching command.
                </li>
              )}
              {results.map((item) => (
                <li key={item.label}>
                  <button
                    type="button"
                    className="flex w-full items-start justify-between rounded-lg border border-cyan-300/20 px-3 py-3 text-left transition hover:border-cyan-200/60 hover:bg-cyan-400/10"
                    onClick={() => onSelect(item)}
                  >
                    <span className="font-semibold text-cyan-50">{item.label}</span>
                    <span className="text-xs text-cyan-100/70">{item.hint}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </>
  );
}
