import { useEffect, useMemo, useRef, useState } from "react";

type CommandItem = {
  label: string;
  href: string;
  description: string;
};

const DEFAULT_ITEMS: CommandItem[] = [
  {
    label: "Resume",
    href: "/resume",
    description: "Experience, impact metrics, and downloadable CV."
  },
  {
    label: "Projects",
    href: "/projects",
    description: "Bento overview of ETL, NLP, and computer vision work."
  },
  {
    label: "Contact",
    href: "/contact",
    description: "Email and professional networking channels."
  },
  {
    label: "GitHub",
    href: "https://github.com/danielshort3",
    description: "Code samples and deployed demos."
  }
];

type CommandPaletteProps = {
  items?: CommandItem[];
};

export default function CommandPalette({ items = DEFAULT_ITEMS }: CommandPaletteProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const filteredItems = useMemo(() => {
    const cleaned = query.trim().toLowerCase();
    if (!cleaned) {
      return items;
    }
    return items.filter((item) => {
      return `${item.label} ${item.description}`.toLowerCase().includes(cleaned);
    });
  }, [items, query]);

  useEffect(() => {
    if (isOpen) {
      setQuery("");
      queueMicrotask(() => {
        inputRef.current?.focus();
      });
    }
  }, [isOpen]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const isCmdK = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k";
      if (isCmdK) {
        event.preventDefault();
        setIsOpen((state) => !state);
      }
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    };

    const onOpen = () => setIsOpen(true);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("portfolio:open-command-palette", onOpen);

    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("portfolio:open-command-palette", onOpen);
    };
  }, []);

  return (
    <>
      <button
        type="button"
        className="fixed bottom-6 right-6 z-40 inline-flex items-center gap-2 rounded-full border border-cyan-200/20 bg-gradient-to-r from-[#004d40] to-[#00bcd4] px-4 py-2 text-sm font-semibold text-slate-100 shadow-lg shadow-cyan-500/20 transition hover:brightness-110 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-200"
        aria-label="Open command palette"
        onClick={() => setIsOpen(true)}
      >
        Search
        <span className="rounded bg-slate-900/45 px-2 py-0.5 text-xs">Cmd+K</span>
      </button>

      {isOpen ? (
        <div className="fixed inset-0 z-50 flex items-start justify-center bg-slate-950/75 px-4 pt-[12vh]">
          <div className="w-full max-w-xl rounded-2xl border border-cyan-200/20 bg-deep-panel/95 p-4 shadow-2xl shadow-slate-900/60">
            <div className="mb-3">
              <input
                ref={inputRef}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                type="text"
                placeholder="Search: Resume, Projects, Contact, GitHub"
                className="w-full rounded-xl border border-cyan-200/20 bg-deep-bg/90 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-400 focus:border-cyan-300/70 focus:outline-none"
              />
            </div>

            <ul className="space-y-2">
              {filteredItems.map((item) => (
                <li key={item.label}>
                  <a
                    href={item.href}
                    className="block rounded-xl border border-cyan-200/10 bg-slate-950/30 px-4 py-3 transition hover:border-cyan-300/55 hover:bg-cyan-900/20"
                    onClick={() => setIsOpen(false)}
                  >
                    <p className="font-medium text-slate-100">{item.label}</p>
                    <p className="mt-1 text-sm text-deep-muted">{item.description}</p>
                  </a>
                </li>
              ))}
              {filteredItems.length === 0 ? (
                <li className="rounded-xl border border-cyan-200/10 bg-slate-950/20 px-4 py-5 text-sm text-deep-muted">
                  No match. Try Resume, Projects, Contact, or GitHub.
                </li>
              ) : null}
            </ul>

            <button
              type="button"
              className="mt-4 w-full rounded-xl border border-cyan-200/20 px-3 py-2 text-sm text-deep-text transition hover:border-cyan-300/65"
              onClick={() => setIsOpen(false)}
            >
              Close
            </button>
          </div>
        </div>
      ) : null}
    </>
  );
}
