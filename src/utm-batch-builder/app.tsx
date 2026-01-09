import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const core = require("./core");

type InputMode = "single" | "list" | "csvColumn";
type CombinationMode = "cartesian" | "zip" | "templateRows" | "groups";
type SpacesMode = "underscore" | "dash" | "none";

type FieldInputState = {
  mode: InputMode;
  single: string;
  list: string;
  csvColumnIndex: number | null;
};

type NormalizationState = {
  lowercase: boolean;
  spaces: SpacesMode;
  stripSpecial: boolean;
  slugify: boolean;
};

type CustomParamState = {
  id: string;
  key: string;
  value: FieldInputState;
};

type RelationshipGroupState = {
  id: string;
  name: string;
  keys: string[];
};

type AppConfigState = {
  mode: CombinationMode;
  overrideExisting: boolean;
  normalization: NormalizationState;
  excludeRulesText: string;
  csvText: string;
  previewLimit: number;
  maxRows: number;
  relationshipGroups: RelationshipGroupState[];
  baseUrl: FieldInputState;
  utm: {
    source: FieldInputState;
    medium: FieldInputState;
    campaign: FieldInputState;
    content: FieldInputState;
    term: FieldInputState;
  };
  customParams: CustomParamState[];
};

type CampaignBuilderMode = "cartesian" | "zip";

type CampaignBuilderState = {
  template: string;
  mode: CampaignBuilderMode;
  tokens: Record<string, FieldInputState>;
  generated: string[];
  lastError: string | null;
};

type CsvColumnOption = { index: number; label: string };

type WorkerResponse =
  | { type: "start"; requestId: string; estimatedTotal: number; paramKeys: string[]; warnings: string[] }
  | { type: "chunk"; requestId: string; rows: any[]; generatedCount: number }
  | { type: "done"; requestId: string; generatedCount: number }
  | { type: "cancelled"; requestId: string; generatedCount: number }
  | { type: "error"; requestId: string; errors: string[]; warnings: string[] };

const STORAGE_PRESETS_KEY = "utm-batch-builder.presets.v1";
const STORAGE_LAST_KEY = "utm-batch-builder.last.v1";

const makeId = () => {
  const c = (globalThis as any).crypto;
  if (c && typeof c.randomUUID === "function") return c.randomUUID();
  return `${Math.random().toString(16).slice(2)}${Date.now().toString(16)}`;
};

const makeField = (overrides: Partial<FieldInputState> = {}): FieldInputState => ({
  mode: "single",
  single: "",
  list: "",
  csvColumnIndex: null,
  ...overrides,
});

const defaultConfig = (): AppConfigState => ({
  mode: "cartesian",
  overrideExisting: true,
  normalization: {
    lowercase: true,
    spaces: "underscore",
    stripSpecial: false,
    slugify: false,
  },
  excludeRulesText: "",
  csvText: "",
  previewLimit: 10,
  maxRows: 50000,
  relationshipGroups: [],
  baseUrl: makeField({
    mode: "list",
    single: "",
    list: "",
  }),
  utm: {
    source: makeField({ mode: "list" }),
    medium: makeField({ mode: "single", single: "cpc" }),
    campaign: makeField({ mode: "single" }),
    content: makeField({ mode: "single" }),
    term: makeField({ mode: "single" }),
  },
  customParams: [],
});

const defaultCampaignBuilder = (): CampaignBuilderState => ({
  template: "{initiative}_{market}_{channel}_{objective}_{flight}_{creative}",
  mode: "cartesian",
  tokens: {},
  generated: [],
  lastError: null,
});

const safeJsonParse = (raw: string | null) => {
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch (_) {
    return null;
  }
};

const stripCsvForStorage = (config: AppConfigState) => ({
  ...config,
  csvText: "",
});

const escapeCsvCell = (value: string) => {
  const v = String(value ?? "");
  if (/[",\r\n]/.test(v)) return `"${v.replace(/"/g, "\"\"")}"`;
  return v;
};

const buildCsv = (rows: any[], paramKeys: string[]) => {
  const headers = ["base_url", ...paramKeys, "final_url"];
  const lines = [headers.map(escapeCsvCell).join(",")];
  rows.forEach((row) => {
    const baseUrl = String(row?.baseUrl ?? "");
    const params = row?.params || {};
    const finalUrl = String(row?.finalUrl ?? "");
    const cells = [baseUrl, ...paramKeys.map(k => String(params[k] ?? "")), finalUrl];
    lines.push(cells.map(escapeCsvCell).join(","));
  });
  return lines.join("\n");
};

const downloadTextFile = (filename: string, text: string, mime = "text/plain;charset=utf-8") => {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
};

const copyToClipboard = async (text: string) => {
  const value = String(text ?? "");
  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch (_) {}

  try {
    const ta = document.createElement("textarea");
    ta.value = value;
    ta.setAttribute("readonly", "true");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    ta.style.top = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  } catch (_) {
    return false;
  }
};

const extractTemplateTokens = (template: string) => {
  const tokens: string[] = [];
  const seen = new Set<string>();
  const re = /\{([^}]+)\}/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(template || "")) !== null) {
    const token = String(m[1] || "").trim();
    if (!token) continue;
    if (seen.has(token)) continue;
    seen.add(token);
    tokens.push(token);
  }
  return tokens;
};

const applyTemplate = (template: string, valuesByToken: Record<string, string>) => {
  return String(template || "").replace(/\{([^}]+)\}/g, (_m, tokenName) => {
    const key = String(tokenName || "").trim();
    return valuesByToken[key] ?? "";
  });
};

const FieldEditor = ({
  label,
  required,
  field,
  onChange,
  csvColumns,
  combinationMode,
  placeholder,
  help,
}: {
  label: string;
  required?: boolean;
  field: FieldInputState;
  onChange: (next: FieldInputState) => void;
  csvColumns: CsvColumnOption[];
  combinationMode: CombinationMode;
  placeholder?: string;
  help?: string;
}) => {
  const hasCsv = csvColumns.length > 0;
  const isTemplateRows = combinationMode === "templateRows";
  const [isDropOver, setIsDropOver] = useState(false);

  const readDroppedCsvColumnIndex = (e: React.DragEvent) => {
    if (!hasCsv) return null;
    const raw = String(e.dataTransfer.getData("application/x-utmtool-csvcol") || "").trim();
    const index = Number(raw);
    if (!Number.isFinite(index)) return null;
    const exists = csvColumns.some((c) => c.index === index);
    if (!exists) return null;
    return index;
  };

  const setMode = (mode: InputMode) => {
    if (mode === "csvColumn" && field.csvColumnIndex === null) {
      onChange({ ...field, mode, csvColumnIndex: csvColumns[0]?.index ?? 0 });
      return;
    }
    onChange({ ...field, mode });
  };

  const modeOptions = isTemplateRows
    ? [
        { value: "single", label: "None" },
        { value: "list", label: "List (one per row)" },
        { value: "csvColumn", label: "From CSV column" },
      ]
    : [
        { value: "single", label: "Single value" },
        { value: "list", label: "List (newline-separated)" },
        { value: "csvColumn", label: "From CSV column" },
      ];

  return (
    <div
      className={`utmtool-field${isDropOver ? " is-drop-over" : ""}`}
      onDragOver={(e) => {
        if (!hasCsv) return;
        const types = Array.from(e.dataTransfer.types || []);
        if (!types.includes("application/x-utmtool-csvcol")) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
        setIsDropOver(true);
      }}
      onDragLeave={() => setIsDropOver(false)}
      onDrop={(e) => {
        const index = readDroppedCsvColumnIndex(e);
        if (index === null) return;
        e.preventDefault();
        setIsDropOver(false);
        onChange({ ...field, mode: "csvColumn", csvColumnIndex: index });
      }}
    >
      <div className="utmtool-field-head">
        <label className="utmtool-label">
          {label}
          {required ? <span className="utmtool-required">Required</span> : null}
        </label>
        <div className="utmtool-mode">
          <span className="utmtool-mode-label">{isTemplateRows ? "Overrides" : "Mode"}</span>
          <select
            className="utmtool-select"
            value={field.mode}
            onChange={(e) => setMode(e.target.value as InputMode)}
          >
            {modeOptions.map((opt) => (
              <option
                key={opt.value}
                value={opt.value}
                disabled={opt.value === "csvColumn" && !hasCsv}
              >
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {isTemplateRows ? (
        <div className="utmtool-template-grid">
          <div className="utmtool-template-default">
            <span className="utmtool-sub-label">Default</span>
            <input
              className="utmtool-input"
              type="text"
              value={field.single}
              placeholder={placeholder}
              onChange={(e) => onChange({ ...field, single: e.target.value })}
            />
          </div>

          {field.mode === "list" ? (
            <div className="utmtool-template-overrides">
              <span className="utmtool-sub-label">Overrides (blank = default)</span>
              <textarea
                className="utmtool-textarea"
                value={field.list}
                rows={4}
                placeholder={"row1\nrow2\n\nrow4"}
                onChange={(e) => onChange({ ...field, list: e.target.value })}
              />
            </div>
          ) : null}

          {field.mode === "csvColumn" ? (
            <div className="utmtool-template-overrides">
              <span className="utmtool-sub-label">Overrides column (blank = default)</span>
              <select
                className="utmtool-select"
                value={field.csvColumnIndex ?? ""}
                onChange={(e) =>
                  onChange({ ...field, csvColumnIndex: Number(e.target.value) })
                }
              >
                {csvColumns.map((c) => (
                  <option key={c.index} value={c.index}>
                    {c.label}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="utmtool-value">
          {field.mode === "single" ? (
            <input
              className="utmtool-input"
              type="text"
              value={field.single}
              placeholder={placeholder}
              onChange={(e) => onChange({ ...field, single: e.target.value })}
            />
          ) : null}

          {field.mode === "list" ? (
            <textarea
              className="utmtool-textarea"
              value={field.list}
              rows={4}
              placeholder={placeholder || "value1\nvalue2\nvalue3"}
              onChange={(e) => onChange({ ...field, list: e.target.value })}
            />
          ) : null}

          {field.mode === "csvColumn" ? (
            <select
              className="utmtool-select"
              value={field.csvColumnIndex ?? ""}
              onChange={(e) => onChange({ ...field, csvColumnIndex: Number(e.target.value) })}
            >
              {csvColumns.map((c) => (
                <option key={c.index} value={c.index}>
                  {c.label}
                </option>
              ))}
            </select>
          ) : null}
        </div>
      )}

      {help ? <p className="utmtool-help">{help}</p> : null}
      {(combinationMode === "zip" || combinationMode === "groups") && field.mode === "list" ? (
        <p className="utmtool-help">In row-aligned modes, blank lines keep row alignment.</p>
      ) : null}
    </div>
  );
};

const VirtualizedTable = ({
  rows,
  paramKeys,
  filterQuery,
  onCopyRow,
}: {
  rows: any[];
  paramKeys: string[];
  filterQuery: string;
  onCopyRow: (row: any) => void;
}) => {
  const query = String(filterQuery || "").trim().toLowerCase();
  const filtered = useMemo(() => {
    if (!query) return rows;
    return rows.filter((row) => {
      const baseUrl = String(row?.baseUrl || "").toLowerCase();
      const finalUrl = String(row?.finalUrl || "").toLowerCase();
      if (baseUrl.includes(query) || finalUrl.includes(query)) return true;
      const params = row?.params || {};
      return paramKeys.some((k) => String(params[k] || "").toLowerCase().includes(query));
    });
  }, [rows, query, paramKeys]);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const headerRef = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [headerHeight, setHeaderHeight] = useState(0);
  const rowHeight = 44;
  const height = 520;
  const overscan = 8;

  useEffect(() => {
    const h = headerRef.current?.offsetHeight || 0;
    setHeaderHeight(h);
  }, [paramKeys.length]);

  const totalHeight = filtered.length * rowHeight;
  const effectiveScrollTop = Math.max(0, scrollTop - headerHeight);
  const viewportHeight = Math.max(0, height - headerHeight);
  const startIndex = Math.max(0, Math.floor(effectiveScrollTop / rowHeight) - overscan);
  const visibleCount = Math.ceil(viewportHeight / rowHeight) + overscan * 2;
  const endIndex = Math.min(filtered.length, startIndex + visibleCount);
  const offsetY = startIndex * rowHeight;
  const visible = filtered.slice(startIndex, endIndex);

  const gridTemplate = useMemo(() => {
    const base = "260px";
    const params = paramKeys.map(() => "160px").join(" ");
    const finalUrl = "minmax(420px,1fr)";
    const actions = "120px";
    return [base, params, finalUrl, actions].filter(Boolean).join(" ");
  }, [paramKeys]);

  return (
    <div className="utmtool-results">
      <div className="utmtool-table-scroll" style={{ height }} onScroll={(e) => setScrollTop((e.target as HTMLDivElement).scrollTop)} ref={containerRef}>
        <div className="utmtool-table-header" style={{ gridTemplateColumns: gridTemplate }} ref={headerRef}>
          <div className="utmtool-th">base_url</div>
          {paramKeys.map((k) => (
            <div key={k} className="utmtool-th">
              {k}
            </div>
          ))}
          <div className="utmtool-th">final_url</div>
          <div className="utmtool-th">actions</div>
        </div>

        <div className="utmtool-table-body" style={{ height: totalHeight }}>
          <div className="utmtool-table-rows" style={{ transform: `translateY(${offsetY}px)` }}>
            {visible.map((row, idx) => {
              const params = row?.params || {};
              return (
                <div
                  key={`${startIndex + idx}-${row?.finalUrl || ""}`}
                  className="utmtool-tr"
                  style={{ gridTemplateColumns: gridTemplate, height: rowHeight }}
                >
                  <div className="utmtool-td utmtool-cell-mono">{String(row?.baseUrl || "")}</div>
                  {paramKeys.map((k) => (
                    <div key={k} className="utmtool-td utmtool-cell-mono">
                      {String(params[k] || "")}
                    </div>
                  ))}
                  <div className="utmtool-td utmtool-cell-mono utmtool-final-url">
                    {String(row?.finalUrl || "")}
                  </div>
                  <div className="utmtool-td utmtool-actions">
                    <button type="button" className="btn-secondary utmtool-copy-btn" onClick={() => onCopyRow(row)}>
                      Copy
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <p className="utmtool-help">Showing {filtered.length.toLocaleString("en-US")} rows.</p>
    </div>
  );
};

const AccordionSection = ({
  title,
  subtitle,
  defaultOpen = true,
  children,
}: {
  title: string;
  subtitle?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) => {
  return (
    <details className="utmtool-accordion" open={defaultOpen}>
      <summary className="utmtool-accordion-summary">
        <div className="utmtool-accordion-text">
          <div className="utmtool-accordion-title">{title}</div>
          {subtitle ? <div className="utmtool-accordion-subtitle">{subtitle}</div> : null}
        </div>
        <span className="utmtool-accordion-chevron" aria-hidden="true"></span>
      </summary>
      <div className="utmtool-accordion-body">{children}</div>
    </details>
  );
};

const MODE_HELP: Record<CombinationMode, { label: string; description: string }> = {
  cartesian: {
    label: "A) Full Cartesian",
    description: "All combinations of every list value across fields.",
  },
  zip: {
    label: "B) Zip by Row",
    description: "Row-aligned lists (line 1 with line 1, etc.).",
  },
  templateRows: {
    label: "C) Template + Rows",
    description: "Defaults + per-row overrides (blank override falls back to default).",
  },
  groups: {
    label: "D) Grouped (Relationships)",
    description: "Zip within groups, Cartesian across groups (drag & drop).",
  },
};

const ModePicker = ({
  value,
  onChange,
}: {
  value: CombinationMode;
  onChange: (next: CombinationMode) => void;
}) => {
  const modes: CombinationMode[] = ["cartesian", "zip", "templateRows", "groups"];
  return (
    <fieldset className="utmtool-mode-picker">
      <legend className="utmtool-mode-legend">Combination mode</legend>
      <div className="utmtool-mode-grid">
        {modes.map((mode) => (
          <label
            key={mode}
            className={`utmtool-mode-card${value === mode ? " is-selected" : ""}`}
          >
            <input
              type="radio"
              name="utmtool-mode"
              value={mode}
              checked={value === mode}
              onChange={() => onChange(mode)}
            />
            <div className="utmtool-mode-title">{MODE_HELP[mode].label}</div>
            <div className="utmtool-mode-desc">{MODE_HELP[mode].description}</div>
          </label>
        ))}
      </div>
    </fieldset>
  );
};

const RelationshipBuilder = ({
  keys,
  keyInputs,
  requiredKeys,
  csvRowCount,
  csvColumns,
  groups,
  onChangeGroups,
  onChangeKeyInput,
  onAddCustomParamKey,
}: {
  keys: string[];
  keyInputs: Record<string, FieldInputState | undefined>;
  requiredKeys: Set<string>;
  csvRowCount: number;
  csvColumns: CsvColumnOption[];
  groups: RelationshipGroupState[];
  onChangeGroups: (next: RelationshipGroupState[]) => void;
  onChangeKeyInput: (key: string, next: FieldInputState) => void;
  onAddCustomParamKey?: (key: string) => string | null;
}) => {
  const availableKeys = useMemo(() => keys.filter(Boolean), [keys]);
  const [selectedKey, setSelectedKey] = useState<string>("base_url");
  const [newCustomKey, setNewCustomKey] = useState("");
  const [newCustomKeyError, setNewCustomKeyError] = useState<string | null>(null);

  const sanitizeGroups = useCallback((next: RelationshipGroupState[]) => {
    const allowed = new Set(availableKeys);
    const seen = new Set<string>();
    return (next || []).map((g) => {
      const deduped: string[] = [];
      (g.keys || []).forEach((k) => {
        const key = String(k || "").trim();
        if (!key) return;
        if (!allowed.has(key)) return;
        if (seen.has(key)) return;
        seen.add(key);
        deduped.push(key);
      });
      return { ...g, keys: deduped };
    });
  }, [availableKeys]);

  const normalizedGroups = useMemo(() => sanitizeGroups(groups || []), [groups, sanitizeGroups]);

  const assigned = useMemo(() => {
    const set = new Set<string>();
    normalizedGroups.forEach((g) => (g.keys || []).forEach((k) => set.add(k)));
    return set;
  }, [normalizedGroups]);

  const independentKeys = useMemo(() => availableKeys.filter((k) => !assigned.has(k)), [availableKeys, assigned]);

  const getKeyLabel = (key: string) => (key === "base_url" ? "Base URL" : key);

  const parseList = (text: string) => core.parseMultilinePreserveEmpty(text || "");

  const getKeyStats = (key: string) => {
    const field = keyInputs[key];
    if (!field) return { length: 0, nonEmpty: 0, missing: requiredKeys.has(key) };
    if (field.mode === "csvColumn") {
      const len = csvRowCount;
      return { length: len, nonEmpty: len, missing: requiredKeys.has(key) && len === 0 };
    }
    if (field.mode === "list") {
      const rows = parseList(field.list);
      const nonEmpty = rows.filter((v: string) => String(v || "").trim() !== "").length;
      const missing = requiredKeys.has(key) && nonEmpty === 0;
      return { length: rows.length, nonEmpty, missing };
    }
    const value = String(field.single || "").trim();
    const missing = requiredKeys.has(key) && !value;
    return { length: 1, nonEmpty: value ? 1 : 0, missing };
  };

  useEffect(() => {
    if (availableKeys.includes(selectedKey)) return;
    setSelectedKey(availableKeys.includes("base_url") ? "base_url" : (availableKeys[0] || ""));
  }, [availableKeys, selectedKey]);

  const moveKeyToGroup = (key: string, groupId: string | null) => {
    const next = normalizedGroups.map((g) => ({ ...g, keys: (g.keys || []).filter((k) => k !== key) }));
    if (groupId) {
      const idx = next.findIndex((g) => g.id === groupId);
      if (idx >= 0) next[idx] = { ...next[idx], keys: (next[idx].keys || []).concat(key) };
    }
    onChangeGroups(sanitizeGroups(next));
  };

  const addGroup = () => {
    const name = `Zip group ${normalizedGroups.length + 1}`;
    onChangeGroups(sanitizeGroups(normalizedGroups.concat({ id: makeId(), name, keys: [] })));
  };

  const removeGroup = (groupId: string) => {
    onChangeGroups(sanitizeGroups(normalizedGroups.filter((g) => g.id !== groupId)));
  };

  const renameGroup = (groupId: string, name: string) => {
    onChangeGroups(sanitizeGroups(normalizedGroups.map((g) => (g.id === groupId ? { ...g, name } : g))));
  };

  const moveGroup = (groupId: string, dir: -1 | 1) => {
    const idx = normalizedGroups.findIndex((g) => g.id === groupId);
    if (idx < 0) return;
    const nextIdx = idx + dir;
    if (nextIdx < 0 || nextIdx >= normalizedGroups.length) return;
    const next = normalizedGroups.slice();
    const temp = next[idx];
    next[idx] = next[nextIdx];
    next[nextIdx] = temp;
    onChangeGroups(sanitizeGroups(next));
  };

  const onDragStartKey = (key: string) => (e: React.DragEvent) => {
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("application/x-utmtool-key", key);
    e.dataTransfer.setData("text/plain", key);
  };

  const onDropToIndependent = (e: React.DragEvent) => {
    e.preventDefault();
    const key = String(e.dataTransfer.getData("application/x-utmtool-key") || e.dataTransfer.getData("text/plain") || "").trim();
    if (!key) return;
    moveKeyToGroup(key, null);
  };

  const onDropToGroup = (groupId: string) => (e: React.DragEvent) => {
    e.preventDefault();
    const key = String(e.dataTransfer.getData("application/x-utmtool-key") || e.dataTransfer.getData("text/plain") || "").trim();
    if (!key) return;
    moveKeyToGroup(key, groupId);
  };

  const allowDrop = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const selectedField = selectedKey ? keyInputs[selectedKey] : undefined;
  const selectedStats = selectedKey ? getKeyStats(selectedKey) : null;
  const selectedGroupId = useMemo(() => {
    const group = normalizedGroups.find((g) => (g.keys || []).includes(selectedKey));
    return group ? group.id : "";
  }, [normalizedGroups, selectedKey]);
  const placeholderByKey: Record<string, string> = {
    base_url: "https://example.com/landing",
    utm_source: "google",
    utm_medium: "cpc",
    utm_campaign: "spring_sale",
    utm_content: "creative_a",
    utm_term: "keyword_here",
  };

  const addCustomField = () => {
    if (!onAddCustomParamKey) return;
    const err = onAddCustomParamKey(newCustomKey);
    if (err) {
      setNewCustomKeyError(err);
      return;
    }
    const nextKey = String(newCustomKey || "").trim();
    setNewCustomKey("");
    setNewCustomKeyError(null);
    if (nextKey) setSelectedKey(nextKey);
  };

  return (
    <div className="utmtool-rel">
      <div className="utmtool-rel-layout">
        <div className="utmtool-rel-board">
          <p className="utmtool-help">
            Drag fields into a zip group to lock them together by row. Fields left independent combine via Full Cartesian.
          </p>

          <div className="utmtool-rel-block">
            <h4>Independent fields</h4>
            <div
              className="utmtool-rel-dropzone"
              onDragOver={allowDrop}
              onDrop={onDropToIndependent}
            >
              {independentKeys.length ? (
                <div className="utmtool-pill-row">
                  {independentKeys.map((key) => {
                    const stats = getKeyStats(key);
                    return (
                      <button
                        key={key}
                        type="button"
                        className={`utmtool-pill${stats.missing ? " is-missing" : ""}${selectedKey === key ? " is-selected" : ""}`}
                        draggable
                        onClick={() => setSelectedKey(key)}
                        onDragStart={onDragStartKey(key)}
                        title={stats.missing ? "Missing required values" : "Click to edit"}
                      >
                        <span className="utmtool-pill-text">{getKeyLabel(key)}</span>
                        <span className="utmtool-pill-badge">{stats.length}</span>
                      </button>
                    );
                  })}
                </div>
              ) : (
                <p className="utmtool-help">All fields are currently assigned to zip groups.</p>
              )}
            </div>
          </div>

          <div className="utmtool-rel-block">
            <div className="utmtool-rel-groups-head">
              <h4>Zip groups (row-aligned)</h4>
              <button type="button" className="btn-secondary" onClick={addGroup}>
                Add zip group
              </button>
            </div>

            {normalizedGroups.length ? (
              <div className="utmtool-rel-groups">
                {normalizedGroups.map((group) => {
                  const groupRowCount = (group.keys || []).reduce((acc, key) => {
                    const len = getKeyStats(key).length;
                    return Math.max(acc, len > 1 ? len : 1);
                  }, 1);
                  const hasMismatch = (group.keys || []).some((key) => {
                    const len = getKeyStats(key).length;
                    return len !== 1 && len !== groupRowCount;
                  });
                  return (
                  <div key={group.id} className={`utmtool-rel-group${hasMismatch ? " is-mismatch" : ""}`}>
                    <div className="utmtool-rel-group-head">
                      <input
                        className="utmtool-input"
                        type="text"
                        value={group.name}
                        onChange={(e) => renameGroup(group.id, e.target.value)}
                        aria-label="Group name"
                      />
                      <span
                        className={`utmtool-rel-group-meta${hasMismatch ? " is-mismatch" : ""}`}
                        title={hasMismatch ? "Some fields have mismatched list lengths for this zip group." : "Zip group row count"}
                      >
                        {groupRowCount} rows
                      </span>
                      <div className="utmtool-rel-group-actions">
                        <button
                          type="button"
                          className="btn-secondary"
                          onClick={() => moveGroup(group.id, -1)}
                          disabled={normalizedGroups[0]?.id === group.id}
                        >
                          Up
                        </button>
                        <button
                          type="button"
                          className="btn-secondary"
                          onClick={() => moveGroup(group.id, 1)}
                          disabled={normalizedGroups[normalizedGroups.length - 1]?.id === group.id}
                        >
                          Down
                        </button>
                        <button type="button" className="btn-secondary" onClick={() => removeGroup(group.id)}>
                          Remove
                        </button>
                      </div>
                    </div>

                    <div className="utmtool-rel-group-add">
                      <label className="utmtool-label" htmlFor={`utmtool-rel-add-${group.id}`}>Add field</label>
                      <select
                        id={`utmtool-rel-add-${group.id}`}
                        className="utmtool-select"
                        value=""
                        onChange={(e) => {
                          const key = String(e.target.value || "").trim();
                          if (!key) return;
                          moveKeyToGroup(key, group.id);
                        }}
                      >
                        <option value="">Choose…</option>
                        {independentKeys.map((key) => (
                          <option key={key} value={key}>
                            {getKeyLabel(key)}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div
                      className="utmtool-rel-dropzone"
                      onDragOver={allowDrop}
                      onDrop={onDropToGroup(group.id)}
                    >
                      {group.keys.length ? (
                        <div className="utmtool-pill-row">
                          {group.keys.map((key) => {
                            const stats = getKeyStats(key);
                            return (
                              <button
                                key={key}
                                type="button"
                                className={`utmtool-pill${stats.missing ? " is-missing" : ""}${selectedKey === key ? " is-selected" : ""}`}
                                draggable
                                onClick={() => setSelectedKey(key)}
                                onDragStart={onDragStartKey(key)}
                                title="Click to edit • Drag back to Independent to remove"
                              >
                                <span className="utmtool-pill-text">{getKeyLabel(key)}</span>
                                <span className="utmtool-pill-badge">{stats.length}</span>
                              </button>
                            );
                          })}
                        </div>
                      ) : (
                        <p className="utmtool-help">Drop fields here to zip them by row.</p>
                      )}
                    </div>

                    <p className="utmtool-help">
                      Tip: fields in a zip group should have matching row counts (or be single values).
                    </p>
                  </div>
                );
                })}
              </div>
            ) : (
              <p className="utmtool-help">No zip groups yet. Add one to define row-level relationships.</p>
            )}
          </div>

          {csvColumns.length === 0 ? (
            <p className="utmtool-help">CSV column mapping is available after you upload a CSV in “Data sources”.</p>
          ) : null}
        </div>

        <div className="utmtool-rel-inspector">
          <div className="utmtool-rel-inspector-head">
            <h4>Field inspector</h4>
            {selectedKey ? (
              <span className={`utmtool-pill-badge${selectedStats?.missing ? " is-missing" : ""}`}>
                {selectedStats?.length ?? 0}
              </span>
            ) : null}
          </div>

          {selectedKey ? (
            <>
              <p className="utmtool-help">
                Selected: <strong>{getKeyLabel(selectedKey)}</strong>
                {requiredKeys.has(selectedKey) ? " (required)" : ""}
              </p>

              {selectedField ? (
                <FieldEditor
                  label={selectedKey === "base_url" ? "Landing page URL(s)" : selectedKey}
                  required={requiredKeys.has(selectedKey)}
                  field={selectedField}
                  onChange={(next) => onChangeKeyInput(selectedKey, next)}
                  csvColumns={csvColumns}
                  combinationMode="groups"
                  placeholder={placeholderByKey[selectedKey] || "value"}
                  help={selectedKey === "base_url" ? "Tip: put Base URL inside a zip group to align it with campaigns row-by-row." : undefined}
                />
              ) : (
                <p className="utmtool-help">This field is not currently available.</p>
              )}

              <div className="utmtool-row">
                <label className="utmtool-label" htmlFor="utmtool-rel-placement">Relationship</label>
                <select
                  id="utmtool-rel-placement"
                  className="utmtool-select"
                  value={selectedGroupId}
                  onChange={(e) => {
                    const id = String(e.target.value || "");
                    moveKeyToGroup(selectedKey, id || null);
                  }}
                >
                  <option value="">Independent (Cartesian)</option>
                  {normalizedGroups.map((g) => (
                    <option key={g.id} value={g.id}>
                      Zip group: {g.name}
                    </option>
                  ))}
                </select>
              </div>

              {onAddCustomParamKey ? (
                <div className="utmtool-rel-add-custom">
                  <h4>Add a custom field</h4>
                  <div className="utmtool-row">
                    <input
                      className="utmtool-input"
                      type="text"
                      value={newCustomKey}
                      placeholder="audience"
                      onChange={(e) => setNewCustomKey(e.target.value)}
                    />
                    <button type="button" className="btn-secondary" onClick={addCustomField}>
                      Add
                    </button>
                  </div>
                  {newCustomKeyError ? <p className="utmtool-error">{newCustomKeyError}</p> : null}
                  <p className="utmtool-help">Custom fields appear as draggable chips once added.</p>
                </div>
              ) : null}
            </>
          ) : (
            <p className="utmtool-help">Select a field to edit its values.</p>
          )}
        </div>
      </div>
    </div>
  );
};

const App = () => {
  const [config, setConfig] = useState<AppConfigState>(defaultConfig);
  const [campaignBuilder, setCampaignBuilder] = useState<CampaignBuilderState>(defaultCampaignBuilder);

  const [presetName, setPresetName] = useState("");
  const [presets, setPresets] = useState<any[]>([]);

  const workerRef = useRef<Worker | null>(null);
  const activeRequestId = useRef<string | null>(null);

  const [status, setStatus] = useState<"idle" | "generating" | "done" | "error" | "cancelled">("idle");
  const [errors, setErrors] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [estimatedTotal, setEstimatedTotal] = useState<number>(0);
  const [generatedCount, setGeneratedCount] = useState<number>(0);
  const [paramKeys, setParamKeys] = useState<string[]>([]);
  const [rows, setRows] = useState<any[]>([]);
  const [filterQuery, setFilterQuery] = useState("");
  const [toast, setToast] = useState<string | null>(null);
  const toastTimeoutRef = useRef<number | null>(null);

  const csvMeta = useMemo(() => {
    const csvText = String(config.csvText || "");
    if (!csvText.trim()) return { columns: [] as CsvColumnOption[], rowCount: 0, hasCsv: false };
    const parsed = core.parseCsv(csvText);
    const headers = core.getCsvHeaders(parsed) || [];
    const columns = headers.map((h: string, idx: number) => {
      const name = String(h || "").trim() || "(blank)";
      return { index: idx, label: `${name} (col ${idx + 1})` };
    });
    return { columns, rowCount: Math.max(0, parsed.length - 1), hasCsv: true };
  }, [config.csvText]);

  const relationshipRequiredKeys = useMemo(() => new Set([
    "base_url",
    "utm_source",
    "utm_medium",
    "utm_campaign",
  ]), []);

  const relationshipKeys = useMemo(() => {
    const keys: string[] = [
      "base_url",
      "utm_source",
      "utm_medium",
      "utm_campaign",
      "utm_content",
      "utm_term",
    ];
    (config.customParams || []).forEach((p) => {
      const key = String(p?.key || "").trim();
      if (!key) return;
      if (!core.isValidParamKey(key)) return;
      if (!keys.includes(key)) keys.push(key);
    });
    return keys;
  }, [config.customParams]);

  const relationshipKeyInputs = useMemo(() => {
    const map: Record<string, FieldInputState | undefined> = {
      base_url: config.baseUrl,
      utm_source: config.utm.source,
      utm_medium: config.utm.medium,
      utm_campaign: config.utm.campaign,
      utm_content: config.utm.content,
      utm_term: config.utm.term,
    };
    (config.customParams || []).forEach((p) => {
      const key = String(p?.key || "").trim();
      if (!key) return;
      if (!core.isValidParamKey(key)) return;
      map[key] = p.value;
    });
    return map;
  }, [config.baseUrl, config.customParams, config.utm]);

  const showToast = useCallback((message: string) => {
    setToast(message);
    if (toastTimeoutRef.current) window.clearTimeout(toastTimeoutRef.current);
    toastTimeoutRef.current = window.setTimeout(() => setToast(null), 1800);
  }, []);

  const getGenerationRequest = useCallback(() => {
    return {
      mode: config.mode,
      overrideExisting: config.overrideExisting,
      normalization: config.normalization,
      excludeRulesText: config.excludeRulesText,
      csvText: config.csvText,
      relationshipGroups: config.relationshipGroups,
      baseUrl: config.baseUrl,
      utm: config.utm,
      customParams: config.customParams.map((p) => ({ key: p.key, value: p.value })),
    };
  }, [config]);

  const preflight = useCallback((limit: number) => {
    const req = getGenerationRequest();
    const { errors, warnings, resolved } = core.resolveAndValidateConfig(req);
    if (errors && errors.length) return { ok: false as const, errors, warnings: warnings || [], estimatedTotal: 0 };
    const estimatedTotal = core.estimateTotalRows(resolved);
    if (!Number.isFinite(estimatedTotal) || estimatedTotal < 0) {
      return { ok: false as const, errors: ["Unable to estimate row count."], warnings: warnings || [], estimatedTotal: 0 };
    }
    if (limit !== Infinity && estimatedTotal > limit) {
      // The generator will still stop at `limit`, but we warn so the user understands the truncation.
      const note = `Estimated ${estimatedTotal.toLocaleString("en-US")} rows; this run will stop at ${limit.toLocaleString("en-US")}.`;
      return { ok: true as const, req, warnings: [...(warnings || []), note], estimatedTotal };
    }
    return { ok: true as const, req, warnings: warnings || [], estimatedTotal };
  }, [getGenerationRequest]);

  const startWorker = useCallback(() => {
    if (workerRef.current) return workerRef.current;
    const worker = new Worker("/dist/utm-batch-builder.worker.js");
    workerRef.current = worker;
    worker.addEventListener("message", (event: MessageEvent) => {
      const msg = (event.data || {}) as WorkerResponse;
      if (!msg || typeof msg !== "object") return;
      if (!activeRequestId.current || msg.requestId !== activeRequestId.current) return;

      if (msg.type === "start") {
        setWarnings(msg.warnings || []);
        setEstimatedTotal(msg.estimatedTotal || 0);
        setParamKeys(msg.paramKeys || []);
        return;
      }

      if (msg.type === "chunk") {
        setGeneratedCount(msg.generatedCount || 0);
        setRows((prev) => prev.concat(msg.rows || []));
        return;
      }

      if (msg.type === "done") {
        setGeneratedCount(msg.generatedCount || 0);
        setStatus("done");
        return;
      }

      if (msg.type === "cancelled") {
        setGeneratedCount(msg.generatedCount || 0);
        setStatus("cancelled");
        return;
      }

      if (msg.type === "error") {
        setWarnings(msg.warnings || []);
        setErrors(msg.errors || ["Generation failed."]);
        setStatus("error");
      }
    });
    return worker;
  }, []);

  const cancelGeneration = useCallback(() => {
    if (!activeRequestId.current) return;
    const id = activeRequestId.current;
    const worker = startWorker();
    worker.postMessage({ type: "cancel", requestId: id });
    activeRequestId.current = null;
    setStatus("cancelled");
  }, [startWorker]);

  const runGeneration = useCallback(async (kind: "preview" | "full") => {
    const limit = kind === "preview"
      ? Math.max(1, Math.floor(config.previewLimit || 10))
      : Math.max(1, Math.floor(config.maxRows || 50000));
    const check = preflight(limit);

    setErrors([]);
    setWarnings([]);
    setRows([]);
    setGeneratedCount(0);
    setEstimatedTotal(0);
    setParamKeys([]);
    setFilterQuery("");

    if (!check.ok) {
      setErrors(check.errors || ["Invalid configuration."]);
      setWarnings(check.warnings || []);
      setStatus("error");
      return;
    }

    setWarnings(check.warnings || []);
    setEstimatedTotal(check.estimatedTotal || 0);
    setStatus("generating");

    const requestId = makeId();
    activeRequestId.current = requestId;
    const worker = startWorker();
    worker.postMessage({ type: "generate", requestId, config: check.req, limit, chunkSize: 500 });
  }, [config.maxRows, config.previewLimit, preflight, startWorker]);

  const handleCsvUpload = async (file: File | null) => {
    if (!file) return;
    const text = await file.text();
    setConfig((prev) => ({ ...prev, csvText: text }));
  };

  const addCustomParam = () => {
    setConfig((prev) => ({
      ...prev,
      customParams: prev.customParams.concat({
        id: makeId(),
        key: "",
        value: makeField({ mode: prev.mode === "templateRows" ? "single" : "single" }),
      }),
    }));
  };

  const updateCustomParam = (id: string, next: Partial<CustomParamState>) => {
    setConfig((prev) => ({
      ...prev,
      customParams: prev.customParams.map((p) => (p.id === id ? { ...p, ...next } : p)),
    }));
  };

  const removeCustomParam = (id: string) => {
    setConfig((prev) => ({ ...prev, customParams: prev.customParams.filter((p) => p.id !== id) }));
  };

  const changeCombinationMode = (next: CombinationMode) => {
    setConfig((prev) => ({ ...prev, mode: next }));
    if (next !== "groups") return;
    window.setTimeout(() => {
      const el = document.getElementById("utmtool-relationships");
      if (el && typeof el.scrollIntoView === "function") {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }, 0);
  };

  const setFieldInputForKey = useCallback((key: string, next: FieldInputState) => {
    if (key === "base_url") {
      setConfig((prev) => ({ ...prev, baseUrl: next }));
      return;
    }
    if (key === "utm_source") {
      setConfig((prev) => ({ ...prev, utm: { ...prev.utm, source: next } }));
      return;
    }
    if (key === "utm_medium") {
      setConfig((prev) => ({ ...prev, utm: { ...prev.utm, medium: next } }));
      return;
    }
    if (key === "utm_campaign") {
      setConfig((prev) => ({ ...prev, utm: { ...prev.utm, campaign: next } }));
      return;
    }
    if (key === "utm_content") {
      setConfig((prev) => ({ ...prev, utm: { ...prev.utm, content: next } }));
      return;
    }
    if (key === "utm_term") {
      setConfig((prev) => ({ ...prev, utm: { ...prev.utm, term: next } }));
      return;
    }

    setConfig((prev) => ({
      ...prev,
      customParams: (prev.customParams || []).map((p) => (
        String(p?.key || "").trim() === key ? { ...p, value: next } : p
      )),
    }));
  }, []);

  const addCustomParamKeyFromBuilder = useCallback((rawKey: string) => {
    const key = String(rawKey || "").trim();
    if (!key) return "Enter a key.";
    if (!core.isValidParamKey(key)) return "Invalid key (use letters, numbers, _ or -).";
    if (["base_url", "utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term"].includes(key)) {
      return `“${key}” is reserved. Use the UTM fields above instead.`;
    }
    const existing = (config.customParams || []).some((p) => String(p?.key || "").trim() === key);
    if (existing) return `Custom field “${key}” already exists.`;

    setConfig((prev) => ({
      ...prev,
      customParams: (prev.customParams || []).concat({
        id: makeId(),
        key,
        value: makeField({ mode: prev.mode === "templateRows" ? "single" : "list" }),
      }),
    }));
    showToast(`Added “${key}”.`);
    return null;
  }, [config.customParams, showToast]);

  const savePreset = () => {
    const name = presetName.trim();
    if (!name) {
      showToast("Enter a preset name.");
      return;
    }
    const next = (presets || []).filter((p: any) => p.name !== name).concat({
      name,
      savedAt: new Date().toISOString(),
      config: stripCsvForStorage(config),
      campaignBuilder,
    });
    setPresets(next);
    setPresetName("");
    try {
      localStorage.setItem(STORAGE_PRESETS_KEY, JSON.stringify(next));
      showToast("Preset saved.");
    } catch (_) {
      showToast("Unable to save preset (storage blocked).");
    }
  };

  const loadPreset = (name: string) => {
    const preset = (presets || []).find((p: any) => p.name === name);
    if (!preset) return;
    const loaded = preset.config || {};
    setConfig((prev) => ({
      ...defaultConfig(),
      ...loaded,
      csvText: "",
      customParams: Array.isArray(loaded.customParams) ? loaded.customParams : prev.customParams,
    }));
    if (preset.campaignBuilder) setCampaignBuilder(preset.campaignBuilder);
    showToast(`Loaded "${name}".`);
  };

  const deletePreset = (name: string) => {
    const next = (presets || []).filter((p: any) => p.name !== name);
    setPresets(next);
    try {
      localStorage.setItem(STORAGE_PRESETS_KEY, JSON.stringify(next));
      showToast("Preset deleted.");
    } catch (_) {
      showToast("Unable to update presets (storage blocked).");
    }
  };

  const generateCampaignValues = () => {
    const tokenList = extractTemplateTokens(campaignBuilder.template);
    if (!tokenList.length) {
      setCampaignBuilder((prev) => ({ ...prev, generated: [], lastError: "Template has no {tokens}." }));
      return;
    }

    const csvRows = config.csvText ? core.parseCsv(config.csvText) : null;
    const getTokenValues = (field: FieldInputState, label: string) => {
      if (field.mode === "list") return core.parseMultiline(field.list).map((v: string) => core.normalizeValue(v, config.normalization));
      if (field.mode === "csvColumn") {
        if (!csvRows) throw new Error(`${label} is mapped to CSV but no CSV is loaded.`);
        return core.getCsvColumnValues(csvRows, field.csvColumnIndex).map((v: string) => core.normalizeValue(v, config.normalization));
      }
      const single = core.normalizeValue(field.single, config.normalization);
      return single ? [single] : [""];
    };

    try {
      const tokenValues: Record<string, string[]> = {};
      tokenList.forEach((t) => {
        tokenValues[t] = getTokenValues(campaignBuilder.tokens[t] || makeField(), t);
      });

      let generated: string[] = [];

      if (campaignBuilder.mode === "zip") {
        const lengths = tokenList.map((t) => (tokenValues[t]?.length || 0)).map((len) => (len > 1 ? len : 1));
        const rowCount = Math.max(1, ...lengths);
        tokenList.forEach((t) => {
          const vals = tokenValues[t] || [];
          if (vals.length !== 1 && vals.length !== rowCount) {
            throw new Error(`Token "${t}" has ${vals.length} values but must be length 1 or ${rowCount} for Zip mode.`);
          }
        });
        for (let i = 0; i < rowCount; i++) {
          const valuesByToken: Record<string, string> = {};
          tokenList.forEach((t) => {
            const vals = tokenValues[t] || [""];
            valuesByToken[t] = vals.length === 1 ? vals[0] : (vals[i] ?? "");
          });
          generated.push(applyTemplate(campaignBuilder.template, valuesByToken));
        }
      } else {
        const fields = tokenList.map((t) => ({ token: t, values: (tokenValues[t] || []).filter((v) => v !== "") }));
        const normalized = fields.map((f) => ({ ...f, values: f.values.length ? f.values : [""] }));
        const indices = new Array(normalized.length).fill(0);
        while (true) {
          const valuesByToken: Record<string, string> = {};
          normalized.forEach((f, idx) => {
            valuesByToken[f.token] = f.values[indices[idx]];
          });
          generated.push(applyTemplate(campaignBuilder.template, valuesByToken));

          let pos = indices.length - 1;
          while (pos >= 0) {
            indices[pos] += 1;
            if (indices[pos] < normalized[pos].values.length) break;
            indices[pos] = 0;
            pos -= 1;
          }
          if (pos < 0) break;
        }
      }

      setCampaignBuilder((prev) => ({
        ...prev,
        generated,
        lastError: null,
      }));
    } catch (err: any) {
      setCampaignBuilder((prev) => ({ ...prev, generated: [], lastError: err?.message || String(err) }));
    }
  };

  const useGeneratedCampaigns = () => {
    if (!campaignBuilder.generated.length) {
      showToast("No campaigns generated yet.");
      return;
    }
    setConfig((prev) => ({
      ...prev,
      utm: {
        ...prev.utm,
        campaign: {
          ...prev.utm.campaign,
          mode: "list",
          list: campaignBuilder.generated.join("\n"),
        },
      },
    }));
    showToast("utm_campaign updated.");
  };

  // Load presets + last config on mount.
  useEffect(() => {
    const presetData = safeJsonParse(localStorage.getItem(STORAGE_PRESETS_KEY));
    if (Array.isArray(presetData)) setPresets(presetData);

    const last = safeJsonParse(localStorage.getItem(STORAGE_LAST_KEY));
    if (last && typeof last === "object") {
      if (last.config) {
        setConfig((prev) => ({ ...defaultConfig(), ...last.config, customParams: last.config.customParams || prev.customParams }));
      }
      if (last.campaignBuilder) setCampaignBuilder(last.campaignBuilder);
    }
  }, []);

  const customKeyByIdRef = useRef<Record<string, string>>({});
  useEffect(() => {
    const prev = customKeyByIdRef.current;
    const renames: Array<{ from: string; to: string }> = [];
    const next: Record<string, string> = {};

    (config.customParams || []).forEach((p) => {
      const id = String(p?.id || "");
      const to = String(p?.key || "").trim();
      const from = prev[id] || "";
      next[id] = to;
      if (from && to && from !== to) renames.push({ from, to });
    });

    customKeyByIdRef.current = next;
    if (!renames.length) return;

    setConfig((prevConfig) => {
      const updated = (prevConfig.relationshipGroups || []).map((g) => ({
        ...g,
        keys: (g.keys || []).map((k) => {
          const rename = renames.find((r) => r.from === k);
          return rename ? rename.to : k;
        }),
      }));
      return { ...prevConfig, relationshipGroups: updated };
    });
  }, [config.customParams]);

  useEffect(() => {
    setConfig((prevConfig) => {
      if (!prevConfig.relationshipGroups?.length) return prevConfig;
      const allowed = new Set(relationshipKeys);
      const used = new Set<string>();
      let changed = false;

      const nextGroups = prevConfig.relationshipGroups.map((g) => {
        const keys: string[] = [];
        (g.keys || []).forEach((raw) => {
          const key = String(raw || "").trim();
          if (!key) { changed = true; return; }
          if (!allowed.has(key)) { changed = true; return; }
          if (used.has(key)) { changed = true; return; }
          used.add(key);
          keys.push(key);
        });
        if (keys.length !== (g.keys || []).length) changed = true;
        return { ...g, keys };
      });

      if (!changed) return prevConfig;
      return { ...prevConfig, relationshipGroups: nextGroups };
    });
  }, [relationshipKeys]);

  // Persist last config (without CSV contents).
  useEffect(() => {
    const handle = window.setTimeout(() => {
      try {
        localStorage.setItem(
          STORAGE_LAST_KEY,
          JSON.stringify({
            savedAt: new Date().toISOString(),
            config: stripCsvForStorage(config),
            campaignBuilder,
          })
        );
      } catch (_) {}
    }, 450);
    return () => window.clearTimeout(handle);
  }, [config, campaignBuilder]);

  // Keep campaign token inputs in sync with template.
  useEffect(() => {
    const tokenList = extractTemplateTokens(campaignBuilder.template);
    setCampaignBuilder((prev) => {
      const nextTokens: Record<string, FieldInputState> = {};
      tokenList.forEach((t) => {
        nextTokens[t] = prev.tokens[t] || makeField({ mode: "list" });
      });
      const sameKeys = Object.keys(prev.tokens).length === Object.keys(nextTokens).length
        && Object.keys(prev.tokens).every(k => nextTokens[k]);
      if (sameKeys) return prev;
      return { ...prev, tokens: nextTokens };
    });
  }, [campaignBuilder.template]);

  const exportCsv = async () => {
    if (!rows.length) {
      showToast("No results to export.");
      return;
    }
    const csv = buildCsv(rows, paramKeys);
    downloadTextFile("utm-urls.csv", csv, "text/csv;charset=utf-8");
  };

  const copyAll = async () => {
    if (!rows.length) {
      showToast("No results to copy.");
      return;
    }
    const ok = await copyToClipboard(rows.map(r => String(r?.finalUrl || "")).join("\n"));
    showToast(ok ? "Copied all URLs." : "Copy failed.");
  };

  const copyRow = async (row: any) => {
    const ok = await copyToClipboard(String(row?.finalUrl || ""));
    showToast(ok ? "Copied." : "Copy failed.");
  };

  return (
    <div className="utmtool-app" data-testid="utmtool-app">
      {toast ? <div className="utmtool-toast" role="status">{toast}</div> : null}

      <div className="utmtool-layout">
        <AccordionSection
          title="1) Data sources"
          subtitle="Landing pages + optional CSV mapping."
          defaultOpen
        >
          <div className="utmtool-card">
            <h3>Base URLs</h3>
            <FieldEditor
              label="Landing page URL(s)"
              required
              field={config.baseUrl}
              onChange={(next) => setConfig((prev) => ({ ...prev, baseUrl: next }))}
              csvColumns={csvMeta.columns}
              combinationMode={config.mode}
              placeholder="https://example.com/landing"
              help={config.mode === "cartesian" ? "Enter 1+ URLs. If using List, add one URL per line." : undefined}
            />
          </div>

          <div className="utmtool-card">
            <h3>CSV (optional)</h3>
            <div className="utmtool-row">
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={(e) => handleCsvUpload(e.target.files?.[0] || null)}
              />
              <button
                type="button"
                className="btn-secondary"
                onClick={() => setConfig((prev) => ({ ...prev, csvText: "" }))}
                disabled={!config.csvText}
              >
                Clear CSV
              </button>
            </div>
            <p className="utmtool-help">
              {csvMeta.hasCsv
                ? `Loaded CSV with ${csvMeta.rowCount.toLocaleString("en-US")} rows and ${csvMeta.columns.length} columns. Drag a column chip onto any field to map it.`
                : "Upload a CSV to map Base URL / UTM fields / custom params from columns."}
            </p>
            {csvMeta.hasCsv ? (
              <div className="utmtool-csv-chips" aria-label="CSV columns">
                {csvMeta.columns.map((c) => (
                  <div
                    key={c.index}
                    className="utmtool-csv-chip"
                    draggable
                    onDragStart={(e) => {
                      e.dataTransfer.effectAllowed = "copy";
                      e.dataTransfer.setData("application/x-utmtool-csvcol", String(c.index));
                      e.dataTransfer.setData("text/plain", String(c.index));
                    }}
                    title="Drag onto a field to map"
                  >
                    {c.label}
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </AccordionSection>

        <AccordionSection
          title="2) UTM parameters"
          subtitle="Required UTMs first, optional UTMs tucked away."
          defaultOpen
        >
          <div className="utmtool-card">
            <h3>Required</h3>
            <FieldEditor
              label="utm_source"
              required
              field={config.utm.source}
              onChange={(next) => setConfig((prev) => ({ ...prev, utm: { ...prev.utm, source: next } }))}
              csvColumns={csvMeta.columns}
              combinationMode={config.mode}
              placeholder="google"
            />
            <FieldEditor
              label="utm_medium"
              required
              field={config.utm.medium}
              onChange={(next) => setConfig((prev) => ({ ...prev, utm: { ...prev.utm, medium: next } }))}
              csvColumns={csvMeta.columns}
              combinationMode={config.mode}
              placeholder="cpc"
            />
            <FieldEditor
              label="utm_campaign"
              required
              field={config.utm.campaign}
              onChange={(next) => setConfig((prev) => ({ ...prev, utm: { ...prev.utm, campaign: next } }))}
              csvColumns={csvMeta.columns}
              combinationMode={config.mode}
              placeholder="spring_sale"
            />

            <details className="utmtool-inline-details">
              <summary className="utmtool-inline-summary">Optional UTMs</summary>
              <div className="utmtool-inline-body">
                <FieldEditor
                  label="utm_content"
                  field={config.utm.content}
                  onChange={(next) => setConfig((prev) => ({ ...prev, utm: { ...prev.utm, content: next } }))}
                  csvColumns={csvMeta.columns}
                  combinationMode={config.mode}
                  placeholder="creative_a"
                />
                <FieldEditor
                  label="utm_term"
                  field={config.utm.term}
                  onChange={(next) => setConfig((prev) => ({ ...prev, utm: { ...prev.utm, term: next } }))}
                  csvColumns={csvMeta.columns}
                  combinationMode={config.mode}
                  placeholder="keyword_here"
                />
              </div>
            </details>
          </div>
        </AccordionSection>

        <AccordionSection
          title="3) Custom parameters"
          subtitle="Optional extra query params (audience, creative, placement, flight, etc.)."
          defaultOpen={config.customParams.length > 0}
        >
          <div className="utmtool-card">
            <div className="utmtool-card-head">
              <h3>Custom parameters</h3>
              <button type="button" className="btn-secondary" onClick={addCustomParam}>
                Add param
              </button>
            </div>
            {config.customParams.length ? (
              <div className="utmtool-custom-list">
                {config.customParams.map((p) => (
                  <div key={p.id} className="utmtool-custom-row">
                    <div className="utmtool-custom-head">
                      <label className="utmtool-label" htmlFor={`custom-key-${p.id}`}>Key</label>
                      <input
                        id={`custom-key-${p.id}`}
                        className="utmtool-input"
                        type="text"
                        value={p.key}
                        placeholder="audience"
                        onChange={(e) => updateCustomParam(p.id, { key: e.target.value })}
                      />
                      <button type="button" className="btn-secondary" onClick={() => removeCustomParam(p.id)}>
                        Remove
                      </button>
                    </div>
                    <FieldEditor
                      label="Value"
                      field={p.value}
                      onChange={(next) => updateCustomParam(p.id, { value: next })}
                      csvColumns={csvMeta.columns}
                      combinationMode={config.mode}
                      placeholder="prospecting"
                    />
                  </div>
                ))}
              </div>
            ) : (
              <p className="utmtool-help">Add any extra parameters you need (e.g., audience, creative, placement).</p>
            )}
          </div>
        </AccordionSection>

        <AccordionSection
          title="4) Naming helpers"
          subtitle="Build campaign names from tokens, then plug into utm_campaign."
          defaultOpen={false}
        >
          <div className="utmtool-card">
            <h3>Campaign Name Builder</h3>
            <div className="utmtool-campaign-row">
              <label className="utmtool-label" htmlFor="utmtool-campaign-template">Template</label>
              <input
                id="utmtool-campaign-template"
                className="utmtool-input"
                type="text"
                value={campaignBuilder.template}
                onChange={(e) => setCampaignBuilder((prev) => ({ ...prev, template: e.target.value }))}
              />
            </div>
            <div className="utmtool-campaign-row">
              <label className="utmtool-label" htmlFor="utmtool-campaign-mode">Build mode</label>
              <select
                id="utmtool-campaign-mode"
                className="utmtool-select"
                value={campaignBuilder.mode}
                onChange={(e) => setCampaignBuilder((prev) => ({ ...prev, mode: e.target.value as CampaignBuilderMode }))}
              >
                <option value="cartesian">Full Cartesian</option>
                <option value="zip">Zip by Row</option>
              </select>
            </div>

            {extractTemplateTokens(campaignBuilder.template).length ? (
              <div className="utmtool-token-list">
                {extractTemplateTokens(campaignBuilder.template).map((token) => (
                  <FieldEditor
                    key={token}
                    label={`{${token}}`}
                    field={campaignBuilder.tokens[token] || makeField({ mode: "list" })}
                    onChange={(next) => setCampaignBuilder((prev) => ({ ...prev, tokens: { ...prev.tokens, [token]: next } }))}
                    csvColumns={csvMeta.columns}
                    combinationMode="cartesian"
                    placeholder={token}
                  />
                ))}
              </div>
            ) : (
              <p className="utmtool-help">Add tokens like <code className="utmtool-code">{`{initiative}`}</code> to the template to configure inputs.</p>
            )}

            <div className="utmtool-actions-row">
              <button type="button" className="btn-primary" onClick={generateCampaignValues}>
                Generate campaigns
              </button>
              <button type="button" className="btn-secondary" onClick={useGeneratedCampaigns} disabled={!campaignBuilder.generated.length}>
                Use as utm_campaign
              </button>
            </div>

            {campaignBuilder.lastError ? <p className="utmtool-error">{campaignBuilder.lastError}</p> : null}
            {campaignBuilder.generated.length ? (
              <p className="utmtool-help">
                Generated <strong>{campaignBuilder.generated.length.toLocaleString("en-US")}</strong> campaign values (showing first 6):
                <span className="utmtool-codeblock">
                  {campaignBuilder.generated.slice(0, 6).join("\n")}
                </span>
              </p>
            ) : null}
          </div>
        </AccordionSection>

        <AccordionSection
          title="5) Rules & relationships"
          subtitle="Choose how lists combine, plus formatting and exclusions."
          defaultOpen
        >
          <div className="utmtool-card">
            <h3>Combination mode</h3>
            <ModePicker
              value={config.mode}
              onChange={changeCombinationMode}
            />

            <div className="utmtool-row utmtool-row-inline">
              <label className="utmtool-check">
                <input
                  type="checkbox"
                  checked={config.overrideExisting}
                  onChange={(e) => setConfig((prev) => ({ ...prev, overrideExisting: e.target.checked }))}
                />
                Override existing params on base URLs
              </label>
            </div>

            <details className="utmtool-inline-details">
              <summary className="utmtool-inline-summary">Formatting + exclusions</summary>
              <div className="utmtool-inline-body">
                <div className="utmtool-subsection">
                  <h4>Normalization</h4>
                  <div className="utmtool-row utmtool-row-inline">
                    <label className="utmtool-check">
                      <input
                        type="checkbox"
                        checked={config.normalization.lowercase}
                        onChange={(e) => setConfig((prev) => ({ ...prev, normalization: { ...prev.normalization, lowercase: e.target.checked } }))}
                      />
                      Lowercase
                    </label>
                    <label className="utmtool-check">
                      <input
                        type="checkbox"
                        checked={config.normalization.stripSpecial}
                        onChange={(e) => setConfig((prev) => ({ ...prev, normalization: { ...prev.normalization, stripSpecial: e.target.checked } }))}
                        disabled={config.normalization.slugify}
                      />
                      Strip special characters
                    </label>
                    <label className="utmtool-check">
                      <input
                        type="checkbox"
                        checked={config.normalization.slugify}
                        onChange={(e) => setConfig((prev) => ({ ...prev, normalization: { ...prev.normalization, slugify: e.target.checked } }))}
                      />
                      Slugify (strict)
                    </label>
                  </div>
                  <div className="utmtool-row">
                    <label className="utmtool-label" htmlFor="utmtool-spaces">Spaces</label>
                    <select
                      id="utmtool-spaces"
                      className="utmtool-select"
                      value={config.normalization.spaces}
                      onChange={(e) => setConfig((prev) => ({ ...prev, normalization: { ...prev.normalization, spaces: e.target.value as SpacesMode } }))}
                    >
                      <option value="underscore">Replace with underscores</option>
                      <option value="dash">Replace with dashes</option>
                      <option value="none">Keep spaces</option>
                    </select>
                  </div>
                </div>

                <div className="utmtool-subsection">
                  <h4>Exclude rules</h4>
                  <textarea
                    className="utmtool-textarea"
                    rows={4}
                    value={config.excludeRulesText}
                    placeholder={"utm_medium=display & utm_source=google_search\nplacement=feed & creative=video"}
                    onChange={(e) => setConfig((prev) => ({ ...prev, excludeRulesText: e.target.value }))}
                  />
                  <p className="utmtool-help">One rule per line. All pairs must match to exclude a row.</p>
                </div>
              </div>
            </details>
          </div>

          {config.mode === "groups" ? (
            <div className="utmtool-card" id="utmtool-relationships">
              <h3>Relationships</h3>
              <RelationshipBuilder
                keys={relationshipKeys}
                keyInputs={relationshipKeyInputs}
                requiredKeys={relationshipRequiredKeys}
                csvRowCount={csvMeta.rowCount}
                csvColumns={csvMeta.columns}
                groups={config.relationshipGroups}
                onChangeGroups={(next) => setConfig((prev) => ({ ...prev, relationshipGroups: next }))}
                onChangeKeyInput={setFieldInputForKey}
                onAddCustomParamKey={addCustomParamKeyFromBuilder}
              />
            </div>
          ) : null}
        </AccordionSection>

        <AccordionSection
          title="6) Generate & results"
          subtitle="Preview first, then generate + export."
          defaultOpen
        >
          <div className="utmtool-card">
            <h3>Generate</h3>
            <div className="utmtool-actions-row">
              <button type="button" className="btn-primary" onClick={() => runGeneration("preview")} disabled={status === "generating"}>
                Preview ({Math.max(1, Math.floor(config.previewLimit || 10))})
              </button>
              <button type="button" className="btn-secondary" onClick={() => runGeneration("full")} disabled={status === "generating"}>
                Generate
              </button>
              <button type="button" className="btn-secondary" onClick={cancelGeneration} disabled={status !== "generating"}>
                Cancel
              </button>
            </div>
            <div className="utmtool-row utmtool-row-inline">
              <label className="utmtool-label" htmlFor="utmtool-preview-limit">Preview rows</label>
              <input
                id="utmtool-preview-limit"
                className="utmtool-input utmtool-number"
                type="number"
                min={1}
                max={200}
                value={config.previewLimit}
                onChange={(e) => setConfig((prev) => ({ ...prev, previewLimit: Number(e.target.value) }))}
              />
              <label className="utmtool-label" htmlFor="utmtool-max-rows">Max rows</label>
              <input
                id="utmtool-max-rows"
                className="utmtool-input utmtool-number"
                type="number"
                min={1}
                value={config.maxRows}
                onChange={(e) => setConfig((prev) => ({ ...prev, maxRows: Number(e.target.value) }))}
              />
            </div>

            <div className="utmtool-stats">
              <div>
                <span className="utmtool-stat-label">Status</span>
                <span className="utmtool-stat-value">{status}</span>
              </div>
              <div>
                <span className="utmtool-stat-label">Estimated</span>
                <span className="utmtool-stat-value">{estimatedTotal ? estimatedTotal.toLocaleString("en-US") : "—"}</span>
              </div>
              <div>
                <span className="utmtool-stat-label">Generated</span>
                <span className="utmtool-stat-value">{generatedCount ? generatedCount.toLocaleString("en-US") : "—"}</span>
              </div>
            </div>

            {errors.length ? (
              <div className="utmtool-banner utmtool-banner-error" role="alert">
                <strong>Fix these issues:</strong>
                <ul>
                  {errors.map((e, idx) => <li key={idx}>{e}</li>)}
                </ul>
              </div>
            ) : null}

            {warnings.length ? (
              <div className="utmtool-banner utmtool-banner-warn" role="status">
                <strong>Notes:</strong>
                <ul>
                  {warnings.map((w, idx) => <li key={idx}>{w}</li>)}
                </ul>
              </div>
            ) : null}
          </div>

          <div className="utmtool-card">
            <div className="utmtool-card-head">
              <h3>Results</h3>
              <div className="utmtool-actions-row utmtool-actions-row-tight">
                <button type="button" className="btn-secondary" onClick={copyAll} disabled={!rows.length}>
                  Copy all
                </button>
                <button type="button" className="btn-secondary" onClick={exportCsv} disabled={!rows.length}>
                  Export CSV
                </button>
              </div>
            </div>

            <div className="utmtool-row">
              <label className="utmtool-label" htmlFor="utmtool-filter">Search</label>
              <input
                id="utmtool-filter"
                className="utmtool-input"
                type="text"
                value={filterQuery}
                placeholder="Filter by URL or value..."
                onChange={(e) => setFilterQuery(e.target.value)}
              />
            </div>

            {rows.length ? (
              <VirtualizedTable
                rows={rows}
                paramKeys={paramKeys}
                filterQuery={filterQuery}
                onCopyRow={copyRow}
              />
            ) : (
              <p className="utmtool-help">Generate to see results here.</p>
            )}
          </div>
        </AccordionSection>

        <AccordionSection
          title="Saved presets"
          subtitle="Save/load configurations in your browser."
          defaultOpen={false}
        >
          <div className="utmtool-card">
            <h3>Presets</h3>
            <div className="utmtool-row">
              <input
                className="utmtool-input"
                type="text"
                value={presetName}
                placeholder="Preset name"
                onChange={(e) => setPresetName(e.target.value)}
              />
              <button type="button" className="btn-secondary" onClick={savePreset}>
                Save
              </button>
            </div>
            {presets.length ? (
              <div className="utmtool-presets">
                {presets
                  .slice()
                  .sort((a: any, b: any) => String(a.name).localeCompare(String(b.name)))
                  .map((p: any) => (
                    <div key={p.name} className="utmtool-preset-row">
                      <button type="button" className="btn-secondary" onClick={() => loadPreset(p.name)}>
                        Load {p.name}
                      </button>
                      <button type="button" className="btn-secondary" onClick={() => deletePreset(p.name)}>
                        Delete
                      </button>
                    </div>
                  ))}
              </div>
            ) : (
              <p className="utmtool-help">Saved locally in your browser (localStorage).</p>
            )}
          </div>
        </AccordionSection>
      </div>
    </div>
  );
};

const mount = () => {
  const rootEl = document.getElementById("utm-batch-builder-root");
  if (!rootEl) return;
  createRoot(rootEl).render(<App />);
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", mount);
} else {
  mount();
}
