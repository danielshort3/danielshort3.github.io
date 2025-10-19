#!/usr/bin/env node
/**
 * Lightweight JS minifier inspired by Douglas Crockford's JSMin (public domain).
 * Removes comments and collapses whitespace while preserving strings & regex.
 */
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const srcDir = path.join(root, 'js');
const outDir = path.join(root, 'dist', 'js');
fs.rmSync(outDir, { recursive: true, force: true });

function readJsFiles(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  return entries.flatMap(entry => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      return readJsFiles(full);
    }
    if (entry.isFile() && entry.name.endsWith('.js')) {
      return [full];
    }
    return [];
  });
}

function jsmin(code) {
  const EOF = '';
  let index = 0;
  let lookahead = EOF;
  let theA = '\n';
  let theB = EOF;
  let output = '';

  const isAlphanum = (c) => {
    if (!c) return false;
    if (/^[a-z0-9_$]$/i.test(c)) return true;
    const codePoint = c.charCodeAt(0);
    return codePoint > 126;
  };

  function get() {
    let c;
    if (lookahead !== EOF) {
      c = lookahead;
      lookahead = EOF;
    } else {
      if (index >= code.length) return EOF;
      c = code.charAt(index);
      index += 1;
    }
    if (c === '\r') return '\n';
    if (c === '\u0000') return '\n';
    return c;
  }

  function peek() {
    lookahead = get();
    return lookahead;
  }

  function next() {
    let c = get();
    if (c === '/') {
      const p = peek();
      if (p === '/') {
        while (true) {
          c = get();
          if (c === '\n' || c === EOF) return c;
        }
      }
      if (p === '*') {
        get();
        while (true) {
          const ch = get();
          if (ch === EOF) throw new Error('Unterminated comment.');
          if (ch === '*') {
            if (peek() === '/') {
              get();
              return ' ';
            }
          }
        }
      }
    }
    return c;
  }

  function action(d) {
    if (d <= 1) {
      output += theA;
    }
    if (d <= 2) {
      theA = theB;
      if (theA === '\'' || theA === '"' || theA === '`') {
        const quote = theA;
        output += theA;
        while (true) {
          theA = get();
          if (theA === EOF) throw new Error('Unterminated string literal.');
          output += theA;
          if (theA === quote) break;
          if (theA === '\\') {
            theA = get();
            if (theA === EOF) throw new Error('Unterminated string escape.');
            output += theA;
          }
        }
        theA = get();
      }
    }
    if (d <= 3) {
      theB = next();
      if (
        theB === '/' &&
        (theA === '(' || theA === ',' || theA === '=' || theA === ':' || theA === '[' ||
         theA === '!' || theA === '&' || theA === '|' || theA === '?' || theA === '{' ||
         theA === '}' || theA === ';' || theA === '\n')
      ) {
        output += theA;
        output += theB;
        while (true) {
          theA = get();
          if (theA === EOF) throw new Error('Unterminated regular expression literal.');
          output += theA;
          if (theA === '/') break;
          if (theA === '\\') {
            theA = get();
            if (theA === EOF) throw new Error('Unterminated regular expression literal.');
            output += theA;
          }
        }
        theB = next();
      }
    }
  }

  action(3);
  while (theA !== EOF) {
    switch (theA) {
      case ' ':
        if (isAlphanum(theB)) {
          action(1);
        } else {
          action(2);
        }
        break;
      case '\n':
        if ('{[(+-!~'.includes(theB)) {
          action(1);
        } else if (theB === ' ') {
          action(3);
        } else if (isAlphanum(theB)) {
          action(1);
        } else {
          action(2);
        }
        break;
      default:
        if (theB === ' ') {
          if (isAlphanum(theA)) {
            action(1);
          } else {
            action(3);
          }
        } else if (theB === '\n') {
          if ('}])+-"\'`'.includes(theA) || isAlphanum(theA)) {
            action(1);
          } else {
            action(3);
          }
        } else {
          action(1);
        }
    }
  }

  return output.trim();
}

function ensureDir(pathName) {
  fs.mkdirSync(pathName, { recursive: true });
}

function minifyFile(srcPath) {
  const rel = path.relative(srcDir, srcPath);
  const dest = path.join(outDir, rel);
  ensureDir(path.dirname(dest));
  const code = fs.readFileSync(srcPath, 'utf8');
  const minified = jsmin(code);
  fs.writeFileSync(dest, minified, 'utf8');
  return { srcPath, bytes: Buffer.byteLength(minified, 'utf8'), dest };
}

ensureDir(outDir);
const files = readJsFiles(srcDir);
const results = files.map(minifyFile);
const totalBytes = results.reduce((sum, item) => sum + item.bytes, 0);
console.log(`Minified ${results.length} JS files → ${(totalBytes/1024).toFixed(1)} kB`);
