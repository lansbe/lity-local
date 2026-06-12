import hljs from 'highlight.js/lib/core'
import bash from 'highlight.js/lib/languages/bash'
import c from 'highlight.js/lib/languages/c'
import cpp from 'highlight.js/lib/languages/cpp'
import css from 'highlight.js/lib/languages/css'
import diff from 'highlight.js/lib/languages/diff'
import go from 'highlight.js/lib/languages/go'
import java from 'highlight.js/lib/languages/java'
import javascript from 'highlight.js/lib/languages/javascript'
import json from 'highlight.js/lib/languages/json'
import markdown from 'highlight.js/lib/languages/markdown'
import python from 'highlight.js/lib/languages/python'
import rust from 'highlight.js/lib/languages/rust'
import shell from 'highlight.js/lib/languages/shell'
import sql from 'highlight.js/lib/languages/sql'
import typescript from 'highlight.js/lib/languages/typescript'
import xml from 'highlight.js/lib/languages/xml'
import yaml from 'highlight.js/lib/languages/yaml'
import { useMemo, useState } from 'react'

import { CheckIcon, CopyIcon } from './Icons'

// Register only common languages on highlight.js CORE instead of importing all
// ~190 — keeps the initial bundle small. Each module also registers its own
// aliases (js/ts/py, sh via bash, html/svg via xml…); unknown languages fall
// back to auto-detect, then to escaped plain text.
hljs.registerLanguage('bash', bash)
hljs.registerLanguage('c', c)
hljs.registerLanguage('cpp', cpp)
hljs.registerLanguage('css', css)
hljs.registerLanguage('diff', diff)
hljs.registerLanguage('go', go)
hljs.registerLanguage('java', java)
hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('json', json)
hljs.registerLanguage('markdown', markdown)
hljs.registerLanguage('python', python)
hljs.registerLanguage('rust', rust)
hljs.registerLanguage('shell', shell)
hljs.registerLanguage('sql', sql)
hljs.registerLanguage('typescript', typescript)
hljs.registerLanguage('xml', xml)
hljs.registerLanguage('yaml', yaml)

interface CodeBlockProps {
  value: string
  language?: string
}

function highlight(value: string, language?: string): string {
  try {
    if (language && hljs.getLanguage(language)) {
      return hljs.highlight(value, { language }).value
    }
    return hljs.highlightAuto(value).value
  } catch {
    return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  }
}

const PREVIEWABLE = new Set(['html', 'svg', 'xml'])

export function CodeBlock({ value, language }: CodeBlockProps) {
  const [copied, setCopied] = useState(false)
  const [preview, setPreview] = useState(false)
  const html = useMemo(() => highlight(value, language), [value, language])
  const lang = (language || '').toLowerCase()
  const canPreview = PREVIEWABLE.has(lang)
  const srcDoc =
    lang === 'svg' || lang === 'xml'
      ? `<!doctype html><body style="margin:0;display:flex;align-items:center;justify-content:center">${value}</body>`
      : value

  async function copy() {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard may be unavailable; ignore */
    }
  }

  return (
    <div className="not-prose my-3 overflow-hidden rounded-xl border border-black/30 bg-[#282c34] text-white shadow-sm dark:border-white/10">
      <div className="flex items-center justify-between border-b border-white/[0.07] px-3 py-1.5 text-caption text-white/45">
        <span className="font-mono lowercase tracking-wide">{language || 'texte'}</span>
        <div className="flex items-center gap-1">
          {canPreview && (
            <button
              type="button"
              onClick={() => setPreview((current) => !current)}
              className={`rounded px-1.5 py-1 font-medium transition-colors hover:bg-white/10 hover:text-white/90 ${preview ? 'text-accent' : ''}`}
            >
              {preview ? 'Code' : 'Aperçu'}
            </button>
          )}
          <button
            type="button"
            onClick={copy}
            className="flex items-center gap-1.5 rounded px-1.5 py-1 font-medium transition-colors hover:bg-white/10 hover:text-white/90"
          >
            {copied ? <CheckIcon className="h-3.5 w-3.5" /> : <CopyIcon className="h-3.5 w-3.5" />}
            {copied ? 'Copié' : 'Copier'}
          </button>
        </div>
      </div>
      {preview && canPreview ? (
        <iframe
          title="Aperçu"
          sandbox="allow-scripts"
          srcDoc={srcDoc}
          className="h-80 w-full border-0 bg-white"
        />
      ) : (
        <pre className="overflow-x-auto px-4 py-3 text-[13px] leading-relaxed">
          <code className="hljs bg-transparent p-0" dangerouslySetInnerHTML={{ __html: html }} />
        </pre>
      )}
    </div>
  )
}
