import { lazy, Suspense } from 'react'
import rehypeKatex from 'rehype-katex'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import { CodeBlock } from './CodeBlock'

// Mermaid is heavy — load it only when a diagram is actually rendered.
const Mermaid = lazy(() => import('./Mermaid').then((module) => ({ default: module.Mermaid })))

const components: Components = {
  // react-markdown v9 wraps block code in <pre><code>. We flatten <pre> so our
  // CodeBlock (a <div>) is not nested inside a <pre>, then decide block vs
  // inline inside the code renderer.
  pre: ({ children }) => <>{children}</>,
  code: ({ className, children, node, ...props }) => {
    const text = String(children ?? '')
    const match = /language-(\w+)/.exec(className || '')
    const isBlock = Boolean(match) || text.includes('\n')
    if (match?.[1] === 'mermaid') {
      return (
        <Suspense fallback={<div className="my-3 text-callout text-tertiary">Rendu du diagramme…</div>}>
          <Mermaid chart={text.replace(/\n$/, '')} />
        </Suspense>
      )
    }
    if (isBlock) {
      return <CodeBlock language={match?.[1]} value={text.replace(/\n$/, '')} />
    }
    return (
      <code
        className="rounded-[5px] bg-surface-2 px-1.5 py-0.5 font-mono text-[0.85em] text-primary"
        {...props}
      >
        {children}
      </code>
    )
  },
  a: ({ children, node, ...props }) => (
    <a
      className="font-medium text-accent underline decoration-accent/30 underline-offset-2 transition-colors hover:decoration-accent"
      target="_blank"
      rel="noreferrer"
      {...props}
    >
      {children}
    </a>
  ),
}

export function Markdown({ content }: { content: string }) {
  return (
    <div className="prose prose-stone max-w-none text-body-lg dark:prose-invert prose-headings:font-semibold prose-headings:tracking-[-0.011em] prose-p:leading-relaxed prose-pre:bg-transparent prose-pre:p-0 prose-li:marker:text-tertiary">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
