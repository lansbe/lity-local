import mermaid from 'mermaid'
import { useEffect, useState } from 'react'

mermaid.initialize({ startOnLoad: false, securityLevel: 'strict' })

let counter = 0

export function Mermaid({ chart }: { chart: string }) {
  const [svg, setSvg] = useState('')
  const [error, setError] = useState(false)

  useEffect(() => {
    let cancelled = false
    counter += 1
    mermaid
      .render(`mermaid-${counter}`, chart)
      .then((result) => {
        if (!cancelled) setSvg(result.svg)
      })
      .catch(() => {
        if (!cancelled) setError(true)
      })
    return () => {
      cancelled = true
    }
  }, [chart])

  if (error) {
    return (
      <pre className="not-prose my-3 overflow-x-auto rounded-xl border border-white/10 bg-[#282c34] p-3 text-footnote text-white/70">
        {chart}
      </pre>
    )
  }

  return (
    <div
      className="not-prose my-3 flex justify-center overflow-x-auto rounded-xl border border-hairline bg-surface p-3"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}
