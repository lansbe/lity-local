import type { StepEvent } from '../types'

export interface Source {
  url: string
  title: string
  domain: string
}

const URL_RE = /https?:\/\/[^\s)]+/

/**
 * Pull the web sources (url + best-effort title) out of an agent's web_search /
 * fetch_url tool results, so the answer can show clickable source cards.
 */
export function extractSources(steps: StepEvent[] | undefined): Source[] {
  if (!steps) return []
  const seen = new Set<string>()
  const sources: Source[] = []

  for (const step of steps) {
    if (step.kind !== 'tool_result' || !step.summary) continue
    if (step.name !== 'web_search' && step.name !== 'fetch_url') continue

    let lastTitle = ''
    for (const raw of step.summary.split('\n')) {
      const line = raw.trim()
      const match = line.match(URL_RE)
      if (match) {
        const url = match[0].replace(/[.,;:!?)]+$/, '')
        if (!seen.has(url)) {
          seen.add(url)
          let domain = url
          try {
            domain = new URL(url).hostname.replace(/^www\./, '')
          } catch {
            /* keep raw url */
          }
          const before = line.slice(0, match.index).replace(/^[#\d.\s)]+/, '').trim()
          sources.push({ url, title: before || lastTitle || domain, domain })
        }
      } else if (line) {
        lastTitle = line.replace(/^[#\d.\s)]+/, '').trim()
      }
    }
  }
  return sources
}
