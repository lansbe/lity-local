import { bridge } from '../bridge'
import type { Source } from '../lib/sources'
import { GlobeIcon } from './Icons'

/** Clickable source cards for web-search answers (open in the system browser). */
export function SourceCards({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null
  return (
    <div className="mt-3.5">
      <div className="mb-1.5 text-caption font-medium uppercase tracking-wide text-tertiary">Sources</div>
      <div className="flex flex-wrap gap-2">
        {sources.map((source) => (
          <button
            key={source.url}
            type="button"
            onClick={() => bridge.openExternal(source.url)}
            title={source.url}
            className="flex max-w-[16rem] items-center gap-2 rounded-lg border border-hairline bg-surface px-2.5 py-1.5 text-left text-footnote shadow-xs transition-colors hover:border-accent/40 hover:bg-accent/5"
          >
            <GlobeIcon className="h-3.5 w-3.5 flex-none text-tertiary" />
            <span className="min-w-0">
              <span className="block truncate font-medium text-primary">{source.title}</span>
              <span className="block truncate text-tertiary">{source.domain}</span>
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
