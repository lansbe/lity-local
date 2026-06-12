import { useState } from 'react'

import { BrainIcon, ChevronRightIcon } from './Icons'

/**
 * Collapsible "Raisonnement" block for reasoning models (DeepSeek-R1 & co.).
 * Collapsed by default; the header shows an in-progress hint while streaming.
 */
export function ReasoningBlock({ text, open }: { text: string; open: boolean }) {
  const [show, setShow] = useState(false)

  return (
    <div className="mb-2.5 overflow-hidden rounded-lg border border-hairline bg-surface-2/60">
      <button
        type="button"
        onClick={() => setShow((value) => !value)}
        className="flex w-full items-center gap-1.5 px-3 py-2 text-footnote font-medium text-secondary transition-colors hover:text-primary"
      >
        <BrainIcon className="h-3.5 w-3.5 flex-none text-tertiary" />
        <span>{open ? 'Raisonnement · en cours…' : 'Raisonnement'}</span>
        <ChevronRightIcon
          className={`h-3 w-3 flex-none text-tertiary transition-transform ${show ? 'rotate-90' : ''}`}
        />
      </button>
      {show && (
        <div className="whitespace-pre-wrap border-t border-hairline px-3 py-2.5 font-mono text-caption leading-relaxed text-secondary">
          {text || '…'}
        </div>
      )}
    </div>
  )
}
