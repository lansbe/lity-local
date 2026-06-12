import { useState } from 'react'

import type { StepEvent } from '../types'
import { CheckIcon, ChevronRightIcon, FileIcon, SearchIcon, TerminalIcon, XIcon } from './Icons'

const TOOL_LABEL: Record<string, string> = {
  list_files: 'Liste les fichiers',
  read_file: 'Lit',
  search: 'Recherche',
  run_command: 'Exécute',
  write_file: 'Écrit',
  edit_file: 'Modifie',
}

function toolIcon(name: string) {
  if (name === 'run_command') return <TerminalIcon className="h-3.5 w-3.5" />
  if (name === 'search') return <SearchIcon className="h-3.5 w-3.5" />
  return <FileIcon className="h-3.5 w-3.5" />
}

function describe(step: StepEvent): string {
  const name = step.name ?? ''
  const label = TOOL_LABEL[name] ?? name
  const arg =
    (step.args?.path as string) || (step.args?.query as string) || (step.args?.command as string)
  return arg ? `${label} ${arg}` : label
}

/** Codex-style activity log of the agent's tool steps for one assistant turn. */
export function StepTimeline({ steps }: { steps: StepEvent[] }) {
  const [open, setOpen] = useState(true)
  const calls = steps.filter((step) => step.kind === 'tool_call')

  return (
    <div className="mb-2.5 overflow-hidden rounded-lg border border-hairline bg-surface-2/60">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-1.5 px-3 py-2 text-left text-footnote font-medium text-secondary transition-colors hover:text-primary"
      >
        <ChevronRightIcon
          className={`h-3 w-3 flex-none text-tertiary transition-transform ${open ? 'rotate-90' : ''}`}
        />
        <span>
          {calls.length} étape{calls.length > 1 ? 's' : ''} d'agent
        </span>
      </button>
      {open && (
        <ol className="space-y-1.5 border-t border-hairline px-3 py-2">
          {steps.map((step, index) => {
            if (step.kind === 'receipts') {
              // Anti-hallucination: warn only when the answer is NOT backed by a
              // successful tool (every call failed). Grounded turns stay quiet.
              if (step.grounded !== false) return null
              return (
                <li key={index} className="flex items-start gap-2 text-caption text-warn">
                  <XIcon className="mt-0.5 h-3 w-3 flex-none" />
                  <span>Réponse non confirmée par un outil (tous les appels ont échoué).</span>
                </li>
              )
            }
            if (step.kind === 'tool_call') {
              return (
                <li key={index} className="flex items-center gap-2 text-footnote text-secondary">
                  <span className="text-tertiary">{toolIcon(step.name ?? '')}</span>
                  <span className="font-mono">{describe(step)}</span>
                </li>
              )
            }
            return (
              <li
                key={index}
                className="flex items-start gap-2 pl-6 text-caption text-tertiary"
                title={step.summary}
              >
                {step.ok ? (
                  <CheckIcon className="mt-0.5 h-3 w-3 flex-none text-success" />
                ) : (
                  <XIcon className="mt-0.5 h-3 w-3 flex-none text-danger" />
                )}
                <span className="truncate">
                  {(step.summary ?? '').split('\n')[0] || (step.ok ? 'OK' : 'Échec')}
                </span>
              </li>
            )
          })}
        </ol>
      )}
    </div>
  )
}
