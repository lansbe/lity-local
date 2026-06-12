import { Button } from '../ui'
import { CheckIcon, TerminalIcon, XIcon } from './Icons'

export interface PendingApproval {
  id: number
  command: string
}

interface CommandApprovalBannerProps {
  pending: PendingApproval | null
  onDecision: (allow: boolean) => void
}

export function CommandApprovalBanner({ pending, onDecision }: CommandApprovalBannerProps) {
  if (!pending) return null

  return (
    <div className="px-4">
      <div className="mx-auto mb-2 max-w-[45rem] rounded-xl border border-warn/30 bg-warn/[0.07] p-3">
        <p className="flex items-center gap-2 text-callout font-medium text-primary">
          <TerminalIcon className="h-4 w-4 text-warn" />
          L'agent veut exécuter une commande
        </p>
        <pre className="my-2.5 overflow-x-auto rounded-lg bg-[#282c34] px-3 py-2 font-mono text-footnote text-white">
          {pending.command}
        </pre>
        <div className="flex justify-end gap-2">
          <Button size="sm" variant="ghost" icon={<XIcon className="h-3.5 w-3.5" />} onClick={() => onDecision(false)}>
            Refuser
          </Button>
          <Button size="sm" variant="primary" icon={<CheckIcon className="h-3.5 w-3.5" />} onClick={() => onDecision(true)}>
            Approuver
          </Button>
        </div>
      </div>
    </div>
  )
}
