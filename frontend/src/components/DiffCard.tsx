import type { Change } from '../types'
import { Badge, Button } from '../ui'
import { Diff } from './Diff'
import { CheckIcon, XIcon } from './Icons'

interface DiffCardProps {
  change: Change
  onApply: (change: Change) => void
  onReject: (change: Change) => void
}

export function DiffCard({ change, onApply, onReject }: DiffCardProps) {
  const isCreate = change.kind === 'create'
  const before = isCreate ? '' : change.block.search_content
  const after = isCreate ? change.block.content : change.block.replace_content
  const done = change.status !== 'pending'

  return (
    <div className="overflow-hidden rounded-xl border border-hairline shadow-xs">
      <div className="flex items-center gap-2 border-b border-hairline bg-surface-2 px-3 py-2">
        <Badge tone={isCreate ? 'success' : 'warn'}>{isCreate ? 'Création' : 'Modification'}</Badge>
        <span className="min-w-0 flex-1 truncate font-mono text-footnote text-secondary">
          {change.block.file_path}
        </span>
        {change.status === 'applied' && (
          <span className="text-footnote font-medium text-success">Appliqué</span>
        )}
        {change.status === 'rejected' && <span className="text-footnote text-tertiary">Rejeté</span>}
        {change.status === 'error' && (
          <span className="text-footnote font-medium text-danger" title={change.message}>
            Échec
          </span>
        )}
      </div>

      <Diff before={before} after={after} />

      {!done && (
        <div className="flex items-center justify-end gap-2 border-t border-hairline bg-surface px-3 py-2">
          <Button size="sm" variant="ghost" icon={<XIcon className="h-3.5 w-3.5" />} onClick={() => onReject(change)}>
            Rejeter
          </Button>
          <Button size="sm" variant="primary" icon={<CheckIcon className="h-3.5 w-3.5" />} onClick={() => onApply(change)}>
            Appliquer
          </Button>
        </div>
      )}
      {change.status === 'error' && change.message && (
        <div className="border-t border-hairline bg-danger/8 px-3 py-2 text-footnote text-danger">
          {change.message}
        </div>
      )}
    </div>
  )
}
