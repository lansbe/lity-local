import { diffLines } from 'diff'
import { useMemo } from 'react'

import { cx } from '../lib/cx'

type RowType = 'add' | 'del' | 'ctx'

interface Row {
  type: RowType
  text: string
}

function buildRows(before: string, after: string): Row[] {
  const rows: Row[] = []
  for (const part of diffLines(before ?? '', after ?? '')) {
    const type: RowType = part.added ? 'add' : part.removed ? 'del' : 'ctx'
    const lines = part.value.replace(/\n$/, '').split('\n')
    for (const line of lines) {
      rows.push({ type, text: line })
    }
  }
  return rows
}

// GitHub/Codex-style: tinted line backgrounds + a colored gutter sign, on the
// app surface (works in both light and dark via the diff tokens).
const ROW_CLASS: Record<RowType, string> = {
  add: 'bg-diff-add text-primary',
  del: 'bg-diff-del text-primary',
  ctx: 'text-secondary',
}
const SIGN_CLASS: Record<RowType, string> = {
  add: 'text-diff-add',
  del: 'text-diff-del',
  ctx: 'text-tertiary/60',
}
const SIGN: Record<RowType, string> = { add: '+', del: '-', ctx: ' ' }

export function Diff({ before, after }: { before: string; after: string }) {
  const rows = useMemo(() => buildRows(before, after), [before, after])
  return (
    <pre className="not-prose m-0 overflow-x-auto bg-surface py-1 font-mono text-[12px] leading-[1.6]">
      {rows.map((row, index) => (
        <div key={index} className={cx('flex', ROW_CLASS[row.type])}>
          <span className={cx('w-7 flex-none select-none pr-2 text-right', SIGN_CLASS[row.type])}>
            {SIGN[row.type]}
          </span>
          <span className="whitespace-pre-wrap break-all pr-3">{row.text || ' '}</span>
        </div>
      ))}
    </pre>
  )
}
