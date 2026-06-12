import { useEffect, useState } from 'react'

import { bridge } from '../bridge'
import type { CliUsage, ProviderUsage } from '../types'
import { Button, Modal } from '../ui'
import { ActivityIcon, RefreshIcon } from './Icons'

const EMPTY_PROVIDER: ProviderUsage = {
  turns: 0,
  cost_usd: 0,
  input_tokens: 0,
  output_tokens: 0,
  total_tokens: 0,
  by_model: [],
}

const EMPTY: CliUsage = {
  claude: EMPTY_PROVIDER,
  codex: EMPTY_PROVIDER,
  grok: EMPTY_PROVIDER,
}

const PROVIDERS: { key: keyof CliUsage; label: string; showCost: boolean }[] = [
  { key: 'claude', label: 'Anthropic · Claude', showCost: true },
  { key: 'codex', label: 'OpenAI · Codex', showCost: false },
  { key: 'grok', label: 'xAI · Grok', showCost: false },
]

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)} M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)} k`
  return String(n)
}

function fmtCost(c: number): string {
  if (!c) return '—'
  return `$${c < 1 ? c.toFixed(4) : c.toFixed(2)}`
}

function ProviderCard({
  label,
  usage,
  showCost,
}: {
  label: string
  usage: ProviderUsage
  showCost: boolean
}) {
  return (
    <div className="rounded-xl border border-hairline bg-surface-2/50 p-3.5">
      <div className="mb-3 flex items-baseline justify-between gap-2">
        <span className="text-body font-medium text-primary">{label}</span>
        <span className="text-caption text-tertiary">
          {usage.turns} tour{usage.turns > 1 ? 's' : ''}
        </span>
      </div>

      <div className="mb-3 grid grid-cols-3 gap-2 text-center">
        <div className="rounded-lg bg-surface-2 py-2">
          <div className="text-body font-semibold text-primary">{fmtTokens(usage.input_tokens)}</div>
          <div className="text-caption text-tertiary">entrée</div>
        </div>
        <div className="rounded-lg bg-surface-2 py-2">
          <div className="text-body font-semibold text-primary">{fmtTokens(usage.output_tokens)}</div>
          <div className="text-caption text-tertiary">sortie</div>
        </div>
        <div className="rounded-lg bg-surface-2 py-2">
          <div className="text-body font-semibold text-primary">
            {showCost ? fmtCost(usage.cost_usd) : fmtTokens(usage.total_tokens)}
          </div>
          <div className="text-caption text-tertiary">{showCost ? 'coût' : 'total'}</div>
        </div>
      </div>

      {usage.by_model.length === 0 ? (
        <p className="py-1 text-center text-footnote text-tertiary">
          Aucun usage pour l'instant.
        </p>
      ) : (
        <div className="space-y-1">
          <div className="flex items-center gap-2 px-1 text-caption uppercase tracking-wide text-tertiary">
            <span className="min-w-0 flex-1">Modèle</span>
            <span className="w-16 text-right">entrée</span>
            <span className="w-16 text-right">sortie</span>
            {showCost && <span className="w-14 text-right">coût</span>}
          </div>
          {usage.by_model.map((row) => (
            <div
              key={row.model}
              className="flex items-center gap-2 rounded-lg bg-surface-2 px-2.5 py-1.5 text-footnote"
            >
              <span className="min-w-0 flex-1 truncate font-medium text-primary" title={row.model}>
                {row.model}
              </span>
              <span className="w-16 text-right tabular-nums text-secondary">
                {fmtTokens(row.input_tokens)}
              </span>
              <span className="w-16 text-right tabular-nums text-secondary">
                {fmtTokens(row.output_tokens)}
              </span>
              {showCost && (
                <span className="w-14 text-right tabular-nums text-secondary">
                  {fmtCost(row.cost_usd)}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function UsageModal({ onClose }: { onClose: () => void }) {
  const [usage, setUsage] = useState<CliUsage>(EMPTY)

  function refresh() {
    bridge
      .usage()
      .then((value) => setUsage({ ...EMPTY, ...value }))
      .catch(() => {})
  }

  useEffect(refresh, [])

  return (
    <Modal
      title="Usage Claude & Codex"
      icon={<ActivityIcon className="h-5 w-5" />}
      onClose={onClose}
      footer={
        <>
          <button
            type="button"
            onClick={refresh}
            className="mr-auto flex items-center gap-1.5 text-footnote text-tertiary transition-colors hover:text-secondary"
          >
            <RefreshIcon className="h-3.5 w-3.5" />
            Rafraîchir
          </button>
          <Button variant="primary" onClick={onClose}>
            Fermer
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        {PROVIDERS.map((provider) => (
          <ProviderCard
            key={provider.key}
            label={provider.label}
            usage={usage[provider.key] || EMPTY_PROVIDER}
            showCost={provider.showCost}
          />
        ))}
        <p className="text-caption leading-relaxed text-tertiary">
          Consommation de cette session (depuis le démarrage de l'app), par modèle. Les quotas
          d'abonnement restants (fenêtres 5 h / hebdo) ne sont pas exposés par les CLI en mode
          headless — lance <code>/usage</code> (Claude) ou <code>/status</code> (Codex) dans un
          terminal pour les voir.
        </p>
      </div>
    </Modal>
  )
}
