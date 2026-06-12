import { BoxIcon } from './Icons'

interface EmptyStateProps {
  onPick: (text: string) => void
  noModels?: boolean
  onManageModels?: () => void
}

const SUGGESTIONS = [
  'Explique-moi ce que fait ce projet',
  'Écris une fonction Python qui lit un fichier JSON',
  'Donne-moi des idées pour améliorer mon code',
  'Résume les points clés de la programmation asynchrone',
]

export function EmptyState({ onPick, noModels, onManageModels }: EmptyStateProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-6 py-10 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-xl border border-hairline bg-surface-2 text-accent">
        <BoxIcon className="h-7 w-7" />
      </div>
      <h1 className="mt-4 text-title-1 text-primary">Lity</h1>
      <p className="mt-2 max-w-md text-body text-secondary">
        Pose une question, demande du code, ou ouvre un dossier de travail pour analyser tes
        fichiers — le tout en local.
      </p>

      {noModels && (
        <div className="mt-6 flex max-w-sm flex-col items-center gap-3 rounded-xl border border-warn/25 bg-warn/10 px-5 py-4">
          <p className="text-callout text-warn">Aucun modèle Ollama n'est installé.</p>
          <button
            type="button"
            onClick={onManageModels}
            className="inline-flex h-9 items-center gap-2 rounded-md bg-solid px-3.5 text-callout font-medium text-solid-contrast transition-colors hover:bg-solid-hover"
          >
            <BoxIcon className="h-4 w-4" />
            Installer un modèle
          </button>
        </div>
      )}

      <div className="mt-9 grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => onPick(suggestion)}
            className="rounded-xl border border-hairline bg-surface px-4 py-3 text-left text-callout text-secondary shadow-xs transition-all duration-fast hover:-translate-y-px hover:bg-surface-2 hover:text-primary hover:shadow-sm"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  )
}
