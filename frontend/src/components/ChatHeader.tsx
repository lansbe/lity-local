import { IconButton, Menu, MenuItem, MenuSeparator } from '../ui'
import type { CharacterProfile } from '../types'
import {
  ActivityIcon,
  BoxIcon,
  BrainIcon,
  CodeIcon,
  FolderIcon,
  ImageIcon,
  MoonIcon,
  MoreIcon,
  PuzzleIcon,
  SidebarIcon,
  SparklesIcon,
  SunIcon,
} from './Icons'

interface ChatHeaderProps {
  title: string
  workdir: string
  theme: 'light' | 'dark'
  workspaceOpen: boolean
  pendingChanges: number
  activeCharacter: CharacterProfile | null
  onChooseWorkdir: () => void
  onToggleTheme: () => void
  onToggleWorkspace: () => void
  onToggleSidebar: () => void
  onOpenMemory: () => void
  onOpenModels: () => void
  onOpenSkills: () => void
  onOpenCharacters: () => void
  onOpenInstructions: () => void
  onOpenUsage: () => void
}

export function ChatHeader({
  title,
  workdir,
  theme,
  workspaceOpen,
  pendingChanges,
  activeCharacter,
  onChooseWorkdir,
  onToggleTheme,
  onToggleWorkspace,
  onToggleSidebar,
  onOpenMemory,
  onOpenModels,
  onOpenSkills,
  onOpenCharacters,
  onOpenInstructions,
  onOpenUsage,
}: ChatHeaderProps) {
  const workdirName = workdir ? workdir.split(/[\\/]/).filter(Boolean).pop() : ''
  const characterImage =
    activeCharacter?.thumbnail || activeCharacter?.emotions.neutral?.image || ''

  return (
    <header className="flex items-center gap-2 border-b border-hairline bg-surface px-2.5 py-2.5">
      <IconButton label="Barre latérale" onClick={onToggleSidebar}>
        <SidebarIcon className="h-4 w-4" />
      </IconButton>

      <div className="min-w-0 flex-1">
        <h1 className="truncate text-callout font-semibold text-primary" title={title}>
          {title}
        </h1>
        {(workdirName || activeCharacter) && (
          <div className="mt-0.5 flex min-w-0 items-center gap-2">
            {workdirName && (
              <button
                type="button"
                onClick={onChooseWorkdir}
                title={workdir}
                className="-ml-0.5 flex min-w-0 max-w-full items-center gap-1 rounded px-0.5 text-caption text-tertiary transition-colors hover:text-secondary"
              >
                <FolderIcon className="h-3 w-3 flex-none" />
                <span className="truncate">{workdirName}</span>
              </button>
            )}
            {activeCharacter && (
              <button
                type="button"
                onClick={onOpenCharacters}
                title={activeCharacter.name}
                className="flex min-w-0 items-center gap-1 rounded px-0.5 text-caption text-accent transition-colors hover:text-accent-hover"
              >
                <span className="h-3.5 w-3.5 flex-none overflow-hidden rounded bg-accent/12">
                  {characterImage ? (
                    <img src={characterImage} alt="" className="h-full w-full object-cover" />
                  ) : (
                    <SparklesIcon className="h-3.5 w-3.5" />
                  )}
                </span>
                <span className="truncate">{activeCharacter.name}</span>
              </button>
            )}
          </div>
        )}
      </div>

      <IconButton label="Atelier de code" active={workspaceOpen} onClick={onToggleWorkspace} className="relative">
        <CodeIcon className="h-4 w-4" />
        {pendingChanges > 0 && (
          <span className="absolute -right-1 -top-1 flex h-4 min-w-[16px] items-center justify-center rounded-full bg-accent px-1 text-caption-2 font-semibold text-accent-contrast">
            {pendingChanges}
          </span>
        )}
      </IconButton>

      <Menu
        align="right"
        trigger={({ open, onClick }) => (
          <IconButton label="Plus" onClick={onClick} active={open}>
            <MoreIcon className="h-4 w-4" />
          </IconButton>
        )}
      >
        <MenuItem icon={<BrainIcon className="h-4 w-4" />} onClick={onOpenMemory}>
          Mémoire
        </MenuItem>
        <MenuItem icon={<SparklesIcon className="h-4 w-4" />} onClick={onOpenInstructions}>
          Instructions de conversation
        </MenuItem>
        <MenuItem icon={<BoxIcon className="h-4 w-4" />} onClick={onOpenModels}>
          Gérer les modèles
        </MenuItem>
        <MenuItem icon={<PuzzleIcon className="h-4 w-4" />} onClick={onOpenSkills}>
          Compétences
        </MenuItem>
        <MenuItem icon={<ImageIcon className="h-4 w-4" />} onClick={onOpenCharacters}>
          Personnages
        </MenuItem>
        <MenuItem icon={<ActivityIcon className="h-4 w-4" />} onClick={onOpenUsage}>
          Usage Claude & Codex
        </MenuItem>
        <MenuSeparator />
        <MenuItem
          icon={theme === 'dark' ? <SunIcon className="h-4 w-4" /> : <MoonIcon className="h-4 w-4" />}
          onClick={onToggleTheme}
        >
          {theme === 'dark' ? 'Thème clair' : 'Thème sombre'}
        </MenuItem>
      </Menu>
    </header>
  )
}
