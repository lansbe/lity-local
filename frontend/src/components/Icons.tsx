interface IconProps {
  className?: string
}

function svg(path: React.ReactNode) {
  return function Icon({ className = 'h-4 w-4' }: IconProps) {
    return (
      <svg
        className={className}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {path}
      </svg>
    )
  }
}

export const PlusIcon = svg(
  <>
    <path d="M12 5v14" />
    <path d="M5 12h14" />
  </>,
)

export const SendIcon = svg(<path d="M5 12h14M13 6l6 6-6 6" />)

export const PuzzleIcon = svg(
  <path d="M14 4a2 2 0 1 1 4 0v1h2a1 1 0 0 1 1 1v3h-1a2 2 0 1 0 0 4h1v3a1 1 0 0 1-1 1h-3v-1a2 2 0 1 0-4 0v1H6a1 1 0 0 1-1-1v-3H4a2 2 0 1 1 0-4h1V6a1 1 0 0 1 1-1h3V4a2 2 0 0 1 4 0z" />,
)

export const StopIcon = svg(<rect x="6" y="6" width="12" height="12" rx="2" />)

export const TrashIcon = svg(
  <>
    <path d="M3 6h18" />
    <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    <path d="M6 6l1 14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-14" />
  </>,
)

export const CopyIcon = svg(
  <>
    <rect x="9" y="9" width="13" height="13" rx="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </>,
)

export const CheckIcon = svg(<path d="M20 6L9 17l-5-5" />)

export const SunIcon = svg(
  <>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </>,
)

export const MoonIcon = svg(<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />)

export const FolderIcon = svg(
  <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />,
)

export const PencilIcon = svg(
  <>
    <path d="M12 20h9" />
    <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" />
  </>,
)

export const SearchIcon = svg(
  <>
    <circle cx="11" cy="11" r="7" />
    <path d="M21 21l-4.3-4.3" />
  </>,
)

export const CodeIcon = svg(<path d="M16 18l6-6-6-6M8 6l-6 6 6 6" />)

export const ChevronRightIcon = svg(<path d="M9 6l6 6-6 6" />)

export const SparklesIcon = svg(
  <path d="M12 4l1.6 4.4L18 10l-4.4 1.6L12 16l-1.6-4.4L6 10l4.4-1.6z" />,
)

export const GlobeIcon = svg(
  <>
    <circle cx="12" cy="12" r="9" />
    <path d="M3 12h18" />
    <path d="M12 3a14 14 0 0 1 0 18a14 14 0 0 1 0-18z" />
  </>,
)

export const FileIcon = svg(
  <>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <path d="M14 2v6h6" />
  </>,
)

export const XIcon = svg(<path d="M18 6L6 18M6 6l12 12" />)

export const GearIcon = svg(
  <>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </>,
)

export const ZapIcon = svg(<path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" />)

export const PaperclipIcon = svg(
  <path d="M21.4 11.1l-9.2 9.2a5 5 0 0 1-7-7l9.2-9.2a3.3 3.3 0 0 1 4.7 4.7l-9.2 9.2a1.7 1.7 0 0 1-2.4-2.4l8.5-8.5" />,
)

export const MicIcon = svg(
  <>
    <rect x="9" y="2" width="6" height="12" rx="3" />
    <path d="M5 10a7 7 0 0 0 14 0M12 19v3" />
  </>,
)

export const VolumeIcon = svg(
  <>
    <path d="M11 5L6 9H2v6h4l5 4z" />
    <path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a9 9 0 0 1 0 14" />
  </>,
)

export const ImageIcon = svg(
  <>
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <circle cx="9" cy="9" r="2" />
    <path d="M21 15l-5-5L5 21" />
  </>,
)

export const VideoIcon = svg(
  <>
    <rect x="2" y="6" width="13" height="12" rx="2" />
    <path d="M15 10l6-3v10l-6-3z" />
  </>,
)

export const BrainIcon = svg(
  <path d="M9 3a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 1 5 3 3 0 0 0 5 1 3 3 0 0 0 5-1 3 3 0 0 0 1-5 3 3 0 0 0-2-5 3 3 0 0 0-3-3 3 3 0 0 0-2 1 3 3 0 0 0-2-1zM12 4v16" />,
)

export const ActivityIcon = svg(<path d="M3 12h4l3 8 4-16 3 8h4" />)

export const BoxIcon = svg(
  <>
    <path d="M21 16V8a2 2 0 0 0-1-1.7l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.7l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
    <path d="M3.3 7L12 12l8.7-5M12 22V12" />
  </>,
)

export const PinIcon = svg(<path d="M9 4h6l-1 7 4 3v2H6v-2l4-3-1-7zM12 16v6" />)

export const DownloadIcon = svg(
  <>
    <path d="M12 3v12" />
    <path d="M7 10l5 5 5-5" />
    <path d="M5 21h14" />
  </>,
)

export const TerminalIcon = svg(
  <>
    <path d="M4 17l6-6-6-6" />
    <path d="M12 19h8" />
  </>,
)

export const RefreshIcon = svg(
  <>
    <path d="M21 2v6h-6" />
    <path d="M3 12a9 9 0 0 1 15-6.7L21 8" />
    <path d="M3 22v-6h6" />
    <path d="M21 12a9 9 0 0 1-15 6.7L3 16" />
  </>,
)

export const ChevronDownIcon = svg(<path d="M6 9l6 6 6-6" />)

export const MoreIcon = svg(
  <>
    <circle cx="5" cy="12" r="1" />
    <circle cx="12" cy="12" r="1" />
    <circle cx="19" cy="12" r="1" />
  </>,
)

export const SidebarIcon = svg(
  <>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M9 4v16" />
  </>,
)

export const SlidersIcon = svg(
  <>
    <path d="M4 6h10M18 6h2" />
    <path d="M4 12h2M10 12h10" />
    <path d="M4 18h12M20 18h0" />
    <circle cx="16" cy="6" r="2" />
    <circle cx="8" cy="12" r="2" />
    <circle cx="18" cy="18" r="2" />
  </>,
)

export const StarIcon = svg(
  <path d="M12 3l2.6 5.3 5.9.9-4.3 4.1 1 5.8-5.2-2.7-5.2 2.7 1-5.8L3.5 9.2l5.9-.9z" />,
)

export const CpuIcon = svg(
  <>
    <rect x="6" y="6" width="12" height="12" rx="2" />
    <path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" />
  </>,
)

export const ArrowUpIcon = svg(<path d="M12 19V5M6 11l6-6 6 6" />)

export const WrenchIcon = svg(
  <path d="M14.7 6.3a4 4 0 0 1-5.2 5.2L4 17v3h3l5.5-5.5a4 4 0 0 1 5.2-5.2l-2.7 2.7-1.8-.3-.3-1.8z" />,
)

export const SquarePenIcon = svg(
  <>
    <path d="M11 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5" />
    <path d="M18.5 2.5a2.1 2.1 0 0 1 3 3L12 15l-4 1 1-4z" />
  </>,
)

export const HardDriveIcon = svg(
  <>
    <path d="M22 12H2M5.5 6h13l3 6v6a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1v-6z" />
    <path d="M6 16h.01M10 16h.01" />
  </>,
)
