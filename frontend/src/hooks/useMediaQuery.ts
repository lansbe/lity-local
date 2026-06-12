import { useEffect, useState } from 'react'

/** Subscribe to a CSS media query. SSR-safe, updates on viewport changes. */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(query).matches,
  )

  useEffect(() => {
    const mql = window.matchMedia(query)
    const onChange = () => setMatches(mql.matches)
    onChange()
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [query])

  return matches
}

// Semantic breakpoint helpers, aligned with Tailwind's defaults.
// Mobile  < md (768)  : both side panels are overlays, single column.
// Compact < lg (1024) : sidebar persistent/rail; workspace overlays.
// Desktop ≥ lg        : true three-pane push layout.
export const useIsMobile = () => useMediaQuery('(max-width: 767px)')
export const useIsCompact = () => useMediaQuery('(max-width: 1023px)')
