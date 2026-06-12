// Assistant content may contain raw FILE / CREATE / SEARCH-REPLACE blocks. Those
// are surfaced as diff cards in the workspace, so we strip them from the chat
// text to avoid the giant-markdown / duplicate-garbage rendering. Tolerant of
// malformed blocks (it just removes FILE: lines and anything between the
// <<<<<<< and >>>>>>> markers).
export function stripFileBlocks(text: string): string {
  if (!text) return text
  const lines = text.split('\n')
  const out: string[] = []
  let inBlock = false
  let removedAny = false

  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed.startsWith('FILE:')) {
      removedAny = true
      continue
    }
    if (trimmed.startsWith('<<<<<<<')) {
      inBlock = true
      removedAny = true
      continue
    }
    if (trimmed.startsWith('>>>>>>>')) {
      inBlock = false
      removedAny = true
      continue
    }
    if (inBlock) continue
    out.push(line)
  }

  const result = out.join('\n').trim()
  if (removedAny && !result) {
    return '*Proposition de fichier — voir l’atelier de code.*'
  }
  return result
}
