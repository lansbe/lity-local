export interface SplitReasoning {
  /** Reasoning text, or null when the message has no <think> block at all. */
  reasoning: string | null
  /** The visible answer (content with <think> blocks removed). */
  answer: string
  /** True while a <think> is still open (streaming / in-progress). */
  open: boolean
}

/**
 * Separate DeepSeek-R1-style `<think>…</think>` reasoning from the answer.
 *
 * Handles multiple closed blocks and a trailing unclosed `<think>` (which
 * happens mid-stream): everything after the open tag is treated as
 * in-progress reasoning so the UI can show a collapsible block immediately.
 */
export function splitReasoning(content: string): SplitReasoning {
  const text = content ?? ''
  if (!/<think>/i.test(text)) {
    return { reasoning: null, answer: text, open: false }
  }

  const blocks: string[] = []
  let answer = text.replace(/<think>([\s\S]*?)<\/think>/gi, (_match, inner: string) => {
    blocks.push(inner.trim())
    return ''
  })

  let open = false
  const trailing = /<think>/i.exec(answer)
  if (trailing) {
    open = true
    blocks.push(answer.slice(trailing.index + trailing[0].length).trim())
    answer = answer.slice(0, trailing.index)
  }

  return { reasoning: blocks.join('\n\n').trim(), answer: answer.trim(), open }
}
