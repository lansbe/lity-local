import { useEffect, useRef } from 'react'

import type { ChatMessage } from '../types'
import { Message } from './Message'

interface MessageListProps {
  messages: ChatMessage[]
  busy: boolean
  characterAvatar?: string
  onRegenerate: () => void
  onEdit: (text: string) => void
}

export function MessageList({
  messages,
  busy,
  characterAvatar = '',
  onRegenerate,
  onEdit,
}: MessageListProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const stickToBottom = useRef(true)

  function handleScroll() {
    const element = containerRef.current
    if (!element) return
    const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight
    stickToBottom.current = distanceFromBottom < 80
  }

  useEffect(() => {
    if (stickToBottom.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'auto' })
    }
  }, [messages])

  const lastIndex = messages.length - 1
  const lastUserIndex = messages.map((message) => message.role).lastIndexOf('user')

  return (
    <div ref={containerRef} onScroll={handleScroll} className="flex-1 overflow-y-auto">
      <div className="mx-auto flex max-w-[45rem] flex-col gap-7 px-4 py-8">
        {messages.map((message, index) => (
          <Message
            key={index}
            message={message}
            characterAvatar={characterAvatar}
            onRegenerate={
              message.role === 'assistant' && index === lastIndex && !busy && !message.pending
                ? onRegenerate
                : undefined
            }
            onEdit={message.role === 'user' && index === lastUserIndex && !busy ? onEdit : undefined}
          />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
