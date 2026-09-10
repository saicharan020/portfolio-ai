import type { VoiceMessage } from "./useVoiceAssistant";

interface CaptionBarProps {
  messages: VoiceMessage[];
  interimTranscript: string;
}

export function CaptionBar({ messages, interimTranscript }: CaptionBarProps) {
  const hasContent = messages.length > 0 || interimTranscript.length > 0;

  return (
    <div
      className="max-h-64 w-full overflow-y-auto rounded-md border border-zinc-300 bg-white p-3 text-sm dark:border-zinc-700 dark:bg-zinc-900"
      role="log"
      aria-live="polite"
      aria-label="Voice conversation transcript"
    >
      {!hasContent && (
        <p className="text-zinc-500 dark:text-zinc-400">Your conversation will appear here.</p>
      )}

      <ul className="flex flex-col gap-2">
        {messages.map((m, i) => (
          <li
            key={i}
            className={
              m.role === "user"
                ? "text-black dark:text-zinc-50"
                : "text-zinc-700 dark:text-zinc-300"
            }
          >
            <span className="font-medium">{m.role === "user" ? "You" : "Assistant"}:</span>{" "}
            {m.text}
          </li>
        ))}

        {interimTranscript && (
          <li className="italic text-zinc-500 dark:text-zinc-400">
            <span className="font-medium">You:</span> {interimTranscript}…
          </li>
        )}
      </ul>
    </div>
  );
}
