"use client";

import { useState } from "react";
import { sendChatMessage } from "@/lib/apiClient";
import { CaptionBar } from "@/components/voice/CaptionBar";
import { VoiceOrb } from "@/components/voice/VoiceOrb";
import { useVoiceAssistant } from "@/components/voice/useVoiceAssistant";

export default function Home() {
  const voice = useVoiceAssistant();

  const [message, setMessage] = useState("");
  const [reply, setReply] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSend() {
    if (!message.trim()) return;
    setLoading(true);
    setError(null);
    setReply(null);
    try {
      const res = await sendChatMessage(message);
      setReply(res.reply);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-zinc-50 px-6 py-12 font-sans dark:bg-black">
      <main className="flex w-full max-w-xl flex-col gap-4">
        <h1 className="text-2xl font-semibold text-black dark:text-zinc-50">
          Portfolio AI — Phase 2 test
        </h1>

        <section className="flex flex-col items-center gap-4 rounded-md border border-zinc-300 bg-white p-6 dark:border-zinc-700 dark:bg-zinc-900">
          <VoiceOrb
            status={voice.status}
            supported={voice.supported}
            onStart={voice.start}
            onStop={voice.stop}
          />
          <CaptionBar messages={voice.messages} interimTranscript={voice.interimTranscript} />
          {voice.errorMessage && (
            <p className="w-full rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
              {voice.errorMessage}
            </p>
          )}
        </section>

        <div className="flex items-center gap-3 text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-500">
          <div className="h-px flex-1 bg-zinc-300 dark:bg-zinc-700" />
          Or type instead
          <div className="h-px flex-1 bg-zinc-300 dark:bg-zinc-700" />
        </div>

        <textarea
          className="min-h-24 w-full rounded-md border border-zinc-300 bg-white p-3 text-black dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50"
          placeholder="Ask something..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
        />

        <button
          onClick={handleSend}
          disabled={loading || !message.trim()}
          className="w-fit rounded-full bg-foreground px-5 py-2 text-sm font-medium text-background transition-colors hover:bg-[#383838] disabled:opacity-50 dark:hover:bg-[#ccc]"
        >
          {loading ? "Sending..." : "Send"}
        </button>

        {error && (
          <p className="rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
            {error}
          </p>
        )}

        {reply && (
          <div className="rounded-md border border-zinc-300 bg-white p-3 text-sm text-black dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50">
            {reply}
          </div>
        )}
      </main>
    </div>
  );
}
