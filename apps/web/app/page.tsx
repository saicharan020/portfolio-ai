"use client";

import { useState } from "react";
import { sendChatMessage } from "@/lib/apiClient";

export default function Home() {
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
    <div className="flex min-h-screen flex-col items-center justify-center bg-zinc-50 px-6 font-sans dark:bg-black">
      <main className="flex w-full max-w-xl flex-col gap-4">
        <h1 className="text-2xl font-semibold text-black dark:text-zinc-50">
          Portfolio AI — Phase 1 test
        </h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Sends a message to the FastAPI backend, which forwards it to the
          local LLM and returns a reply.
        </p>

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
