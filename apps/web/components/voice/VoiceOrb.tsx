"use client";

import type { VoiceStatus } from "./useVoiceAssistant";

interface VoiceOrbProps {
  status: VoiceStatus;
  supported: boolean;
  onStart: () => void;
  onStop: () => void;
}

const STATUS_LABEL: Record<VoiceStatus, string> = {
  idle: "Tap to speak",
  listening: "Listening… tap to end conversation",
  thinking: "Thinking… tap to end conversation",
  speaking: "Speaking… tap to end conversation",
  error: "Tap to try again",
};

const STATUS_COLOR: Record<VoiceStatus, string> = {
  idle: "bg-zinc-800 dark:bg-zinc-200",
  listening: "bg-red-500 animate-pulse",
  thinking: "bg-amber-500 animate-pulse",
  speaking: "bg-emerald-500 animate-pulse",
  error: "bg-zinc-400",
};

export function VoiceOrb({ status, supported, onStart, onStop }: VoiceOrbProps) {
  // Once a conversation has started, tapping again at any phase (listening,
  // thinking, or speaking) ends it - only "idle"/"error" start a new one.
  const active = status === "listening" || status === "thinking" || status === "speaking";

  function handleClick() {
    if (!supported) return;
    if (active) {
      onStop();
    } else {
      onStart();
    }
  }

  return (
    <div className="flex flex-col items-center gap-3">
      <button
        type="button"
        onClick={handleClick}
        disabled={!supported}
        aria-pressed={active}
        aria-label={supported ? STATUS_LABEL[status] : "Voice input unavailable in this browser"}
        className={`h-24 w-24 rounded-full text-white shadow-md transition-transform disabled:cursor-not-allowed disabled:opacity-60 ${STATUS_COLOR[status]} ${
          active ? "scale-110" : "scale-100"
        }`}
      />
      <p className="text-sm text-zinc-600 dark:text-zinc-400" aria-hidden="true">
        {supported ? STATUS_LABEL[status] : "Voice input isn't supported in this browser"}
      </p>
    </div>
  );
}
