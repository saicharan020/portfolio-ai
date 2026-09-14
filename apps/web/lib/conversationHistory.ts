// Phase 6: conversational context. Kept as small, pure, dependency-free
// helpers so they can be unit-tested directly (no need to render the voice
// hook or mock the Speech APIs) and shared between the voice assistant and
// the API client without either importing the other.

export interface ConversationTurn {
  role: "user" | "assistant";
  text: string;
}

// "the last 6-10 messages" per the Phase 6 spec - not persisted anywhere,
// just how much running context is threaded into each /api/chat request.
export const MAX_HISTORY_TURNS = 8;

// Returns the most recent `max` turns, oldest-first (unchanged order) -
// this is what the backend expects (see apps/api/main.py's
// _sanitize_history, which applies the same bound server-side as a
// defensive floor regardless of what the client sends).
export function boundHistory(
  turns: ConversationTurn[],
  max: number = MAX_HISTORY_TURNS
): ConversationTurn[] {
  if (max <= 0) return [];
  return turns.slice(-max);
}
