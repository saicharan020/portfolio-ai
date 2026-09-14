import type { NavigateAction } from "@/lib/actions";
import type { ConversationTurn } from "@/lib/conversationHistory";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface ChatResponse {
  reply: string;
  action: NavigateAction | null;
}

// `history` is optional and omitted entirely when not passed (JSON.stringify
// drops undefined properties) - callers that don't pass it get exactly the
// pre-Phase-6 request body, unchanged.
export async function sendChatMessage(
  message: string,
  history?: ConversationTurn[]
): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });

  if (!res.ok) {
    throw new Error(`Backend request failed: ${res.status}`);
  }

  return res.json();
}
