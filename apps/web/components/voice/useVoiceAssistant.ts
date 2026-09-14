"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { executeAction } from "@/lib/actions";
import { sendChatMessage } from "@/lib/apiClient";
import { boundHistory, type ConversationTurn } from "@/lib/conversationHistory";
import {
  cancelSpeech,
  createSpeechRecognition,
  describeRecognitionError,
  isSpeechRecognitionSupported,
  isSpeechSynthesisSupported,
  speak,
} from "@/lib/speech";

export type VoiceStatus = "idle" | "listening" | "thinking" | "speaking" | "error";

// Same shape the backend expects for conversation history (see
// apps/api/main.py's _sanitize_history) - kept as an alias, not a separate
// type, so the running transcript this hook already maintains can be sent
// straight through as history with no conversion step.
export type VoiceMessage = ConversationTurn;

// Recognition errors that mean retrying immediately won't help (permission
// denied, no mic hardware) - these end the conversation instead of looping.
const FATAL_RECOGNITION_ERRORS = new Set(["not-allowed", "service-not-allowed", "audio-capture"]);

// TEMPORARY latency-diagnosis instrumentation (no behavior change) - logs
// the four checkpoints requested for the response-latency investigation.
// Safe to delete once the source of the added latency is confirmed/fixed.
function logTiming(label: string, sinceLabel?: string, sinceMs?: number) {
  const now = performance.now();
  const delta = sinceMs !== undefined ? ` (+${(now - sinceMs).toFixed(1)}ms since ${sinceLabel})` : "";
  console.debug(`[voice-timing] ${label} @ ${now.toFixed(1)}ms${delta}`);
  return now;
}

// Browser feature support never changes during a session, so there's
// nothing to subscribe to — this just satisfies useSyncExternalStore's API.
function subscribeToNothing() {
  return () => {};
}

function getSupportedSnapshot() {
  return isSpeechRecognitionSupported() && isSpeechSynthesisSupported();
}

// The server has no window/SpeechRecognition, so it always reports
// unsupported. useSyncExternalStore renders this snapshot for both the
// server render and the client's first hydration pass, then swaps to the
// real client snapshot right after — avoiding a hydration mismatch without
// a manual setState-in-effect.
function getServerSupportedSnapshot() {
  return false;
}

export function useVoiceAssistant() {
  const [status, setStatus] = useState<VoiceStatus>("idle");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [messages, setMessages] = useState<VoiceMessage[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  // Mirrors `messages` for synchronous reads inside handleFinalTranscript
  // (a stable useCallback with `[]` deps, so it can't read fresh `messages`
  // state directly without going stale). appendMessage is the only way
  // either gets updated, so they never drift apart.
  const messagesRef = useRef<VoiceMessage[]>([]);

  const appendMessage = useCallback((message: VoiceMessage) => {
    messagesRef.current = [...messagesRef.current, message];
    setMessages(messagesRef.current);
  }, []);

  // True for the whole tap-to-tap conversation session, across every
  // listening/thinking/speaking cycle in between. Distinct from `status`,
  // which reflects only the current phase - this is what tells the
  // listening→thinking→speaking→listening loop whether to keep going.
  const conversationActiveRef = useRef(false);

  // Lets handleFinalTranscript (defined first, called from onresult inside
  // beginListening) and beginListening's own onend handler both restart
  // listening via the latest beginListening closure without the two
  // useCallbacks needing to reference each other directly, which would be a
  // circular dependency.
  const beginListeningRef = useRef<() => void>(() => {});

  // TEMPORARY: timestamp of the most recent "recognition ended with a final
  // result" event, read by handleFinalTranscript for the timing log above.
  const lastRecognitionEndRef = useRef<number>(0);

  const supported = useSyncExternalStore(
    subscribeToNothing,
    getSupportedSnapshot,
    getServerSupportedSnapshot
  );

  const handleFinalTranscript = useCallback(async (rawText: string) => {
    const t1 = lastRecognitionEndRef.current;
    const text = rawText.trim();
    setInterimTranscript("");

    if (!text) {
      if (conversationActiveRef.current) {
        beginListeningRef.current();
      } else {
        setStatus("idle");
      }
      return;
    }

    // Snapshot bounded history BEFORE appending this turn - history sent to
    // the backend is everything that came before the question being asked
    // right now, which is passed separately as `message`.
    const history = boundHistory(messagesRef.current);
    appendMessage({ role: "user", text });
    setStatus("thinking");
    setErrorMessage(null);

    try {
      const t2 = logTiming("API request starting", "recognition ended", t1);
      const res = await sendChatMessage(text, history);
      const t3 = logTiming("API response received", "request started", t2);
      // The user may have tapped stop while this request was in flight -
      // don't resurrect the conversation with a late response.
      if (!conversationActiveRef.current) return;

      appendMessage({ role: "assistant", text: res.reply });
      executeAction(res.action);
      setStatus("speaking");
      logTiming("TTS starting", "response received", t3);

      // Deliberately NOT starting SpeechRecognition here while TTS plays.
      // The Web Speech API gives no access to the raw microphone stream, no
      // audio constraints, and no amplitude/VAD data - there is no
      // reliable, dependency-free way to tell "the room's speakers just
      // played this" apart from "the user just said this". Running
      // recognition concurrently with speech synthesis previously caused
      // the assistant's own TTS to be picked up by the mic, transcribed,
      // and resubmitted as a new user question. Listening only resumes
      // once playback has actually finished (below), which is the only way
      // to guarantee TTS is never treated as user speech.
      speak(
        res.reply,
        () => {
          if (conversationActiveRef.current) {
            beginListeningRef.current();
          } else {
            setStatus("idle");
          }
        },
        () => {
          setErrorMessage(
            "Couldn't play the spoken reply, but the answer is shown in the transcript above."
          );
          if (conversationActiveRef.current) {
            beginListeningRef.current();
          } else {
            setStatus("idle");
          }
        }
      );
    } catch (err) {
      if (!conversationActiveRef.current) return;
      setErrorMessage(err instanceof Error ? err.message : "Something went wrong.");
      beginListeningRef.current();
    }
  }, [appendMessage]);

  // Starts exactly one speech-recognition "session": creates a fresh
  // SpeechRecognition instance (browsers, especially Chrome, don't reliably
  // support reusing one across restarts), wires its handlers, and starts
  // it. Called for the very first listen after a tap, and again after every
  // completed utterance/silence timeout for as long as the conversation is
  // active - never while status is "thinking" or "speaking": nothing calls
  // this until the LLM response has been handled, or until TTS playback has
  // actually finished (speak()'s onEnd/onError below), so the mic is never
  // live while the assistant's own voice might be audible.
  const beginListening = useCallback(() => {
    if (!supported) {
      setErrorMessage(
        "Voice input isn't supported in this browser. Try Chrome or Edge, or use the text box below."
      );
      setStatus("error");
      conversationActiveRef.current = false;
      return;
    }

    // Defensive: guarantee only one live recognition instance ever exists.
    if (recognitionRef.current) {
      const stale = recognitionRef.current;
      stale.onstart = null;
      stale.onresult = null;
      stale.onerror = null;
      stale.onend = null;
      stale.abort();
      recognitionRef.current = null;
    }

    cancelSpeech();
    setErrorMessage(null);
    setInterimTranscript("");

    const recognition = createSpeechRecognition();
    if (!recognition) {
      setErrorMessage("Could not start voice input.");
      setStatus("error");
      conversationActiveRef.current = false;
      return;
    }

    // Local to this one recognition session (fresh per call), so a
    // restarted session always starts with these false again.
    let finalCaptured = false;
    let fatalError = false;

    recognition.onstart = () => setStatus("listening");

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const transcript = result[0]?.transcript ?? "";
        if (result.isFinal) {
          finalCaptured = true;
          lastRecognitionEndRef.current = logTiming("speech recognition ending");
          recognition.stop();
          handleFinalTranscript(transcript);
          return;
        }
        interim += transcript;
      }
      setInterimTranscript(interim);
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      // Caused by our own abort() (explicit stop, or a fresh session
      // replacing a stale one) - not a real error, nothing to report.
      if (event.error === "aborted") return;

      if (FATAL_RECOGNITION_ERRORS.has(event.error)) {
        fatalError = true;
        conversationActiveRef.current = false;
      }

      // "no-speech" fires constantly during normal silence between
      // questions in continuous mode - surfacing it as a user-visible
      // error every time would be noisy, and silence must not end the
      // conversation, so it's silently absorbed here; onend below still
      // restarts listening for it.
      if (event.error !== "no-speech") {
        setErrorMessage(describeRecognitionError(event.error));
      }
      setInterimTranscript("");
    };

    recognition.onend = () => {
      // A newer session already replaced this one - ignore this stale event.
      if (recognitionRef.current !== recognition) return;
      recognitionRef.current = null;

      // A final transcript already handed off to handleFinalTranscript,
      // which owns whatever happens next (thinking/speaking/re-listening).
      if (finalCaptured) return;

      setInterimTranscript("");
      if (conversationActiveRef.current && !fatalError) {
        // Ended with no result (silence/no-speech timeout) - keep going.
        beginListeningRef.current();
      } else {
        setStatus((current) =>
          current === "listening" ? (fatalError ? "error" : "idle") : current
        );
      }
    };

    recognitionRef.current = recognition;
    recognition.start();
  }, [supported, handleFinalTranscript]);

  useEffect(() => {
    beginListeningRef.current = beginListening;
  }, [beginListening]);

  const start = useCallback(() => {
    if (status === "listening" || status === "thinking" || status === "speaking") return;
    conversationActiveRef.current = true;
    beginListening();
  }, [status, beginListening]);

  // The reliable way to interrupt the assistant today: tapping the orb
  // while it's speaking calls speechSynthesis.cancel() immediately (via
  // stop() below) and ends the turn/conversation. This is a deliberate,
  // deterministic signal - unlike trying to detect "the user started
  // talking" from a microphone that may also be hearing the assistant's
  // own voice through the speakers, a tap can never be a false positive.
  const stop = useCallback(() => {
    conversationActiveRef.current = false;
    if (recognitionRef.current) {
      const r = recognitionRef.current;
      r.onstart = null;
      r.onresult = null;
      r.onerror = null;
      r.onend = null;
      r.abort();
      recognitionRef.current = null;
    }
    cancelSpeech();
    setInterimTranscript("");
    setStatus("idle");
  }, []);

  useEffect(() => {
    return () => {
      conversationActiveRef.current = false;
      const r = recognitionRef.current;
      if (r) {
        r.onstart = null;
        r.onresult = null;
        r.onerror = null;
        r.onend = null;
        r.abort();
      }
      cancelSpeech();
    };
  }, []);

  return { status, supported, interimTranscript, messages, errorMessage, start, stop };
}
