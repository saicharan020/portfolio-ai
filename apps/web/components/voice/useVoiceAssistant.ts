"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { executeAction } from "@/lib/actions";
import { sendChatMessage } from "@/lib/apiClient";
import {
  cancelSpeech,
  createSpeechRecognition,
  describeRecognitionError,
  isSpeechRecognitionSupported,
  isSpeechSynthesisSupported,
  speak,
} from "@/lib/speech";

export type VoiceStatus = "idle" | "listening" | "thinking" | "speaking" | "error";

export interface VoiceMessage {
  role: "user" | "assistant";
  text: string;
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

  const supported = useSyncExternalStore(
    subscribeToNothing,
    getSupportedSnapshot,
    getServerSupportedSnapshot
  );

  const handleFinalTranscript = useCallback(async (rawText: string) => {
    const text = rawText.trim();
    setInterimTranscript("");

    if (!text) {
      setStatus("idle");
      return;
    }

    setMessages((prev) => [...prev, { role: "user", text }]);
    setStatus("thinking");
    setErrorMessage(null);

    try {
      const res = await sendChatMessage(text);
      setMessages((prev) => [...prev, { role: "assistant", text: res.reply }]);
      executeAction(res.action);
      setStatus("speaking");
      speak(
        res.reply,
        () => setStatus("idle"),
        () => {
          setErrorMessage(
            "Couldn't play the spoken reply, but the answer is shown in the transcript above."
          );
          setStatus("idle");
        }
      );
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Something went wrong.");
      setStatus("idle");
    }
  }, []);

  const start = useCallback(() => {
    if (!supported) {
      setErrorMessage(
        "Voice input isn't supported in this browser. Try Chrome or Edge, or use the text box below."
      );
      setStatus("error");
      return;
    }

    if (status === "listening") return;

    cancelSpeech();
    setErrorMessage(null);
    setInterimTranscript("");

    const recognition = createSpeechRecognition();
    if (!recognition) {
      setErrorMessage("Could not start voice input.");
      setStatus("error");
      return;
    }

    recognition.onstart = () => setStatus("listening");

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const transcript = result[0]?.transcript ?? "";
        if (result.isFinal) {
          recognition.stop();
          handleFinalTranscript(transcript);
          return;
        }
        interim += transcript;
      }
      setInterimTranscript(interim);
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      setErrorMessage(describeRecognitionError(event.error));
      setStatus("idle");
      setInterimTranscript("");
    };

    recognition.onend = () => {
      setStatus((current) => (current === "listening" ? "idle" : current));
    };

    recognitionRef.current = recognition;
    recognition.start();
  }, [supported, status, handleFinalTranscript]);

  const stop = useCallback(() => {
    recognitionRef.current?.abort();
    setInterimTranscript("");
    setStatus("idle");
  }, []);

  useEffect(() => {
    return () => {
      recognitionRef.current?.abort();
      cancelSpeech();
    };
  }, []);

  return { status, supported, interimTranscript, messages, errorMessage, start, stop };
}
