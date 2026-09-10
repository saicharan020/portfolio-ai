export function isSpeechRecognitionSupported(): boolean {
  if (typeof window === "undefined") return false;
  return Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
}

export function isSpeechSynthesisSupported(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

export function createSpeechRecognition(): SpeechRecognition | null {
  if (typeof window === "undefined") return null;
  const RecognitionCtor = window.SpeechRecognition ?? window.webkitSpeechRecognition;
  if (!RecognitionCtor) return null;

  const recognition = new RecognitionCtor();
  recognition.lang = "en-US";
  recognition.interimResults = true;
  recognition.continuous = false;
  recognition.maxAlternatives = 1;
  return recognition;
}

export function describeRecognitionError(error: string): string {
  switch (error) {
    case "no-speech":
      return "Didn't catch that — try again.";
    case "audio-capture":
      return "No microphone was found.";
    case "not-allowed":
    case "service-not-allowed":
      return "Microphone access was denied. Allow it in your browser settings and try again.";
    case "network":
      return "A network error interrupted voice recognition.";
    case "aborted":
      return "Voice input was stopped.";
    default:
      return `Voice recognition error: ${error}`;
  }
}

export function speak(text: string, onEnd: () => void, onError: () => void): void {
  if (!isSpeechSynthesisSupported()) {
    onError();
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.onend = onEnd;
  utterance.onerror = onError;
  window.speechSynthesis.speak(utterance);
}

export function cancelSpeech(): void {
  if (isSpeechSynthesisSupported()) {
    window.speechSynthesis.cancel();
  }
}
