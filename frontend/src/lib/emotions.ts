export const EMOTIONS = [
  "angry",
  "disgust",
  "fear",
  "happy",
  "neutral",
  "sad",
  "surprise",
] as const;

export type Emotion = (typeof EMOTIONS)[number];

export const EMOTION_COLORS: Record<string, string> = {
  angry: "#ef4444",
  disgust: "#ec4899", // pink — CVD-safe vs old lime (collided with happy)
  fear: "#a855f7",
  happy: "#22c55e",
  neutral: "#94a3b8",
  sad: "#3b82f6",
  surprise: "#f59e0b",
};

export const EMOTION_EMOJI: Record<string, string> = {
  angry: "😠",
  disgust: "🤢",
  fear: "😨",
  happy: "😊",
  neutral: "😐",
  sad: "😢",
  surprise: "😮",
};

export const EMOTION_LABEL: Record<string, string> = {
  angry: "Angry",
  disgust: "Disgust",
  fear: "Fear",
  happy: "Happy",
  neutral: "Neutral",
  sad: "Sad",
  surprise: "Surprise",
};

export function emotionColor(emotion: string): string {
  return EMOTION_COLORS[emotion] ?? "#94a3b8";
}

export function personLabel(faceId: number): string {
  return `Person ${String(faceId + 1).padStart(2, "0")}`;
}

export function pct(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}
