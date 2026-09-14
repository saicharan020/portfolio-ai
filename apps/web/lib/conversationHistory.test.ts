import { describe, expect, it } from "vitest";
import { boundHistory, MAX_HISTORY_TURNS, type ConversationTurn } from "./conversationHistory";

function turns(n: number): ConversationTurn[] {
  return Array.from({ length: n }, (_, i) => ({
    role: i % 2 === 0 ? "user" : "assistant",
    text: `turn ${i}`,
  }));
}

describe("boundHistory", () => {
  it("returns an empty array for empty input", () => {
    expect(boundHistory([])).toEqual([]);
  });

  it("returns the input unchanged when shorter than the max", () => {
    const history = turns(3);
    expect(boundHistory(history)).toEqual(history);
  });

  it("returns the input unchanged when exactly at the max", () => {
    const history = turns(MAX_HISTORY_TURNS);
    expect(boundHistory(history)).toEqual(history);
  });

  it("keeps only the most recent `max` turns, oldest-first, when over the max", () => {
    const history = turns(20);

    const result = boundHistory(history);

    expect(result).toHaveLength(MAX_HISTORY_TURNS);
    expect(result).toEqual(history.slice(-MAX_HISTORY_TURNS));
    expect(result[0].text).toBe(`turn ${20 - MAX_HISTORY_TURNS}`);
    expect(result[result.length - 1].text).toBe("turn 19");
  });

  it("respects a custom max", () => {
    const history = turns(10);

    const result = boundHistory(history, 3);

    expect(result).toEqual(history.slice(-3));
  });

  it("returns an empty array for a non-positive max", () => {
    expect(boundHistory(turns(5), 0)).toEqual([]);
  });

  it("does not mutate the input array", () => {
    const history = turns(20);
    const copy = [...history];

    boundHistory(history);

    expect(history).toEqual(copy);
  });
});
