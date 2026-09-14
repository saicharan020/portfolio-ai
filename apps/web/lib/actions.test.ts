import { afterEach, describe, expect, it, vi } from "vitest";
import { executeAction, type NavigateAction } from "./actions";

describe("executeAction", () => {
  afterEach(() => {
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("scrolls to the matching element for a valid action", () => {
    const el = document.createElement("section");
    el.id = "experience";
    document.body.appendChild(el);
    const scrollSpy = vi.fn();
    el.scrollIntoView = scrollSpy;

    const action: NavigateAction = { type: "navigate_to_section", target: "experience" };
    executeAction(action);

    expect(scrollSpy).toHaveBeenCalledWith({ behavior: "smooth" });
  });

  it("does nothing when the target element does not exist in the DOM", () => {
    const getByIdSpy = vi.spyOn(document, "getElementById");
    const action: NavigateAction = { type: "navigate_to_section", target: "resume" };

    expect(() => executeAction(action)).not.toThrow();
    expect(getByIdSpy).toHaveBeenCalledWith("resume");
  });

  it.each([
    { type: "navigate_to_section", target: "not-a-real-section" },
    { type: "navigate_to_section", target: "<script>alert(1)</script>" },
    { type: "navigate_to_section", target: "javascript:alert(1)" },
    { type: "navigate_to_section", target: "" },
    { type: "navigate_to_section" },
    { type: "delete_everything", target: "home" },
    { target: "home" },
    null,
    undefined,
    "home",
    42,
    ["home"],
  ])("is a no-op for invalid action %j", (invalidAction) => {
    const getByIdSpy = vi.spyOn(document, "getElementById");

    expect(() => executeAction(invalidAction)).not.toThrow();
    expect(getByIdSpy).not.toHaveBeenCalled();
  });

  it("never touches window.location", () => {
    const originalHref = window.location.href;
    executeAction({ type: "navigate_to_section", target: "contact" });
    expect(window.location.href).toBe(originalHref);
  });
});
