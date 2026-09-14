export type SectionId =
  | "home"
  | "about"
  | "experience"
  | "ai_projects"
  | "other_projects"
  | "skills"
  | "education"
  | "certifications"
  | "resume"
  | "github"
  | "contact";

export interface NavigateAction {
  type: "navigate_to_section";
  target: SectionId;
}

const VALID_SECTION_IDS: ReadonlySet<string> = new Set<SectionId>([
  "home",
  "about",
  "experience",
  "ai_projects",
  "other_projects",
  "skills",
  "education",
  "certifications",
  "resume",
  "github",
  "contact",
]);

function isValidNavigateAction(action: unknown): action is NavigateAction {
  if (typeof action !== "object" || action === null) return false;
  const candidate = action as Record<string, unknown>;
  return (
    candidate.type === "navigate_to_section" &&
    typeof candidate.target === "string" &&
    VALID_SECTION_IDS.has(candidate.target)
  );
}

/**
 * The only entry point for acting on an LLM-requested action. Re-validates
 * against a local allow-list (defense in depth against a mismatched or
 * tampered backend response) and, for a valid navigation action, does
 * nothing but scroll to a known, pre-existing element id. No eval, no
 * dynamic selector/URL construction, no innerHTML.
 */
export function executeAction(action: unknown): void {
  if (!isValidNavigateAction(action)) return;
  document.getElementById(action.target)?.scrollIntoView({ behavior: "smooth" });
}
