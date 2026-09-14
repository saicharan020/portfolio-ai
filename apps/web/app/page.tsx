"use client";

import { type ReactNode, useState } from "react";
import { executeAction } from "@/lib/actions";
import { sendChatMessage } from "@/lib/apiClient";
import { CaptionBar } from "@/components/voice/CaptionBar";
import { VoiceOrb } from "@/components/voice/VoiceOrb";
import { useVoiceAssistant } from "@/components/voice/useVoiceAssistant";

// Mirrors apps/api/content/*.md (Phase 5 real content). Keep these in sync
// with the RAG source files if that content ever changes.

const ABOUT_PARAGRAPHS = [
  "Saicharan Reddy Muthyala is an AI/ML Engineer with hands-on experience building and deploying generative AI and machine learning systems, including LLM-powered applications, retrieval-augmented generation (RAG), and production MLOps pipelines.",
  "Skilled across the full ML lifecycle — data engineering, model development, API integration, and cloud deployment — using Python, LangChain, Hugging Face Transformers, and modern MLOps tooling.",
  "Background in backend software development strengthens the ability to design scalable, well-engineered AI systems end to end.",
  "Focused on delivering reliable, high-performance AI solutions that integrate cleanly into real-world products.",
];

const EXPERIENCE = [
  {
    title: "AI/ML Engineer — Chase",
    location: "USA",
    dates: "August 2025 – June 2026",
    bullets: [
      "Engineered and deployed LLM-powered AI applications using Python, LangChain, Hugging Face Transformers, and vector databases, enabling semantic search, document question answering, and enterprise knowledge retrieval with RAG.",
      "Developed scalable REST APIs with FastAPI to serve machine learning and generative AI models, reducing inference latency by 30% and supporting integration with web applications and internal platforms.",
      "Built automated MLOps workflows using MLflow, Docker, GitHub Actions, and Kubernetes to streamline model versioning, testing, and production deployments, reducing release time by 35%.",
      "Designed data preprocessing and feature engineering pipelines using Pandas, NumPy, SQL, and Apache Spark, improving data quality and increasing model performance by 15% across predictive analytics workloads.",
      "Partnered with cross-functional product and engineering teams to evaluate, fine-tune, and monitor AI models, implementing automated performance monitoring and prompt optimization that improved response quality and system reliability.",
    ],
  },
  {
    title: "Python Developer — Tech Silicon",
    location: "Hyderabad, India",
    dates: "January 2022 – December 2023",
    bullets: [
      "Developed and maintained 20+ RESTful APIs using Python and Flask, reducing average response time by 35% and supporting high-volume business applications.",
      "Designed and optimized MySQL database queries and backend services, improving application performance by 45% while reducing feature development time by 30% through modular architecture.",
      "Built automated Python data-processing pipelines handling 100,000+ records daily and automated operational workflows, reducing processing time by 60% and saving 25+ hours of manual effort each week.",
      "Implemented secure authentication using JWT, integrated third-party APIs, and collaborated with cross-functional Agile teams to deliver scalable and reliable software with 99.9% application availability.",
      "Containerized and deployed applications on Linux using Docker, Git, and CI/CD pipelines, reducing deployment time from hours to under 15 minutes while increasing code quality through unit testing and code reviews.",
    ],
  },
];

const AI_PROJECTS = [
  {
    title: "Voice-First AI Portfolio",
    note: "This is the current portfolio project.",
    bullets: [
      "Voice-first AI portfolio designed for recruiter interaction.",
      "Next.js and TypeScript frontend.",
      "Python and FastAPI backend.",
      "Browser-based Speech-to-Text and Text-to-Speech.",
      "Ollama used for local LLM development.",
      "Retrieval-Augmented Generation using portfolio Markdown content.",
      "Embeddings generated with Ollama and nomic-embed-text.",
      "In-memory cosine-similarity retrieval using NumPy.",
      "Controlled AI navigation that can navigate recruiters to relevant portfolio sections.",
      "AI responses are grounded in verified portfolio information.",
      "Controlled actions are used instead of arbitrary browser, JavaScript, shell, or DOM execution.",
      "Architecture is designed so the development LLM/provider can be changed later.",
      "Project is being developed incrementally with testing at each phase.",
    ],
  },
  {
    title: "Voice-Based Email System for Visually Impaired Users",
    bullets: [
      "Built a Python-based voice-controlled email application enabling visually impaired users to compose, send, read, and manage emails using Speech-to-Text and Text-to-Speech technologies.",
      "Integrated speech recognition and voice synthesis to deliver a hands-free, accessibility-focused user experience.",
      "Implemented voice-driven email workflows including composing messages, reading inbox content aloud, and executing email actions through spoken commands.",
      "Improved speech recognition reliability through input validation and error handling while following accessibility-focused design principles.",
    ],
  },
];

const SKILLS = [
  { category: "Languages", items: ["Python", "SQL"] },
  {
    category: "AI/ML & GenAI",
    items: [
      "PyTorch",
      "TensorFlow",
      "Scikit-learn",
      "LangChain",
      "Hugging Face Transformers",
      "RAG",
      "Prompt Engineering",
      "Vector Databases",
    ],
  },
  {
    category: "MLOps & Deployment",
    items: ["MLflow", "Docker", "Kubernetes", "FastAPI", "Flask", "GitHub Actions", "CI/CD"],
  },
  {
    category: "Cloud Platforms",
    items: ["AWS", "Amazon SageMaker", "Amazon EC2", "Amazon S3", "AWS Lambda"],
  },
  {
    category: "Data Engineering",
    items: ["Pandas", "NumPy", "Apache Spark", "MySQL"],
  },
  {
    category: "Software Engineering",
    items: ["REST APIs", "JWT Authentication", "Git", "Agile/Scrum", "Unit Testing"],
  },
];

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section
      id={id}
      className="scroll-mt-6 rounded-md border border-zinc-300 bg-white p-6 dark:border-zinc-700 dark:bg-zinc-900"
    >
      <h2 className="text-lg font-semibold text-black dark:text-zinc-50">{title}</h2>
      <div className="mt-2 space-y-3 text-sm text-zinc-600 dark:text-zinc-400">{children}</div>
    </section>
  );
}

function BulletList({ items }: { items: string[] }) {
  return (
    <ul className="list-disc space-y-1 pl-5">
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

export default function Home() {
  const voice = useVoiceAssistant();

  const [message, setMessage] = useState("");
  const [reply, setReply] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSend() {
    if (!message.trim()) return;
    setLoading(true);
    setError(null);
    setReply(null);
    try {
      const res = await sendChatMessage(message);
      setReply(res.reply);
      executeAction(res.action);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-zinc-50 px-6 py-12 font-sans dark:bg-black">
      <main id="home" className="flex w-full max-w-xl flex-col gap-4">
        <h1 className="text-2xl font-semibold text-black dark:text-zinc-50">
          Portfolio AI — Phase 2 test
        </h1>

        <section className="flex flex-col items-center gap-4 rounded-md border border-zinc-300 bg-white p-6 dark:border-zinc-700 dark:bg-zinc-900">
          <VoiceOrb
            status={voice.status}
            supported={voice.supported}
            onStart={voice.start}
            onStop={voice.stop}
          />
          <CaptionBar messages={voice.messages} interimTranscript={voice.interimTranscript} />
          {voice.errorMessage && (
            <p className="w-full rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
              {voice.errorMessage}
            </p>
          )}
        </section>

        <div className="flex items-center gap-3 text-xs uppercase tracking-wide text-zinc-500 dark:text-zinc-500">
          <div className="h-px flex-1 bg-zinc-300 dark:bg-zinc-700" />
          Or type instead
          <div className="h-px flex-1 bg-zinc-300 dark:bg-zinc-700" />
        </div>

        <textarea
          className="min-h-24 w-full rounded-md border border-zinc-300 bg-white p-3 text-black dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50"
          placeholder="Ask something..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
        />

        <button
          onClick={handleSend}
          disabled={loading || !message.trim()}
          className="w-fit rounded-full bg-foreground px-5 py-2 text-sm font-medium text-background transition-colors hover:bg-[#383838] disabled:opacity-50 dark:hover:bg-[#ccc]"
        >
          {loading ? "Sending..." : "Send"}
        </button>

        {error && (
          <p className="rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
            {error}
          </p>
        )}

        {reply && (
          <div className="rounded-md border border-zinc-300 bg-white p-3 text-sm text-black dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50">
            {reply}
          </div>
        )}

        <Section id="about" title="About">
          {ABOUT_PARAGRAPHS.map((paragraph) => (
            <p key={paragraph}>{paragraph}</p>
          ))}
        </Section>

        <Section id="experience" title="Experience">
          {EXPERIENCE.map((job) => (
            <div key={job.title}>
              <h3 className="font-medium text-black dark:text-zinc-50">{job.title}</h3>
              <p className="text-xs text-zinc-500 dark:text-zinc-500">
                {job.location} · {job.dates}
              </p>
              <div className="mt-1">
                <BulletList items={job.bullets} />
              </div>
            </div>
          ))}
        </Section>

        <Section id="ai_projects" title="AI Projects">
          {AI_PROJECTS.map((project) => (
            <div key={project.title}>
              <h3 className="font-medium text-black dark:text-zinc-50">{project.title}</h3>
              {project.note && (
                <p className="text-xs italic text-zinc-500 dark:text-zinc-500">{project.note}</p>
              )}
              <div className="mt-1">
                <BulletList items={project.bullets} />
              </div>
            </div>
          ))}
        </Section>

        <Section id="other_projects" title="Other Projects">
          <p>
            Additional project information has not yet been added to this portfolio. The AI/ML
            projects currently documented are covered in the AI Projects section.
          </p>
        </Section>

        <Section id="skills" title="Skills">
          {SKILLS.map((group) => (
            <div key={group.category}>
              <h3 className="font-medium text-black dark:text-zinc-50">{group.category}</h3>
              <p className="mt-1">{group.items.join(", ")}</p>
            </div>
          ))}
        </Section>

        <Section id="education" title="Education">
          <div>
            <h3 className="font-medium text-black dark:text-zinc-50">
              Master of Science in Computer &amp; Information Science
            </h3>
            <p className="text-xs text-zinc-500 dark:text-zinc-500">
              Concordia University Wisconsin · March 2024 – May 2025
            </p>
          </div>
        </Section>

        <Section id="certifications" title="Certifications">
          <p>Certification information has not yet been added to this portfolio.</p>
        </Section>

        <Section id="resume" title="Resume">
          <div>
            <h3 className="font-medium text-black dark:text-zinc-50">
              Saicharan Reddy Muthyala — AI/ML Engineer
            </h3>
            <p className="mt-1">
              AI/ML Engineer with hands-on experience building and deploying generative AI and
              machine learning systems, including LLM-powered applications, retrieval-augmented
              generation (RAG), and production MLOps pipelines. Skilled across the full ML
              lifecycle — data engineering, model development, API integration, and cloud
              deployment.
            </p>
          </div>
          <p>
            See the Experience, Education, Skills, and AI Projects sections above for full
            details.
          </p>
        </Section>

        <Section id="github" title="GitHub">
          <p>
            <a
              href="https://github.com/saicharan020"
              target="_blank"
              rel="noreferrer"
              className="text-black underline dark:text-zinc-50"
            >
              github.com/saicharan020
            </a>
          </p>
        </Section>

        <Section id="contact" title="Contact">
          <p>
            Saicharan Reddy Muthyala
            <br />
            Milwaukee, Wisconsin, USA
          </p>
          <p>
            <a
              href="mailto:saicharanmuthyala020@gmail.com"
              className="text-black underline dark:text-zinc-50"
            >
              saicharanmuthyala020@gmail.com
            </a>
            <br />
            <a href="tel:+14143815605" className="text-black underline dark:text-zinc-50">
              (414) 381-5605
            </a>
          </p>
        </Section>
      </main>
    </div>
  );
}
