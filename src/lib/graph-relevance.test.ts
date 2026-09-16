import { beforeEach, describe, expect, it, vi } from "vitest"
import type { FileNode } from "@/types/wiki"

const { listDirectory, readFile } = vi.hoisted(() => ({
  listDirectory: vi.fn(),
  readFile: vi.fn(),
}))

vi.mock("@/commands/fs", () => ({ listDirectory, readFile }))

import { buildRetrievalGraph, clearGraphCache } from "./graph-relevance"

function files(root: string, count: number): FileNode[] {
  return Array.from({ length: count }, (_, index) => ({
    name: `page-${index}.md`,
    path: `${root}/wiki/page-${index}.md`,
    is_dir: false,
  }))
}

beforeEach(() => {
  clearGraphCache()
  listDirectory.mockReset()
  readFile.mockReset()
})

describe("buildRetrievalGraph", () => {
  it("reads wiki pages concurrently without exceeding the worker limit", async () => {
    listDirectory.mockResolvedValue(files("/project", 40))
    let active = 0
    let peak = 0
    readFile.mockImplementation(async () => {
      active += 1
      peak = Math.max(peak, active)
      await new Promise((resolve) => setTimeout(resolve, 1))
      active -= 1
      return "---\ntype: concept\n---\n# Page"
    })

    const graph = await buildRetrievalGraph("/project", 1)

    expect(graph.nodes.size).toBe(40)
    expect(peak).toBeGreaterThan(1)
    expect(peak).toBeLessThanOrEqual(16)
  })

  it("does not reuse a same-version cache entry across projects", async () => {
    listDirectory.mockImplementation(async (root: string) => files(root.replace(/\/wiki$/, ""), 1))
    readFile.mockImplementation(async (path: string) => `# ${path.includes("project-a") ? "A" : "B"}`)

    const a = await buildRetrievalGraph("/project-a", 7)
    const b = await buildRetrievalGraph("/project-b", 7)

    expect(a.nodes.values().next().value?.title).toBe("A")
    expect(b.nodes.values().next().value?.title).toBe("B")
    expect(readFile).toHaveBeenCalledTimes(2)
  })

  it("includes frontmatter related entries in retrieval links", async () => {
    listDirectory.mockResolvedValue([
      ...files("/project", 2),
    ])
    readFile.mockImplementation(async (path: string) =>
      path.endsWith("page-0.md")
        ? "---\nrelated: [page-1]\n---\n# Source"
        : "# Target",
    )

    const graph = await buildRetrievalGraph("/project", 2)

    expect(graph.nodes.get("page-0")?.outLinks).toContain("page-1")
    expect(graph.nodes.get("page-1")?.inLinks).toContain("page-0")
  })
})
