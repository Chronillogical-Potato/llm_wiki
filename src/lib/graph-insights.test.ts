import { describe, expect, it } from "vitest"
import { detectKnowledgeGaps } from "./graph-insights"
import type { CommunityInfo, GraphNode } from "./wiki-graph"

function nodes(count: number): GraphNode[] {
  return Array.from({ length: count }, (_, index) => ({
    id: `node-${index}`,
    label: `Node ${index}`,
    type: "concept",
    path: `/wiki/node-${index}.md`,
    linkCount: 4,
    community: 0,
  }))
}

function community(overrides: Partial<CommunityInfo>): CommunityInfo {
  return {
    id: 0,
    nodeCount: 100,
    cohesion: 0.02,
    meanIntraDegree: 2,
    topNodes: ["Node 0"],
    ...overrides,
  }
}

describe("detectKnowledgeGaps sparse communities", () => {
  it("does not flag a large community whose mean internal degree is healthy", () => {
    const gaps = detectKnowledgeGaps(nodes(100), [], [community({})])
    expect(gaps.some((gap) => gap.type === "sparse-community")).toBe(false)
  })

  it("flags a community averaging fewer than two internal links per page", () => {
    const gaps = detectKnowledgeGaps(nodes(20), [], [community({
      nodeCount: 20,
      meanIntraDegree: 1.4,
    })])
    expect(gaps.find((gap) => gap.type === "sparse-community")?.description)
      .toContain("1.4 internal links per page")
  })
})
