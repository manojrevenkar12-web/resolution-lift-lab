import { render, screen } from "@testing-library/react";
import { vi } from "vitest";

import report from "../test/report.fixture.json";
import { App } from "./App";

function serve(status: number, body?: unknown) {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(body ? JSON.stringify(body) : null, { status })));
}

describe("dashboard", () => {
  it("answers the three questions from the report", async () => {
    serve(200, report);
    render(<App />);
    expect(await screen.findByRole("heading", { name: "How many billed resolutions were real?" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Did a content fix raise resolutions?" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "What do customers ask that no article answers?" })).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /Prediction-powered inference/ })).toHaveTextContent("99%");
    expect(screen.getByText(/true average/)).toBeInTheDocument();
  });

  it("says how to make a report when there is none", async () => {
    serve(404);
    render(<App />);
    expect(await screen.findByText(/No held-out report yet/)).toBeInTheDocument();
  });

  it("offers a retry when loading fails", async () => {
    serve(500);
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("HTTP 500");
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });
});
