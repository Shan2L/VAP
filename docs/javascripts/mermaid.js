document$.subscribe(async () => {
  const colorScheme = document.body.getAttribute("data-md-color-scheme");
  mermaid.initialize({
    startOnLoad: false,
    theme: colorScheme === "slate" ? "dark" : "default",
  });
  await mermaid.run({
    nodes: document.querySelectorAll(".mermaid"),
  });
});
