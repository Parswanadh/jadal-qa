type Attrs = Record<string, string | number | boolean | undefined | ((ev: Event) => void)>;
type Child = Node | string | null | undefined | false;

function applyAttrs(node: Element, attrs: Attrs): void {
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === false) continue;
    if (typeof value === "function") {
      if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      continue;
    }
    if (value === true) {
      node.setAttribute(key, "");
      continue;
    }
    node.setAttribute(key === "className" ? "class" : key, String(value));
  }
}

function appendChildren(node: Node, children: Child[]): void {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
}

/** Build an HTML element. Text is always set via textContent, never innerHTML. */
export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  applyAttrs(node, attrs);
  appendChildren(node, children);
  return node;
}

const SVG_NS = "http://www.w3.org/2000/svg";

/** Build an SVG element in the SVG namespace. */
export function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  ...children: Child[]
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  applyAttrs(node, attrs);
  appendChildren(node, children);
  return node;
}
