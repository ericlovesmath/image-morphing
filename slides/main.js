import "reveal.js/dist/reveal.css";
import "reveal.js/dist/theme/black.css";
import "reveal.js/plugin/highlight/monokai.css";

import Reveal from "reveal.js";
import Markdown from "reveal.js/plugin/markdown/markdown.esm.js";
import KaTeX from "reveal.js/plugin/math/math.esm";
import RevealHighlight from "reveal.js/plugin/highlight/highlight.esm";

Reveal.initialize({ plugins: [Markdown, KaTeX, RevealHighlight] });
