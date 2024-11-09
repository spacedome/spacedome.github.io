---
author: "Julien"
desc: "Rewriting my website in Hakyll"
keywords: "announcement, haskell, blog"
lang: "en"
title: "Moving from Hugo to Hakyll"
updated: "2024-11-09 10:59"
---

Five years ago I decided to rewrite this website with a static site generator, and chose Hugo on a whim.
After letting it sit around unused for far too long, I moved it to [Hakyll](https://jaspervdj.be/hakyll/).
Now that I'm getting back into Haskell, it was a perfect small project, and Pandoc is unbeatable as a document conversion backend. 

The built in code highlighter works well enough if you override some of the styling.
It is aparently also possible to use pygmentize if you want to go through the trouble.

```haskell
toSlug :: T.Text -> T.Text
toSlug =
  T.intercalate (T.singleton '-') . T.words . T.toLower . clean
```

I chose a perfect time to move, as MathML finally has full support across all browsers as of Chromium 109, so the whole MathJax/KaTeX and JS rendering question is finally over. 
We are free!
Here is an example and the associated markdown:

Inline math with MathML: \\( y = mx +b \\).
Block/display style:
\\[ \ln x = \int_{-\infty}^x \frac 1 y \, \mathrm{d}y . \\]

```text
Inline math with MathML: \\( y = mx +b \\). Block/display style:
\\[ \ln x = \int_{-\infty}^x \frac 1 y \, \mathrm{d}y . \\]
```

This is as easy as enabling the `Ext_tex_math_double_backslash` Pandoc extension and setting `writerHTMLMathMethod = MathML` in the Pandoc writer options.
It is significantly uglier than what TeX would give you, but this is the price you pay if you want to avoid javascript or prerendering SVGs.

\\[\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right) \\]
