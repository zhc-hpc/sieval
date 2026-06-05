"""IFBench instruction-following verifiers, vendored from allenai/IFBench.

Upstream: https://github.com/allenai/IFBench
Pinned commit: 1091c4c3de6c1f6ed12c012ed68f11ea450b0117 (2026-05-04)
Files vendored verbatim: instructions.py, instructions_registry.py,
instructions_util.py, evaluation_lib.py.

Sole adaptation: the upstream flat-layout imports (`import instructions`,
`import instructions_registry`, `import instructions_util`) are rewritten to
package-relative imports (`from . import ...`) so the modules load as a package.
Prompt templates, constraint verifiers, and strict/loose scoring are unchanged.

IFBench (Pyatkin et al., NeurIPS 2025 D&B; arXiv:2507.02833) — 58 new
out-of-domain verifiable constraints; distinct from the Google IFEval
verifiers vendored under ../instruction_following_eval/.
"""
