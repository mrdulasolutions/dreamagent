# Security Policy

## Supported Versions

DreamAgent is pre-1.0. Only the `main` branch is supported. Security fixes
are applied at HEAD.

| Version | Supported |
|---|---|
| `main` | ✅ |
| Tagged releases | only the latest |

## Reporting a Vulnerability

**Do not file a public GitHub issue.**

For security-sensitive reports, please email `security@mrdula.solutions`
with:

- A description of the issue
- Reproduction steps (or a proof of concept)
- The git SHA you're testing against
- Your assessment of impact

We'll acknowledge within 72 hours and follow up with an estimated remediation
timeline. We aim to fix critical issues within 30 days.

## Threat Model

DreamAgent runs on the user's machine and produces local model artifacts.
The primary security considerations are:

### 1. Memory data is sensitive by definition

Memories may include personal information, credentials, contact data, and
behavioral preferences. The extraction prompt instructs the LLM to refuse
extracting secrets (SSNs, passwords, private keys), but we cannot guarantee
zero leakage — the upstream text source can contain anything.

**Best practices:**
- Use the `sensitivity: redact` flag on memories you don't want trained.
- Run `dreamagent extract` against your own data only.
- Don't share trained adapter `.safetensors` files publicly without first
  probing them for memorized PII.
- For maximum privacy, use the `ollama` extraction backend (local-only)
  rather than `anthropic` or `openai`.

### 2. Frontier API keys

`ANTHROPIC_API_KEY` and `OPENAI_API_KEY` give billing-level access to the
respective accounts. Treat them like passwords. Never commit them. The
project's `.gitignore` excludes `.env` and `.env.local`.

### 3. Subprocess invocation

The train stage invokes `python -m mlx_lm lora` as a subprocess. We control
the argument list — no user input is interpolated into a shell. But if
you're vendoring DreamAgent into a larger system, audit the entry points
you expose.

### 4. Adapter integrity

There's currently no signing on adapter files. A malicious adapter could
in theory leak data via inference. If you're running someone else's
adapter, treat it like running their code.

### 5. Memory injection

If your upstream memory source is untrusted (e.g., a multi-user chat
system), a malicious memory could try to inject instructions into the
trained model. The extraction prompt is hardened against this, but the
fundamental defense is: only trust memories from sources you trust.

## Responsible Disclosure

We follow standard responsible disclosure practice:

1. You report privately
2. We acknowledge within 72 hours
3. We coordinate a fix with you
4. We publish a fix and credit the reporter (unless you prefer anonymity)
5. We add a CHANGELOG entry with the CVE if applicable

Thank you for helping keep DreamAgent safe.
