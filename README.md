# Inhibitory earthquake models

Research manuscript and numerical implementation for **Stability and identification in inhibitory earthquake models: Restricted mathematics, nuisance sensitivity and predictive evidence**.

The work studies restricted stability and identification results, nuisance sensitivity, and predictive evidence for earthquake point-process models with inhibition and triggering. Predictive preference does not establish a physical inhibition mechanism. This is a research release for independent review.

**Collaborators:** [Drefetr](https://github.com/Drefetr), [magna-nz](https://github.com/magna-nz); [Claude Opus 5.5](https://www.anthropic.com/claude-opus-5-5), [DeepSeek 4.1 Flash](https://www.deepseek.com/en/news/deepseek-v4-1-flash/), [Gemini 4.0 Argon](https://ai.google.dev/gemini-api/docs/models/gemini-4-argon), [GPT 6.0 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra), [GPT 6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol).

**AI assistance:** the bulk of the work was performed by AI agents. Claude Opus 5.5 and GPT 6.0 Astra developed the initial outline and specification; GPT 6.1 Sol coordinated GPT 6.1 Sol and DeepSeek 4.1 Flash subagents; GPT 6.0 Astra and Gemini 4.0 Argon provided review. AI review is distinct from independent external peer review. See [CONTRIBUTORS.md](CONTRIBUTORS.md).

- [Paper](paper/manuscript.pdf), [explanatory supplement](paper/supplements/explanatory_supplement.pdf), and [F nuisance addendum](paper/supplements/f_nuisance_addendum.pdf).
- [F follow-up scope and findings](docs/F_followup.md): a separately labelled, incomplete continuation; historical results remain preserved.
- [Reproduction instructions](docs/reproducibility.md), [methods](paper/supplements/methods_supplement.md), [data sources and terms](docs/data.md), [limitations](docs/limitations.md).
- [Contribution guidelines](CONTRIBUTING.md) and [citation metadata](CITATION.cff).

## Quick checks

Python 3.13 was used for the local preparation checks. From the repository root:

```sh
python -m venv .venv
# Activate the environment using your shell's usual command.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/check_mathematics.py --output build/mathematics.json
python scripts/verify_repository.py
```

These checks use deterministic inputs; they do not run fitting or simulations. Dependency pins describe the tested public preparation environment, not the recovered historical scientific environment.

## Build and figures

Install Tectonic 0.17.0 from its [official releases](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0), then:

```sh
python scripts/build_paper.py --output build/paper
python scripts/make_figures.py --output build/figures
```

Use `--engine /path/to/tectonic` if needed. Output directories must be new. The paper build uses shipped figure PDFs. Figure regeneration uses compact project-derived results, including aggregate catalogue bins; it writes separately and never downloads a catalogue. Different TeX resources may change pagination.

## Contents and reproduction scope

`paper/` contains LaTeX, original figures, aggregate plotting inputs and selected fit/settings summaries. `engine/` contains preserved numerical implementations and the accepted observed protocol's small source dependency set. Historical relative paths remain because source hashes bind the implementations. [The code guide](docs/code.md) documents the import facade and supported entry points.

Third-party event records and the earlier third-party manuscript PDF are excluded. Obtain external inputs independently under their terms. Exact historical observed replay needs the original matching snapshot; a fresh provider download can differ. A complete rerun of every historical synthetic campaign also needs unshipped history banks and execution states. The reproduction matrix makes these limits explicit.

## Licences

Project-owned software: **MIT**. Current paper, original figures, prose documentation and project-derived result summaries: **CC BY 4.0**. These are separate scopes, not a blanket licence over third-party material. See [licence scope](LICENSES/README.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

Sealed v4 remains unchanged. This public edition adds confirmed attribution and disclosure and documents the smaller distribution. See [provenance](docs/provenance.md) and [validation](validation/README.md).

Publication privacy checks are recorded in [validation/privacy_audit.json](validation/privacy_audit.json). The only engine adaptations are portable interpreter discovery in two optional launchers and the associated public protocol binding; original scientific pins remain separately recorded.

See the [manuscript change note](docs/changes.md) and [public source guide](docs/public_readability.md).
