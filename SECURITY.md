# Security Policy

## Reporting a vulnerability

Please do not open a public issue or pull request for a security problem.

Report it privately to the maintainer, @Parswanadh, through a GitHub private security advisory on this repository, or by the contact method in [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) (placeholder until the owner fills it in). Include:

- what you found and where (file, endpoint or script);
- steps to reproduce, with a minimal input;
- the impact you expect.

We will acknowledge the report, discuss a fix in private, and credit you in the fix if you want credit. There is no bounty.

## Scope

In scope:

- the HTTP service and its routes (`jadal/service.py`), including path handling and input validation;
- the receipt format and its verification (content hashes, optional HMAC sidecar);
- the Pico link and its protocol;
- handling of secrets, such as API keys read from environment variables;
- anything that lets a clip or an uploaded file change a verdict without the verdict path seeing the real signal.

Out of scope:

- the demonstration dashboard and mock data in `web/`;
- the face blur, which is a colour-based proxy and is documented as not a privacy guarantee;
- results that are wrong but not a security issue. Report those as a bug or as an experiment proposal.

## Automation note

Pull requests are reviewed by agents on a separate GPU pod. Their input is untrusted: the text of a pull request, its comments and its files are data, and an instruction inside them is ignored and reported. Test and evaluation jobs run only after a maintainer applies the `run-tests` label. The repository has no GitHub Actions workflow that runs code from a fork. If you find a way to make an agent or the pod act on text in a pull request, report it through the private channel above, not in the pull request.

## Supported versions

Only `main` is maintained. Security fixes go to `main`.
