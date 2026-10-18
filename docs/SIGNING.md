# Signing a manifest

A `.merkle.json` manifest commits to every tensor of a model, but nothing in
it says who produced it. Anyone who can replace the weights can also write a
new manifest. Signing the manifest closes that gap: once its signature checks
out, `merkle-verify` turns it into a verdict on the weights, a diff, and a
repair plan.

The manifest is small (kilobytes to a few megabytes; `--compact` makes it
smaller), so signing it is fast even for very large models, and the weights
are hashed only once, by `merkle-verify verify`.

## With Sigstore model signing

Tested with `model-signing` 1.1.1 (`pip install model-signing`).

Publisher:

```bash
merkle-verify sign ./my-model/                          # writes ./my-model/model.merkle.json
model_signing sign ./my-model/model.merkle.json --signature model.merkle.sig
```

The default `sigstore` method opens an OIDC login in the browser and records
the identity in the signature bundle. Publish `model.merkle.sig` with the model.

Consumer:

```bash
model_signing verify ./my-model/model.merkle.json \
    --signature model.merkle.sig \
    --identity publisher@example.com \
    --identity_provider https://accounts.google.com
merkle-verify verify ./my-model/
```

The first command authenticates the manifest. The second re-hashes the weights
against it and names any tensor that differs.

You can also sign the whole model directory (`model_signing sign ./my-model/`).
That signature covers `model.merkle.json` too (editing the manifest makes
`model_signing verify` fail), but it hashes every weight file during signing
and verification, so the weights get hashed twice.

## With cosign

`cosign sign-blob` signs a single file in the same keyless way. This
repository's tests do not exercise cosign; the commands follow the cosign 2.x
documentation:

```bash
cosign sign-blob ./my-model/model.merkle.json --bundle model.merkle.json.sigstore.json
cosign verify-blob ./my-model/model.merkle.json \
    --bundle model.merkle.json.sigstore.json \
    --certificate-identity publisher@example.com \
    --certificate-oidc-issuer https://accounts.google.com
```

## Without signatures: pin the root

If a signing infrastructure is too much, publish the model root through a
channel you control (the model card, a release note, a git tag), including its
algorithm:

```bash
merkle-verify info ./my-model/model.merkle.json         # root and algorithm
merkle-verify verify ./my-model/ --expected-root sha256:<root>
```

## Repairing from an untrusted mirror

An authenticated manifest also makes any mirror usable for repair. Every chunk
fetched from the mirror is checked against the manifest before it is written:

```bash
merkle-verify repair ./my-model/ --from hf://org/model@<commit>
merkle-verify repair ./my-model/ --from https://mirror.example.com/my-model/
```
