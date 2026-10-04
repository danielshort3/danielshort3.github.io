# In-app game content updates

The standalone Guild app can update its game files while the Android application stays open. Content updates are separate from Android package updates: the native shell, save checkpoint bridge, billing/advertisement bridges, permissions and updater itself still require a normally installed APK. The first supported native API is `1`, introduced in APK version code `16`; its packaged game is content version `1`.

Game updates keep the origin `https://appassets.androidplatform.net/assets/wayfarers/index.html`. This preserves the existing WebView save namespace. The native app owns downloading, signature verification, inventory validation, private staging and activation. The web game does not download arbitrary scripts or handle the release signing key.

## Release ownership

The authoritative game sources remain `pages/games/wayfarers-guild.html`, `js/games/wayfarers-guild/`, `css/games/wayfarers-guild.css`, and `img/wayfarers-guild/`. `build/bundle-wayfarers-android.cjs` builds the same offline application used by the APK. `mobile/android/scripts/prepare-guild-content.cjs` calls that canonical bundler in temporary staging, verifies the captured hashes and creates a complete signed content snapshot. It does not use website `public/` output and does not publish files.

Three scripts stay owned by the installed APK and are excluded from every downloadable content inventory:

- `wayfarers/native-checkpoint.js`: the native loader supplies the saved checkpoint.
- `wayfarers/checkpoint.js`: the installed checkpoint protocol and save acknowledgments.
- `wayfarers/android.js`: the installed Android control/bridge protocol.

`wayfarers/bundle-manifest.json` is build provenance, not a runtime content file, and is also excluded. `wayfarers/android.css` may be patched. The updated `index.html` can still reference the APK-owned scripts at their established paths; the native loader continues serving those from the installed package.

The complete snapshot makes each version independently usable offline and permits rollback without reconstructing a chain of binary deltas. A content ZIP contains only listed regular files; the loader rejects unexpected files, duplicate or ambiguous paths, protected bridge scripts, directories and invalid size/hash records. Files are limited to `8 MiB`, the complete ZIP and expanded snapshot to `32 MiB`, and the inventory to `512` files. Every inventory requires `wayfarers/index.html` and `wayfarers/game.css`.

## Signed protocol

The signed JSON envelope has exactly two properties:

```json
{"payload":"BASE64_OF_EXACT_UTF8_MANIFEST","signature":"BASE64_OF_DER_ECDSA_SIGNATURE"}
```

The long-lived key is EC P-256 (`prime256v1`). The signature is SHA-256 with ECDSA in ASN.1 DER format, compatible with Java `Signature.getInstance("SHA256withECDSA")`. The app embeds the public key as base64 X.509 SubjectPublicKeyInfo (SPKI), compatible with Java `X509EncodedKeySpec`. The signature covers the decoded payload bytes exactly; clients must not parse and reserialize before verification. The payload is limited to `256 KiB`; the complete envelope is limited to `512 KiB`.

The manifest payload is:

```json
{
  "schemaVersion": 1,
  "packageName": "me.danielshort.wayfarers",
  "contentVersion": 2,
  "label": "0.12.0.1",
  "nativeApi": 1,
  "minAppVersionCode": 16,
  "saveSchema": 7,
  "archive": {
    "url": "https://github.com/danielshort3/danielshort3.github.io/releases/download/wayfarers-content-v2/Wayfarers-content-v2-ARCHIVE_HASH_PREFIX.zip",
    "sha256": "EXACT_64_CHARACTER_LOWERCASE_SHA256",
    "size": 123456
  },
  "records": [
    {"path":"wayfarers/index.html","sha256":"EXACT_FILE_SHA256","size":1234},
    {"path":"wayfarers/game.css","sha256":"EXACT_FILE_SHA256","size":5678}
  ]
}
```

`contentVersion` is a strictly increasing integer independent of the APK version. `label` is a readable release label. This native API accepts save schema `7`; a schema migration or a new native bridge contract must first ship through a compatible APK. Do not relabel a save-schema-changing release as schema 7. The builder currently deliberately fixes native API `1`, minimum APK code `16` and save schema `7`.

Only HTTPS assets under this repository's GitHub release-download path are allowed. The fixed signed channel pointer is `https://github.com/danielshort3/danielshort3.github.io/releases/download/wayfarers-guild-updates/latest-content.json`. Android package updates continue using `latest.json` on the same channel release; keep these two independent manifests distinct.

## Prepare and publish

Keep the private signing key outside every checkout, cloud upload, release asset and Android source tree. Preserve it securely across releases; changing the embedded public key requires an APK update. A synthetic interoperability fixture under `mobile/android/scripts/guild-content-protocol-fixture.json` has its own discarded test key and must never be used as a production release identity.

From the repository root, prepare into a fresh directory outside the checkout:

```powershell
node mobile/android/scripts/prepare-guild-content.cjs `
  --key C:\Guild-Signing\content-p256.pem `
  --output C:\Guild-Releases\content-v2 `
  --base-url https://github.com/danielshort3/danielshort3.github.io/releases/download/wayfarers-content-v2/ `
  --version 2 `
  --label 0.12.0.1
```

For every subsequent content release, supply the prior published signed envelope:

```powershell
node mobile/android/scripts/prepare-guild-content.cjs `
  --key C:\Guild-Signing\content-p256.pem `
  --output C:\Guild-Releases\content-v3 `
  --base-url https://github.com/danielshort3/danielshort3.github.io/releases/download/wayfarers-content-v3/ `
  --version 3 `
  --label 0.12.0.2 `
  --previous-envelope C:\Guild-Releases\published-content-v2.json
```

The builder rejects a different signing curve, an in-repository private key, a non-increasing signed version, unsafe paths, changed canonical bytes and incompatible protocol fields. It generates a deterministic ZIP with stored entries and fixed timestamps using only Node's standard library. ECDSA signatures are randomized, so independently preparing the same release can produce a different envelope signature. Archive bytes remain deterministic. Retain the original staged release when retrying publication; do not regenerate and overwrite immutable envelopes.

Generated artifacts:

- `Wayfarers-content-v<VERSION>-<ARCHIVE_HASH_PREFIX>.zip`: complete immutable game snapshot.
- `content-v<VERSION>-<ARCHIVE_HASH_PREFIX>.json`: immutable signed envelope.
- `latest-content.json`: byte-for-byte copy of the same signed envelope for channel publication.
- `CONTENT-SHA256SUMS.txt`: exact checksums of the ZIP and both envelope files.

The builder checks every output destination before writing anything and refuses to replace any different existing artifact. It stages locally only. Publication is a separate authorized release action.

Publish the immutable ZIP, versioned signed envelope and checksums to the versioned GitHub release. Download the public bytes and verify signatures, exact sizes, archive SHA-256 and every extracted file record. Only after verification, replace `latest-content.json` on the fixed `wayfarers-guild-updates` release with the already verified signed bytes. Never replace immutable ZIPs/envelopes or the APK's `latest.json` while publishing a content update.

## Installed verification

Run the dependency-free release tests:

```powershell
node --test mobile/android/scripts/prepare-guild-content.test.cjs
node --test mobile/android/scripts/wayfarers-bundle.test.cjs
```

Java tests must verify the shared fixture's SPKI key, DER signature, exact decoded payload and ZIP entries. Native tests separately verify download bounds, signed inventory enforcement, protected-path rejection, truncated/tampered bytes, offline activation and recovery from interrupted activation.

On a disposable Android installation, preserve a guild checkpoint and settings, install the compatible native shell normally, then apply a publicly signed newer content version through the game's update controls. Verify the APK package/version is unchanged by the content update, the activity stays open, progress/settings remain intact, and the content persists through an offline cold launch. Exercise the previous-content rollback after a failed candidate boot. A locally generated archive or passing JVM suite alone does not prove a released in-app update works.

The application must finish the normal durable checkpoint before switching the WebView's active snapshot. Activation briefly pauses the game and reloads it inside the same native activity. An interrupted download leaves the running version intact. A failed candidate must fall back to its previously verified snapshot or packaged baseline. Normal game reset clears the guild; it must not silently clear content versions or app-update preferences.
