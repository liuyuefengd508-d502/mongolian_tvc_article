# Zenodo DOI Guide for This Repository

Current Zenodo identifiers for this repository:

- Concept DOI: `10.5281/zenodo.20176970`
- Version DOI for `v1.0.0`: `10.5281/zenodo.20176971`

This guide explains how to create the Zenodo DOI for:

`https://github.com/liuyuefengd508-d502/mongolian_tvc_article`

## Before you start

Make sure the repository already contains:

- `CITATION.cff`
- `.zenodo.json`
- `README.md`
- the manuscript and public reproducibility materials you want to archive

This repository already contains those files.

## Step 1: Sign in to Zenodo

1. Go to `https://zenodo.org/`.
2. Sign in.
3. If possible, sign in with GitHub so Zenodo can access your repositories more easily.

## Step 2: Enable the GitHub repository in Zenodo

1. Open the Zenodo GitHub settings page:
   `https://zenodo.org/account/settings/github/`
2. Click `Sync now`.
3. Find the repository:
   `liuyuefengd508-d502/mongolian_tvc_article`
4. Turn the repository switch on.

If the repository does not appear:

- confirm that you are logged into the correct GitHub account
- confirm that the repository is visible to that account
- try `Sync now` again

## Step 3: Check metadata files

Zenodo can read both:

- `CITATION.cff`
- `.zenodo.json`

When `.zenodo.json` exists, Zenodo uses it as the main archive metadata. Before release, verify:

- title
- creators
- contributors
- keywords
- license
- description

If you later obtain the paper DOI, add it to `.zenodo.json` using `related_identifiers`.

## Step 4: Create the first GitHub release

Zenodo archives releases, not ordinary pushes.

1. Open the GitHub repository page.
2. Go to `Releases`.
3. Click `Draft a new release`.
4. Create a new tag:
   `v1.0.0`
5. Release title:
   `Initial public release for TVC resubmission`
6. Paste the contents of `RELEASE_NOTES_v1.0.0.md` into the release description.
7. Publish the release.

## Step 5: Wait for Zenodo archiving

After the release is published:

1. Return to Zenodo.
2. Open your uploads or GitHub-linked records.
3. Wait for Zenodo to create the software record for `v1.0.0`.

Zenodo will generate:

- a `Version DOI` for `v1.0.0`
- a `Concept DOI` for the repository as a whole

## Step 6: Which DOI to use in the paper

Use both if possible:

- `Concept DOI` as the long-term project DOI
- `Version DOI` when you want to identify the exact reproducibility snapshot

If you can only include one DOI in the manuscript, prefer the `Concept DOI` for the repository landing page and optionally mention the released version in text.

## Step 7: Update the manuscript files

After Zenodo generates the DOI:

1. Update:
   - `journal_submission/The_Visual_Computer/mongolian_tvc_article.tex`
   - `journal_submission/The_Visual_Computer/point_by_point_response_draft.md`
2. Replace the temporary Zenodo wording with the actual DOI.
3. Recompile the manuscript PDF.

## Recommended wording

For the repository homepage and release notes, keep language similar to:

`This repository is directly related to the manuscript submitted to The Visual Computer. If you use the code, protocol, or released artifacts, please cite both the manuscript and the Zenodo software record.`

## Troubleshooting

If Zenodo does not archive the release:

1. confirm the repository is enabled in Zenodo
2. confirm you created a GitHub Release, not just a tag
3. wait a few minutes and refresh Zenodo
4. if needed, disable and re-enable the repository in Zenodo, then create a new release

## Official references

- `https://help.zenodo.org/docs/github/enable-repository/`
- `https://help.zenodo.org/docs/github/archive-software/github-upload/`
- `https://help.zenodo.org/docs/github/describe-software/citation-file/`
- `https://help.zenodo.org/docs/github/describe-software/zenodo-json/`
