# Git and portable path setup

This directory is an independent Git repository. It can be cloned anywhere;
tracked code must not depend on `D:\DayCon\...` or another machine-specific path.

## Path configuration

Tracked portable defaults live in `project.properties`, where `project.root=.`
means the directory containing that file. Python code can load paths with
`src/project_config.py`.

For data stored elsewhere, copy the example and edit only the untracked file:

```powershell
Copy-Item project.local.properties.example project.local.properties
```

Alternatively set `CRASHVIDEO_PROJECT_PROPERTIES` to an explicit properties file.
Never commit credentials, tokens, or machine-specific secrets.

## Intentionally untracked

- `data_raw/`: downloaded CCD/comma2k19/DoTA source data
- `Baseline.zip`: 195MB original archive (exceeds GitHub's normal file limit)
- `Baseline/submit.zip`: generated submission archive
- `.venv/`, IDE state, caches, logs, model checkpoints, and `artifacts/`

Small official baseline examples and notebooks remain tracked so a fresh clone can
run structural smoke tests. Manually authored labels may be tracked once created;
raw videos remain excluded.

## Publish

After creating an empty remote repository:

```powershell
git commit -m "Initial crashvideo project"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Review `git status` before every commit. Do not use `git add -f` for ignored data or
model files.
