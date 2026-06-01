# Publication Checklist

This copy was prepared for public GitHub publication from `D:\D\AStockAIAgent`.

Before pushing:

- Choose and add a real open-source license, for example `LICENSE` with MIT/Apache-2.0/GPL text.
- Review `README.md` and add a short English summary near the top.
- Confirm no private data is committed. This copy excludes `data/raw`, `data/processed`, `models`, `reports`, caches, and `*.csv` files.
- Run tests locally:

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest tests
```

Current prepared-copy verification on 2026-06-01:

```text
19 passed in 7.27s
```

- Create the GitHub repository as public under `kevinlu2002/AStockAIAgent`.
- Use the public repo description:

```text
A-share market research agent with feature engineering, model training, risk-aware recommendations, and a Flask dashboard.
```

For Codex for OSS, do not describe this as a high-impact open-source project until it has real public users, stars, issues, or contributors. Use it as evidence of public code quality and maintenance activity, then pair it with meaningful PRs to established projects.
