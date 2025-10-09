# MkDocs Documentation Guide for Teams

This guide explains how to use MkDocs to create professional documentation for your hackathon submission.


**Why MkDocs**: Professional, navigable documentation that  can  be easily browsed during evaluation and presentations.


## Quick Setup

### 1. Install MkDocs
```bash
# Python
pip install mkdocs mkdocs-material

# Or add to requirements.txt
echo "mkdocs>=1.5.0" >> requirements.txt
echo "mkdocs-material>=9.0.0" >> requirements.txt
```

### 2. View existing mkdocs site
2.1 Run below command over the terminal
```bash
mkdocs serve
```
2.2 Open browser and visit: http://localhost:8000


### 2. Adding content
Your repo already has:
```
your-repo/
├── mkdocs.yml          # MkDocs configuration
└── docs/
    └── index.md        # Home page
```

- Here the index.md has sample content. You can edit or replace it.
- Create new md files as needed for the content.Sample reference as below:
  - What are the final equations? What are the packages/algorithms used? 
  - Have you leveraged GenAI across teh rpediction, optimization and webapp development?
  - What are the key hypothesis/assumptions taken from a business POV? 
  - If you had more time/data, what enhancements would you build? 

[Detailed documentation requirements](https://anheuserbuschinbev-my.sharepoint.com/:w:/r/personal/pearlnova_antony_ab-inbev_com/Documents/My%20Documents/instructions.docx?d=w9a258a87d64b4911bd76874b7c6057c0&csf=1&web=1&e=DhIJqF)
### 3.Update mkdocs.yml 

Once you have added the necessary files, update the `mkdocs.yml` to reflect your structure. 

```yaml

nav:
  - Home: index.md
  - Solution:
    - Problem & Approach: solution.md
    - Optimization Algorithm: algorithm.md
    - Results & Analysis: results.md
    - Webapp specifications: techstack.md
```

### 4. Build & Deploy

If you already have the mkdocs serve running, it will auto-refresh on file changes.
If not rerun the `mkdocs serve` command.

On the presentation day, you can present your submission directly from the local mkdocs server and showcase the webapp locally.