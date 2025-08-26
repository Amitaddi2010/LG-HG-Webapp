# Deployment Instructions

## Git Setup
Run these commands in your terminal:

```bash
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"
```

## Deploy to Render

1. **Initialize Git Repository:**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
```

2. **Push to GitHub:**
```bash
git remote add origin https://github.com/Amitaddi2010/LG-HG-Webapp.git
git push -u origin main
```

3. **Deploy on Render:**
- Go to render.com
- Connect GitHub repository
- Choose "Web Service"
- Auto-detects render.yaml config

## Files Ready for Deployment:
- ✅ requirements.txt
- ✅ render.yaml
- ✅ .gitignore
- ✅ README.md
- ✅ app.py (production ready)

Your app will be live at: `https://your-app-name.onrender.com`