# Deployment Instructions

## Option 1: GitHub + Render (Recommended)

1. Install Git from https://git-scm.com/download/win
2. Create GitHub repository
3. Upload files to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
4. Connect GitHub repo to Render.com

## Option 2: Direct Upload to Render

1. Go to render.com and sign up
2. Click "New +" → "Web Service"
3. Choose "Deploy from Git repository"
4. If no Git, use "Deploy from GitHub" and upload files manually
5. Set build command: `pip install -r requirements.txt`
6. Set start command: `python app.py`
7. Deploy on free tier

## Option 3: Alternative Platforms

### Heroku
1. Install Heroku CLI
2. `heroku create your-app-name`
3. `git push heroku main`

### Railway
1. Go to railway.app
2. Connect GitHub or upload files
3. Auto-deploys from repository

## Files Ready for Deployment:
- app.py (Flask app with PORT config)
- requirements.txt (CPU PyTorch + dependencies)
- render.yaml (Render configuration)
- Procfile (Heroku/Railway support)
- runtime.txt (Python version)
- .gitignore (Clean repository)

## Important Notes:
- Model file (final_efficientnet_b0_cv.pth) must be uploaded
- Free tiers have memory/CPU limitations
- App uses CPU-only PyTorch for compatibility