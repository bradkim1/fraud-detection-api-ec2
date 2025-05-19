#!/bin/bash
# Script to initialize git, add all files (excluding large models), commit, and push to GitHub

# Navigate to your project directory
cd ~/model-ec2

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
  echo "Initializing git repository..."
  git init
fi

# Create a .gitignore file to exclude certain files
cat > .gitignore << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Logs
logs/
*.log

# Build artifacts
docs/build/

# Virtual Environment
venv/
ENV/

# IDE files
.idea/
.vscode/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Large model files
*.pkl
model/*.pkl
EOL

# Add a README note about model files
echo -e "\n## Note about model files\n\nThe model files (*.pkl) are not included in this repository due to their size. Please see the instructions in the documentation for generating or downloading these files." >> README.md

# Add all files (excluding those in .gitignore)
echo "Adding files to git..."
git add .

# Check for any large files that might have been missed by .gitignore
large_files=$(find . -type f -size +50M -not -path "./.git/*" | wc -l)
if [ "$large_files" -gt 0 ]; then
  echo "WARNING: Found $large_files files larger than 50MB that may cause issues when pushing to GitHub!"
  echo "Large files found:"
  find . -type f -size +50M -not -path "./.git/*"
  read -p "Continue anyway? (y/n): " continue_anyway
  if [ "$continue_anyway" != "y" ]; then
    echo "Aborting. Please exclude these files before continuing."
    exit 1
  fi
fi

# Configure git (only if not already configured)
if [ -z "$(git config --get user.name)" ]; then
  echo "Configuring git user name and email..."
  read -p "Enter your git username: " username
  read -p "Enter your git email: " email
  git config user.name "$username"
  git config user.email "$email"
fi

# Commit
echo "Committing changes..."
git commit -m "Initial commit of fraud detection API (excluding model files)"

# Add GitHub remote (if not exists)
if ! git remote | grep -q "origin"; then
  echo "Adding GitHub remote..."
  git remote add origin https://github.com/bradkim1/fraud-detection-api-ec2.git
fi

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin master

echo "Done! Your code is now on GitHub (excluding large model files)."
echo "Reminder: Users will need to generate or download model files separately."
