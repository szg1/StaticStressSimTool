#!/bin/bash
set -e # Stop script on any error

# 0. Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI ('gh') is not installed."
    exit 1
fi

cd ..

BRANCH_NAME="$1"

# 1. Update Main first (so we merge into fresh code)
echo "Updating main..."
git switch main
git pull -X theirs

# 2. Detect if a PR exists for this branch targeting main
# We look for a PR where 'head' is your branch and 'base' is main.
echo "Checking for open PRs from '$BRANCH_NAME'..."
PR_NUMBER=$(gh pr list --head "$BRANCH_NAME" --base main --state open --json number --jq '.[0].number')

if [ -n "$PR_NUMBER" ]; then
    # --- SCENARIO A: PR EXISTS ---
    echo "✔ Found PR #$PR_NUMBER! Handling as a Pull Request."
    
    # 'gh pr checkout' is smarter than git switch. 
    # It handles permissions and remotes automatically if the PR is from a fork.
    gh pr checkout "$PR_NUMBER"
    
    # Update the PR code, accepting 'theirs' (incoming from remote) if conflict
    git pull -X theirs
    
    echo "Merging PR #$PR_NUMBER into main..."
else
    # --- SCENARIO B: NO PR (Standard Branch) ---
    echo "✘ No open PR found. Handling as a local branch merge."
    
    git switch "$BRANCH_NAME"
    git pull -X theirs
    
    echo "Merging branch '$BRANCH_NAME' into main..."
fi

# 3. The Merge (Common Logic)
# We switch back to main and merge the feature branch/PR we just updated.
git switch main
git merge -X theirs --no-gpg-sign --no-edit "$BRANCH_NAME"

# 4. Build, Test, and Push
cd experimental_2

if make; then
    echo "Build successful! Pushing to main..."
    # Pushing to main will automatically mark the PR as 'Merged' on GitHub
    cd ..
    git push origin main
    
    cd experimental_2
    # ./stl_viewer 1hole.stl DIN912M10L80.stl 0.4 0.28 5 25 &
    # ./stl_viewer 2holes.stl DIN912M10L80.stl 0.4 0.28 5 25 &
    # ./stl_viewer 2holes_weird.stl DIN912M10L80.stl 0.4 0.28 5 25 &
    ./stl_viewer 1hole_countersunk.stl M10_Countersunk.stl 0.4 0.28 5 25 &
else
    echo "Build FAILED. Reverting merge and exiting."
    # Optional: Reset main back to origin to undo the local merge since build failed
    cd ..
    git reset --hard origin/main
    exit 1
fi