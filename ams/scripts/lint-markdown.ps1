param($file)

if ($file) {
    # Lint and auto-fix the specific file
    markdownlint --fix $file
} else {
    # Lint and auto-fix all .md files in the workspace
    Get-ChildItem -Recurse -Filter *.md | ForEach-Object { markdownlint --fix $_.FullName }
}