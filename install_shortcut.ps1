# Adds "Glide" to the Start menu (and optionally the desktop), using the Glide icon.
param([switch]$Desktop)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root ".venv\Scripts\pythonw.exe"
& (Join-Path $root ".venv\Scripts\python.exe") -c "from glide.desktop import ensure_icon; ensure_icon()"
$icon = Join-Path $root "models\glide.ico"
$shell = New-Object -ComObject WScript.Shell
$targets = @([Environment]::GetFolderPath("Programs"))
if ($Desktop) { $targets += [Environment]::GetFolderPath("Desktop") }
foreach ($dir in $targets) {
  $lnk = $shell.CreateShortcut((Join-Path $dir "Glide.lnk"))
  $lnk.TargetPath = $py
  $lnk.Arguments = "-m glide"
  $lnk.WorkingDirectory = $root
  $lnk.IconLocation = $icon
  $lnk.Description = "Glide bare-hand desk mouse"
  $lnk.Save()
  Write-Output "Created $(Join-Path $dir 'Glide.lnk')"
}
