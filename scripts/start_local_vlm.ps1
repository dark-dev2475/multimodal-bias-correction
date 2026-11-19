param()
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$localDir = Join-Path $taskRoot 'data/local_vlm'
$runtime = Join-Path $localDir 'runtime'
$serverExe = Get-ChildItem -LiteralPath $runtime -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
if (-not $serverExe) { throw 'Run python scripts/setup_local_vlm.py first.' }
if (Get-NetTCPConnection -LocalPort 8087 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 8087 is already in use. Inspect the existing server; do not launch a duplicate.'
}
$modelFile = Join-Path $localDir 'Qwen3VL-2B-Instruct-Q4_K_M.gguf'
$projectorFile = Join-Path $localDir 'mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf'
$serverArgs = @('-m', ('"' + $modelFile + '"'), '--mmproj', ('"' + $projectorFile + '"'),
    '--host', '127.0.0.1', '--port', '8087', '--alias', 'qwen3-vl-2b-local',
    '-c', '8192', '-ngl', 'all', '-ub', '128', '--parallel', '1',
    '--image-min-tokens', '1024', '--image-max-tokens', '1024', '--reasoning', 'off', '--cors-origins', 'localhost')
$config = [ordered]@{
    host = '127.0.0.1'; port = 8087; model_alias = 'qwen3-vl-2b-local';
    context_tokens = 8192; gpu_layers = 'all'; micro_batch = 128; parallel_slots = 1;
    image_min_tokens = 1024; image_max_tokens = 1024; reasoning = 'off'; cors_origins = 'localhost';
    model = 'Qwen3VL-2B-Instruct-Q4_K_M.gguf'; projector = 'mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf'
}
$config | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $localDir 'server_config.json') -Encoding ASCII
$process = Start-Process -FilePath $serverExe.FullName -ArgumentList $serverArgs -WorkingDirectory $runtime -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $localDir 'server.stdout.log') -RedirectStandardError (Join-Path $localDir 'server.stderr.log')
$process.Id | Set-Content -LiteralPath (Join-Path $localDir 'server.pid')
Write-Output "Started loopback-only local VLM (PID $($process.Id)). Logs: $localDir"
