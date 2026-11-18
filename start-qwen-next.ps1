<#
.SYNOPSIS
  Start the qwen-next server with the accepted decode-first profile.

.DESCRIPTION
  Launches llama-server from the qwen-next engine build with the flags and
  QNEXT_* switches of the fastest measured profile (90k context, DFlash 2 drafter
  with 2 guesses, elastic memory lending, lookahead expert prefetch). On replayed
  real agent sessions this was the fastest option at every context size.
  -Mtp selects the earlier profile that drafts with the model's own MTP block.

  Paths come from parameters or from environment variables:
    QNEXT_BIN    folder that contains llama-server.exe
    QNEXT_MODEL  first shard of the target GGUF
    QNEXT_DFLASH DFlash 2 drafter GGUF (default profile; if missing, falls back to -Mtp)
    QNEXT_DRAFT  MTP draft GGUF (used with -Mtp; without it the no-draft profile is used)

.EXAMPLE
  .\start-qwen-next.ps1 -Model D:\models\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf `
                        -DFlash D:\models\Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf

.EXAMPLE
  .\start-qwen-next.ps1 -Mtp -Model D:\models\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf `
                        -Draft D:\models\mtp-Qwen3.8-Flash-Next-Q8_0.gguf

.EXAMPLE
  .\start-qwen-next.ps1 -NoDraft -Context 64000 -Port 8081
#>
[CmdletBinding()]
param(
    [string]$Bin     = $env:QNEXT_BIN,
    [string]$Model   = $env:QNEXT_MODEL,
    [string]$Draft   = $env:QNEXT_DRAFT,
    [string]$DFlash  = $env:QNEXT_DFLASH,
    [switch]$Mtp,        # draft with the model's MTP block instead of the DFlash 2 drafter
    [int]   $Context = 90000,
    [int]   $Port    = 8080,
    [string]$BindHost = '127.0.0.1',
    [int]   $DraftThreads = 8,
    [switch]$NoDraft,
    [switch]$Stock,      # stock flags of the seed (the measured floor), for comparison
    [switch]$DryRun,     # print the command and environment, do not start
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra     # anything else is passed to llama-server unchanged
)

$ErrorActionPreference = 'Stop'

if (-not $Bin) {
    # newest accepted local build if present, else the default build folder
    $root = Join-Path $PSScriptRoot 'engine'
    foreach ($c in 'build\bin', 'build-5080\bin') {
        $p = Join-Path $root $c
        if (Test-Path (Join-Path $p 'llama-server.exe')) { $Bin = $p; break }
    }
}
if (-not $Bin -or -not (Test-Path (Join-Path $Bin 'llama-server.exe'))) {
    throw "llama-server.exe not found. Build the engine (docs/BUILD.md) and pass -Bin or set QNEXT_BIN."
}
if (-not $Model -or -not (Test-Path $Model)) {
    throw "Target model not found. Pass -Model or set QNEXT_MODEL to the first GGUF shard."
}
$useDraft = -not $NoDraft -and -not $Stock
$useDFlash = $useDraft -and -not $Mtp
if ($useDFlash -and (-not $DFlash -or -not (Test-Path $DFlash))) {
    Write-Warning "No DFlash 2 drafter found (-DFlash / QNEXT_DFLASH): falling back to the MTP draft profile."
    $useDFlash = $false
}
if ($useDFlash) { $useDraft = $false }
if ($useDraft -and (-not $Draft -or -not (Test-Path $Draft))) {
    Write-Warning "No MTP draft model given (-Draft / QNEXT_DRAFT): starting the no-draft profile."
    $useDraft = $false
}

$srvArgs = @(
    '--model', $Model,
    '--host', $BindHost, '--port', $Port,
    '--parallel', '1',
    '--flash-attn', 'on',
    '--no-context-shift',
    '-c', $Context, '--fit', 'on', '--fit-ctx', $Context, '--fit-target', '512',
    '--batch-size', '4096',
    '--load-mode', 'none',
    '--jinja', '--reasoning', 'on',
    '--no-mmproj',
    '--metrics'
)

$envSet = [ordered]@{}
if ($Stock) {
    $srvArgs += @('--ubatch-size', '1024', '--spec-type', 'none')
} else {
    $srvArgs += @('--ubatch-size', '2048', '--moe-cache-mib', 'auto',
                  '--checkpoint-min-step', '1024', '--backend-sampling')
    $envSet['QNEXT_MOE_BIGBATCH_BYPASS'] = '1'   # large prefill batches bypass the expert cache
    $envSet['QNEXT_MOE_BATCH_COPY']      = '1'   # missed experts move in few large async copies
    $envSet['QNEXT_ELASTIC_KV']          = '3'   # unused KV reservation is lent to the expert cache
    $envSet['QNEXT_ELASTIC_COMPUTE']     = '1'   # compute-buffer reservation is lent during decode
    $envSet['QNEXT_PF_OVERLAP']          = '1'   # prefill: next layer's uploads overlap this layer's compute
    $envSet['QNEXT_PF_OVERLAP_SLOTS']    = '3'
    $envSet['QNEXT_IDS_ASYNC']           = '1'   # routing ids read back without a full sync
    if ($useDFlash) {
        # DFlash 2 block-diffusion drafter, 2 guesses per round; the small draft batch keeps its buffers at ~640 MiB
        $srvArgs += @('--model-draft', $DFlash, '--spec-type', 'draft-dflash',
                      '--spec-draft-n-max', '2', '--spec-draft-n-ubatch', '256',
                      '-td', $DraftThreads, '-tbd', $DraftThreads)
        $envSet['QNEXT_MOE_LOOKAHEAD']         = '4'
        $envSet['QNEXT_MOE_LOOKAHEAD_MAX_MIB'] = '24'
        $envSet['QNEXT_MOE_LOOKAHEAD_RANK']    = '1'
        $envSet['QNEXT_MOE_LOOKAHEAD_ISSUE']   = 'post'
        $envSet['QNEXT_MOE_LOOKAHEAD_P4_FAST'] = '1'
        $envSet['QNEXT_MOE_MISS_OVERLAP']      = '1'
    } elseif ($useDraft) {
        $srvArgs += @('--model-draft', $Draft, '--spec-type', 'draft-mtp',
                      '--spec-draft-n-max', '2', '--spec-draft-cpu-moe',
                      '--spec-draft-n-ubatch', '512',
                      '-td', $DraftThreads, '-tbd', $DraftThreads)
        $envSet['QNEXT_MTP_PRUNE'] = '1'         # draft pass computes only what it reads
        $envSet['QNEXT_MTP_CROP']  = '1'         # draft buffers cropped, memory goes to the cache
        # lookahead expert prefetch (wins only on multi-token verify rounds, i.e. with the draft)
        $envSet['QNEXT_MOE_LOOKAHEAD']         = '4'
        $envSet['QNEXT_MOE_LOOKAHEAD_MAX_MIB'] = '24'
        $envSet['QNEXT_MOE_LOOKAHEAD_RANK']    = '1'
        $envSet['QNEXT_MOE_LOOKAHEAD_ISSUE']   = 'post'
        $envSet['QNEXT_MOE_LOOKAHEAD_P4_FAST'] = '1'
        # missed experts upload while the rest of the previous layer's graph still runs
        $envSet['QNEXT_MOE_MISS_OVERLAP']      = '1'
    } else {
        $srvArgs += @('--spec-type', 'none')
    }
}
if ($Extra) { $srvArgs += $Extra }

$exe = Join-Path $Bin 'llama-server.exe'
$profileName = if ($Stock) { 'stock (floor)' } elseif ($useDFlash) { 'fastest, DFlash 2 drafter (2 guesses)' } elseif ($useDraft) { 'MTP draft' } else { 'accepted, no draft' }
Write-Host "qwen-next  profile: $profileName   context: $Context   http://${BindHost}:$Port"
foreach ($k in $envSet.Keys) { Write-Host ("  {0}={1}" -f $k, $envSet[$k]) }
Write-Host ("  " + $exe + " " + ($srvArgs -join ' '))
if ($DryRun) { return }

foreach ($k in $envSet.Keys) { Set-Item -Path ("Env:" + $k) -Value $envSet[$k] }
Push-Location $Bin
try { & $exe @srvArgs } finally { Pop-Location }
