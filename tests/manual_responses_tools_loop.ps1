param(
    [string]$AdapterUrl = $(if ($env:ADAPTER_URL) { $env:ADAPTER_URL } else { "http://localhost:8000" }),
    [string]$Model = "minimaxai/minimax-m3",
    [switch]$Stream
)

$ErrorActionPreference = "Stop"

$headers = @{ "Content-Type" = "application/json" }
if ($env:ADAPTER_API_KEY) {
    $headers["Authorization"] = "Bearer $($env:ADAPTER_API_KEY)"
}

$tool = @{
    type = "function"
    name = "get_test_value"
    description = "Retourne une valeur de test"
    parameters = @{
        type = "object"
        properties = @{ key = @{ type = "string" } }
        required = @("key")
        additionalProperties = $false
    }
    strict = $false
}

function Invoke-ResponsesJson {
    param([hashtable]$Body)
    $json = $Body | ConvertTo-Json -Depth 50 -Compress
    Invoke-RestMethod -Method Post -Uri "$AdapterUrl/v1/responses" -Headers $headers -Body $json
}

function ConvertFrom-Sse {
    param([string]$Raw)
    $events = @()
    foreach ($block in ($Raw -split "(`r?`n){2,}")) {
        if (-not $block.Trim()) { continue }
        $eventName = $null
        $data = $null
        foreach ($line in ($block -split "`r?`n")) {
            if ($line.StartsWith("event: ")) { $eventName = $line.Substring(7) }
            if ($line.StartsWith("data: ")) { $data = $line.Substring(6) | ConvertFrom-Json }
        }
        if ($eventName) { $events += [pscustomobject]@{ event = $eventName; data = $data } }
    }
    $events
}

$userText = 'Appelle obligatoirement get_test_value avec key="demo".'

if (-not $Stream) {
    $first = Invoke-ResponsesJson @{
        model = $Model
        stream = $false
        input = $userText
        tools = @($tool)
        tool_choice = "required"
        max_output_tokens = 128
    }

    $call = @($first.output | Where-Object { $_.type -eq "function_call" })[0]
    if (-not $call) { throw "Aucun function_call recu" }

    Write-Host "function_call:"
    $call | ConvertTo-Json -Depth 20

    $resultatOutil = "VALUE_42"
    $second = Invoke-ResponsesJson @{
        model = $Model
        stream = $false
        input = @(
            @{ role = "user"; content = $userText },
            @{ type = "function_call"; call_id = $call.call_id; name = $call.name; arguments = $call.arguments },
            @{ type = "function_call_output"; call_id = $call.call_id; output = $resultatOutil },
            @{ role = "user"; content = "Reponds uniquement avec la valeur obtenue." }
        )
        max_output_tokens = 128
    }

    Write-Host "final output_text:"
    Write-Host $second.output_text
    return
}

$streamBody = @{
    model = $Model
    stream = $true
    input = $userText
    tools = @($tool)
    tool_choice = "required"
    max_output_tokens = 128
} | ConvertTo-Json -Depth 50 -Compress

$raw = Invoke-WebRequest -Method Post -Uri "$AdapterUrl/v1/responses" -Headers $headers -Body $streamBody
$events = ConvertFrom-Sse ([string]$raw.Content)
$interesting = $events | Where-Object {
    $_.event -in @(
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed"
    )
}
$interesting | ForEach-Object {
    "{0} {1}" -f $_.event, ($_.data | ConvertTo-Json -Depth 20 -Compress)
}

