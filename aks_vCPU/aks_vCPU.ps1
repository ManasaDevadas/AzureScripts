Connect-AzAccount

# Cache SKUs by location to avoid repeated calls
$skuCache = @{}

$results = foreach ($sub in Get-AzSubscription) {
    Set-AzContext -SubscriptionId $sub.Id | Out-Null
    Write-Host "Scanning subscription: $($sub.Name)" -ForegroundColor Cyan
    
    # Build AKS cluster lookup from node resource groups (using Azure CLI)
    $aksLookup = @{}
    $aksClusters = az aks list --subscription $sub.Id --query "[].{name:name, nodeRg:nodeResourceGroup}" -o json 2>$null | ConvertFrom-Json
    foreach ($aks in $aksClusters) {
        $aksLookup[$aks.nodeRg] = $aks.name
    }
    
    foreach ($ss in Get-AzVmss) {
        $size = $ss.Sku.Name
        $capacity = $ss.Sku.Capacity
        $location = $ss.Location
        $rg = $ss.ResourceGroupName
        
        # Check if this VMSS belongs to an AKS cluster
        $clusterName = if ($aksLookup.ContainsKey($rg)) { $aksLookup[$rg] } else { "-" }
        
        # Cache SKUs per location
        if (-not $skuCache.ContainsKey($location)) {
            $skuCache[$location] = Get-AzComputeResourceSku -Location $location | 
                Where-Object { $_.ResourceType -eq 'virtualMachines' }
        }
        
        $sku = $skuCache[$location] | Where-Object { $_.Name -eq $size }
        $vcpuPerNode = ($sku.Capabilities | Where-Object { $_.Name -eq 'vCPUs' }).Value
        
        [PSCustomObject]@{
            Subscription = $sub.Name
            AKSCluster = $clusterName
            NodePool = $ss.Name
            Location = $location
            VMSize = $size
            Instances = $capacity
            vCPUsPerNode = [int]$vcpuPerNode
            TotalvCPUs = [int]$vcpuPerNode * $capacity
        }
    }
}

$results | Format-Table -AutoSize
$csvPath = ".\AKS-vCPU-Report-$(Get-Date -Format 'yyyy-MM-dd_HHmmss').csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation
Write-Host "`nCSV exported to: $csvPath" -ForegroundColor Green
