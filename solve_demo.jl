# Runs only the β=0.2 demo solve and patches oct_idk_results.json
# Much faster than re-running the full sweep (~5 min instead of ~40 min)

import Pkg
let installed = keys(Pkg.project().dependencies)
    for pkg in ["JuMP", "HiGHS", "CSV", "DataFrames", "Statistics", "JSON"]
        pkg ∉ installed && Pkg.add(pkg)
    end
end
using JuMP, HiGHS, CSV, DataFrames, Statistics, JSON, LinearAlgebra, Printf, Random

const BETA_DEMO  = 0.2
const N_SUB      = 150
const D_TREE     = 3
const P_FEAT     = 5
const ALPHA      = 0.0001
const MIP_GAP    = 0.03
const TIME_LIMIT = 600.0

file_path = let candidates = [joinpath(@__DIR__, "breast_cancer.csv"),
                              joinpath(@__DIR__, "..", "breast_cancer.csv")]
    f = findfirst(isfile, candidates)
    f === nothing ? error("breast_cancer.csv not found") : candidates[f]
end
data   = CSV.read(file_path, DataFrame, skipto=2, header=false)
X_full = Matrix{Float64}(data[:, 1:30])
y_full = Int.(Vector(data[:, 31]))
N_full = length(y_full)

all_feature_names = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave_pts_mean","symmetry_mean","fractal_dim_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave_pts_se","symmetry_se","fractal_dim_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave_pts_worst","symmetry_worst","fractal_dim_worst"
]

Random.seed!(42)
idx0    = shuffle(findall(y_full .== 0))
idx1    = shuffle(findall(y_full .== 1))
k0      = round(Int, N_SUB * sum(y_full .== 0) / N_full)
k1      = N_SUB - k0
sub_idx = vcat(idx0[1:k0], idx1[1:k1])
X_all   = X_full[sub_idx, :]
y_raw   = y_full[sub_idx]
n       = length(y_raw)

mu0      = mean(X_all[y_raw .== 0, :], dims=1)[:]
mu1      = mean(X_all[y_raw .== 1, :], dims=1)[:]
sigma    = std(X_all, dims=1)[:]
scores   = abs.(mu1 .- mu0) ./ (sigma .+ 1e-8)
feat_idx = sortperm(scores, rev=true)[1:P_FEAT]
X_raw    = X_all[:, feat_idx]

X_min = minimum(X_raw, dims=1)
X_max = maximum(X_raw, dims=1)
rng_f = X_max .- X_min
rng_f[rng_f .== 0] .= 1.0
X     = (X_raw .- X_min) ./ rng_f

n, p     = size(X)
classes  = [0, 1]; K = 2
Y        = [y_raw[i] == classes[k] ? 1 : -1 for i in 1:n, k in 1:K]
sel_names = all_feature_names[feat_idx]

eps_j = zeros(p)
for j in 1:p
    sv = sort(unique(X[:, j]))
    eps_j[j] = length(sv) > 1 ? minimum(diff(sv)) : 1e-4
end
eps_max = maximum(eps_j)

D       = D_TREE
T_total = 2^(D+1) - 1
T_B     = 1:(T_total ÷ 2)
T_L     = (T_total ÷ 2 + 1):T_total

function get_ancestors(t)
    AL, AR = Int[], Int[]
    curr = t
    while curr > 1
        par = div(curr, 2)
        iseven(curr) ? push!(AL, par) : push!(AR, par)
        curr = par
    end
    return AL, AR
end

println("Solving OCT-IDK for β=$BETA_DEMO (demo tree only)…")

beta  = BETA_DEMO
M_big = n
model = Model(HiGHS.Optimizer)
set_silent(model)
set_optimizer_attribute(model, "output_flag",    false)
set_optimizer_attribute(model, "log_to_console", false)
set_optimizer_attribute(model, "mip_rel_gap",    MIP_GAP)
set_optimizer_attribute(model, "time_limit",     TIME_LIMIT)

@variable(model, a[1:p, T_B], Bin)
@variable(model, 0 <= b[T_B] <= 1)
@variable(model, d[T_B], Bin)
@variable(model, z[1:n, T_L], Bin)
@variable(model, l[T_L], Bin)
@variable(model, c[1:K, T_L], Bin)
@variable(model, L_err[T_L] >= 0)
@variable(model, idk[T_L], Bin)
@variable(model, q[T_L] >= 0)

@objective(model, Min,
    (1/n) * sum(L_err[t] + beta * q[t] for t in T_L) +
    ALPHA  * sum(d[t] for t in T_B))

for t in T_B
    @constraint(model, sum(a[j,t] for j in 1:p) == d[t])
    @constraint(model, b[t] <= d[t])
    t > 1 && @constraint(model, d[t] <= d[div(t,2)])
end
@constraint(model, d[1] == 1)
@constraint(model, [i=1:n], sum(z[i,t] for t in T_L) == 1)

for t in T_L
    @constraint(model, [i=1:n], z[i,t] <= l[t])
    @constraint(model, sum(z[i,t] for i in 1:n) >= 5 * l[t])
    @constraint(model, sum(c[k,t] for k in 1:K) + idk[t] == l[t])
    _, AR = get_ancestors(t)
    for m in AR
        @constraint(model, [i=1:n], z[i,t] <= d[m])
    end
end

for t in T_L
    AL, AR = get_ancestors(t)
    for m in AL, i in 1:n
        @constraint(model, sum(a[j,m]*(X[i,j]+eps_j[j]) for j in 1:p) <=
                    b[m] + (1+eps_max)*(1-z[i,t]))
    end
    for m in AR, i in 1:n
        @constraint(model, sum(a[j,m]*X[i,j] for j in 1:p) >= b[m] - (1-z[i,t]))
    end
end

for t in T_L
    N_t = @expression(model, sum(z[i,t] for i in 1:n))
    for k in 1:K
        N_kt = @expression(model, sum(0.5*(1+Y[i,k])*z[i,t] for i in 1:n))
        @constraint(model, L_err[t] >= N_t - N_kt - M_big*(1-c[k,t]))
        @constraint(model, L_err[t] <= N_t - N_kt + M_big*(1-c[k,t]))
    end
end

for t in T_L
    N_t = @expression(model, sum(z[i,t] for i in 1:n))
    @constraint(model, q[t] <= n * idk[t])
    @constraint(model, q[t] <= N_t)
    @constraint(model, q[t] >= N_t - n*(1-idk[t]))
end

t0 = time()
optimize!(model)
solve_time = round(time() - t0, digits=1)

has_sol = primal_status(model) in [MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT]
!has_sol && error("No feasible solution found for β=$BETA_DEMO")

println(@sprintf("  Done in %.0f s", solve_time))

leaf_info = Dict{Int,NamedTuple}()
for t in T_L
    is_active = value(l[t]) > 0.5
    is_idk    = value(idk[t]) > 0.5
    k_idx     = findfirst(k -> value(c[k,t]) > 0.5, 1:K)
    pred      = is_idk ? "IDK" : (k_idx !== nothing ? (classes[k_idx]==0 ? "Malignant" : "Benign") : "none")
    cnt       = round(Int, sum(value(z[i,t]) for i in 1:n))
    leaf_info[t] = (active=is_active, idk=is_idk, pred=pred, count=cnt)
end

preds = fill(-1, n)
for i in 1:n
    for t in T_L
        if value(z[i,t]) > 0.5
            info = leaf_info[t]
            preds[i] = info.idk ? -2 : (info.pred=="Malignant" ? 0 : 1)
            break
        end
    end
end

n_idk     = sum(preds .== -2)
n_decided = n - n_idk
dm        = preds .!= -2
total_err = round(Int, sum(value.(L_err)))
acc_dec   = n_decided > 0 ? 100.0*(n_decided-total_err)/n_decided : 0.0
acc_all   = 100.0*(n - total_err - n_idk)/n
coverage  = 100.0*n_decided/n

TP = sum((preds.==0) .& (y_raw.==0) .& dm)
TN = sum((preds.==1) .& (y_raw.==1) .& dm)
FP = sum((preds.==0) .& (y_raw.==1) .& dm)
FN = sum((preds.==1) .& (y_raw.==0) .& dm)

splits = Dict{Int,NamedTuple}()
for t in T_B
    if value(d[t]) > 0.5
        fj = findfirst(j -> value(a[j,t]) > 0.5, 1:p)
        if fj !== nothing
            thr = value(b[t])*(X_max[fj]-X_min[fj]) + X_min[fj]
            splits[t] = (feat=sel_names[fj], thr=thr)
        end
    end
end

println(@sprintf("  Coverage: %.1f%%  IDK: %d  Acc*: %.2f%%  Acc†: %.2f%%",
    coverage, n_idk, acc_dec, acc_all))

# Patch the existing JSON
json_path = joinpath(@__DIR__, "oct_idk_results.json")
existing  = JSON.parsefile(json_path)

existing["demo_beta"]    = BETA_DEMO
existing["demo_splits"]  = Dict(string(t) => Dict("feat"=>s.feat,"thr"=>s.thr) for (t,s) in splits)
existing["demo_leaves"]  = Dict(string(t) => Dict("pred"=>leaf_info[t].pred,
                                                    "idk"=>leaf_info[t].idk,
                                                    "count"=>leaf_info[t].count,
                                                    "active"=>leaf_info[t].active) for t in T_L)
existing["demo_metrics"] = Dict("coverage"=>coverage,"acc_decided"=>acc_dec,
                                "acc_overall"=>acc_all,"n_idk"=>n_idk,
                                "TP"=>TP,"TN"=>TN,"FP"=>FP,"FN"=>FN)

open(json_path, "w") do f
    JSON.print(f, existing, 2)
end
println("  Patched oct_idk_results.json with β=0.2 tree")
println("  Run: python3 visualize_oct_idk.py")
