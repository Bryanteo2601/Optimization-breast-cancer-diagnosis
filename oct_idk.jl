# ============================================================
#   OCT with IDK (Abstain) Leaves — Solution 2
#   Extension of Bertsimas & Dunn (2017) OCT formulation
#
#   Each leaf now has THREE options:
#     1. Predict Malignant (0)
#     2. Predict Benign    (1)
#     3. IDK / Abstain     — refer for further testing
#
#   IDK is "free" in terms of misclassification, but incurs a
#   penalty β per patient routed to that leaf.  Without β the
#   solver would always abstain; β forces it to commit whenever
#   confident enough.
#
#   Run:  julia oct_idk.jl
# ============================================================

import Pkg
let installed = keys(Pkg.project().dependencies)
    for pkg in ["JuMP", "HiGHS", "CSV", "DataFrames", "Statistics", "JSON"]
        pkg ∉ installed && Pkg.add(pkg)
    end
end
using JuMP, HiGHS, CSV, DataFrames, Statistics, JSON, LinearAlgebra, Printf, Random

# ── Parameters ───────────────────────────────────────────────
const BETA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # IDK penalty sweep
const BETA_DEMO   = 0.2   # β used for the detailed tree printout — has IDK leaves
const N_SUB       = 150
const D_TREE      = 3
const P_FEAT      = 5
const ALPHA       = 0.0001
const MIP_GAP     = 0.03   # 3% gap — D=3 is larger, keep solve time reasonable
const TIME_LIMIT  = 300.0

# ─────────────────────────────────────────────────────────────
# SECTION 1: Data loading (identical to original notebook)
# ─────────────────────────────────────────────────────────────
file_path = let candidates = [joinpath(@__DIR__, "breast_cancer.csv"),
                              joinpath(@__DIR__, "..", "breast_cancer.csv")]
    f = findfirst(isfile, candidates)
    f === nothing ? error("breast_cancer.csv not found") : candidates[f]
end
data      = CSV.read(file_path, DataFrame, skipto=2, header=false)
X_full    = Matrix{Float64}(data[:, 1:30])
y_full    = Int.(Vector(data[:, 31]))
N_full    = length(y_full)

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
classes  = [0, 1];  K = 2
Y        = [y_raw[i] == classes[k] ? 1 : -1 for i in 1:n, k in 1:K]
sel_names = all_feature_names[feat_idx]

eps_j = zeros(p)
for j in 1:p
    sv = sort(unique(X[:, j]))
    eps_j[j] = length(sv) > 1 ? minimum(diff(sv)) : 1e-4
end
eps_max = maximum(eps_j)

# ─────────────────────────────────────────────────────────────
# SECTION 2: Tree structure (identical to original)
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# SECTION 3: Build OCT-IDK model for a given β
# ─────────────────────────────────────────────────────────────
function build_and_solve(beta::Float64)
    M_big = n
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "output_flag",    false)
    set_optimizer_attribute(model, "log_to_console", false)
    set_optimizer_attribute(model, "mip_rel_gap",    MIP_GAP)
    set_optimizer_attribute(model, "time_limit",     TIME_LIMIT)

    # ── Original variables ────────────────────────────────────
    @variable(model, a[1:p, T_B], Bin)
    @variable(model, 0 <= b[T_B] <= 1)
    @variable(model, d[T_B], Bin)
    @variable(model, z[1:n, T_L], Bin)
    @variable(model, l[T_L], Bin)
    @variable(model, c[1:K, T_L], Bin)
    @variable(model, L_err[T_L] >= 0)

    # ── NEW: IDK variables ────────────────────────────────────
    # idk[t] = 1 means leaf t abstains (outputs IDK)
    @variable(model, idk[T_L], Bin)

    # q[t]: number of patients at IDK leaf t (linearised product)
    # q[t] = (Σ_i z[i,t]) * idk[t]
    @variable(model, q[T_L] >= 0)

    # ── Objective: errors + β*IDK patients + complexity ───────
    # β per patient routed to an IDK leaf (normalised by n like errors)
    @objective(model, Min,
        (1/n) * sum(L_err[t] + beta * q[t] for t in T_L) +
        ALPHA  * sum(d[t] for t in T_B))

    # ── Constraint A: splitting structure (unchanged) ─────────
    for t in T_B
        @constraint(model, sum(a[j,t] for j in 1:p) == d[t])
        @constraint(model, b[t] <= d[t])
        t > 1 && @constraint(model, d[t] <= d[div(t,2)])
    end
    @constraint(model, d[1] == 1)

    # ── Constraint B: sample assignment (unchanged) ───────────
    @constraint(model, [i=1:n], sum(z[i,t] for t in T_L) == 1)

    for t in T_L
        @constraint(model, [i=1:n], z[i,t] <= l[t])
        @constraint(model, sum(z[i,t] for i in 1:n) >= 5 * l[t])

        # MODIFIED: active leaf predicts one class OR abstains (IDK)
        # Original was:  Σ_k c[k,t]         == l[t]
        # Solution 2:    Σ_k c[k,t] + idk[t] == l[t]
        @constraint(model, sum(c[k,t] for k in 1:K) + idk[t] == l[t])

        _, AR = get_ancestors(t)
        for m in AR
            @constraint(model, [i=1:n], z[i,t] <= d[m])
        end
    end

    # ── Constraint C: routing (unchanged) ────────────────────
    for t in T_L
        AL, AR = get_ancestors(t)
        for m in AL, i in 1:n
            @constraint(model,
                sum(a[j,m]*(X[i,j]+eps_j[j]) for j in 1:p) <=
                b[m] + (1+eps_max)*(1-z[i,t]))
        end
        for m in AR, i in 1:n
            @constraint(model,
                sum(a[j,m]*X[i,j] for j in 1:p) >= b[m] - (1-z[i,t]))
        end
    end

    # ── Constraint D: error counting (unchanged) ──────────────
    # When idk[t]=1: all c[k,t]=0, so D-constraints become trivially
    # inactive (big-M dominates). The objective then drives L_err[t]=0.
    for t in T_L
        N_t = @expression(model, sum(z[i,t] for i in 1:n))
        for k in 1:K
            N_kt = @expression(model, sum(0.5*(1+Y[i,k])*z[i,t] for i in 1:n))
            @constraint(model, L_err[t] >= N_t - N_kt - M_big*(1-c[k,t]))
            @constraint(model, L_err[t] <= N_t - N_kt + M_big*(1-c[k,t]))
        end
    end

    # ── Constraint E: linearise q[t] = N_t * idk[t] ──────────
    # McCormick envelope for the product of a continuous (N_t ∈ [0,n])
    # and a binary variable (idk[t] ∈ {0,1}).
    for t in T_L
        N_t = @expression(model, sum(z[i,t] for i in 1:n))
        @constraint(model, q[t] <= n * idk[t])          # q=0 when idk=0
        @constraint(model, q[t] <= N_t)                  # q ≤ actual count
        @constraint(model, q[t] >= N_t - n*(1-idk[t]))  # q=N_t when idk=1
    end

    # ── Solve ─────────────────────────────────────────────────
    t0 = time()
    optimize!(model)
    solve_time = round(time() - t0, digits=2)

    status  = termination_status(model)
    has_sol = primal_status(model) in [MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT]
    !has_sol && return nothing

    # ── Extract leaf decisions ────────────────────────────────
    leaf_info = Dict{Int, NamedTuple}()
    for t in T_L
        is_active = value(l[t]) > 0.5
        is_idk    = value(idk[t]) > 0.5
        k_idx     = findfirst(k -> value(c[k,t]) > 0.5, 1:K)
        pred_class = is_idk ? "IDK" : (k_idx !== nothing ? (classes[k_idx]==0 ? "Malignant" : "Benign") : "none")
        cnt        = round(Int, sum(value(z[i,t]) for i in 1:n))
        leaf_info[t] = (active=is_active, idk=is_idk, pred=pred_class, count=cnt)
    end

    # ── Performance metrics ───────────────────────────────────
    preds = fill(-1, n)
    for i in 1:n
        for t in T_L
            if value(z[i,t]) > 0.5
                info = leaf_info[t]
                preds[i] = info.idk ? -2 : (info.pred == "Malignant" ? 0 : 1)  # -2 = IDK
                break
            end
        end
    end

    n_idk    = sum(preds .== -2)
    n_decided = n - n_idk
    decided_mask = preds .!= -2

    total_err = round(Int, sum(value.(L_err)))
    acc_decided = n_decided > 0 ? 100.0 * (n_decided - total_err) / n_decided : 0.0
    acc_overall = 100.0 * (n - total_err - n_idk) / n   # IDK not counted as correct
    coverage    = 100.0 * n_decided / n

    TP = sum((preds .== 0) .& (y_raw .== 0) .& decided_mask)
    TN = sum((preds .== 1) .& (y_raw .== 1) .& decided_mask)
    FP = sum((preds .== 0) .& (y_raw .== 1) .& decided_mask)
    FN = sum((preds .== 1) .& (y_raw .== 0) .& decided_mask)

    # Branch node splits
    splits = Dict{Int, NamedTuple}()
    for t in T_B
        if value(d[t]) > 0.5
            fj = findfirst(j -> value(a[j,t]) > 0.5, 1:p)
            if fj !== nothing
                thr = value(b[t]) * (X_max[fj] - X_min[fj]) + X_min[fj]
                splits[t] = (feat=sel_names[fj], thr=thr, feat_idx=fj)
            end
        end
    end

    return (
        status=status, solve_time=solve_time,
        obj=objective_value(model), gap=100*relative_gap(model),
        leaf_info=leaf_info, splits=splits, preds=preds,
        n_idk=n_idk, coverage=coverage,
        acc_decided=acc_decided, acc_overall=acc_overall,
        TP=TP, TN=TN, FP=FP, FN=FN, total_err=total_err
    )
end

# ─────────────────────────────────────────────────────────────
# SECTION 4: Run demo (single β) + sensitivity sweep
# ─────────────────────────────────────────────────────────────
println("="^62)
println("  OCT-IDK  |  Breast Cancer  |  D=$D  p=$P_FEAT  n=$N_SUB")
println("  Solution 2: Abstain (IDK) option at each leaf")
println("="^62)

# Detailed run for the demo β
println("\n── Running OCT-IDK with β=$BETA_DEMO ──")
res = build_and_solve(BETA_DEMO)

if res !== nothing
    println(@sprintf("\n  Solver : %s  (%.1f s,  gap=%.2f%%)", res.status, res.solve_time, res.gap))

    println("\n  TREE STRUCTURE:")
    for t in T_B
        if haskey(res.splits, t)
            s = res.splits[t]
            println(@sprintf("    Node %d │ %-24s ≤ %.4f", t, s.feat, s.thr))
        else
            println(@sprintf("    Node %d │ no split", t))
        end
    end
    for t in T_L
        info = res.leaf_info[t]
        if info.active
            tag = info.idk ? "IDK ★" : info.pred
            println(@sprintf("    Leaf %d │ → %-11s  (n=%d)", t, tag, info.count))
        else
            println(@sprintf("    Leaf %d │ inactive", t))
        end
    end

    println()
    println("  PERFORMANCE  (★ = IDK leaf — referred for further tests)")
    println(@sprintf("    Coverage    : %.1f %%  (%d/%d patients classified)", res.coverage, n-res.n_idk, n))
    println(@sprintf("    IDK count   : %d patients (%.1f%%) referred", res.n_idk, 100-res.coverage))
    println(@sprintf("    Accuracy*   : %.2f %%  (among classified only)", res.acc_decided))
    println(@sprintf("    Accuracy†   : %.2f %%  (IDK counts as wrong)", res.acc_overall))
    println()
    println("    Confusion Matrix (classified patients only):")
    println("                    Pred Mal    Pred Ben")
    println(@sprintf("    Actual Mal(0)   %5d       %5d", res.TP, res.FN))
    println(@sprintf("    Actual Ben(1)   %5d       %5d", res.FP, res.TN))
end

# ── β sensitivity sweep ───────────────────────────────────────
println("\n" * "─"^62)
println("  β SENSITIVITY SWEEP")
println("─"^62)
println(@sprintf("  %-6s  %-10s  %-12s  %-12s  %-8s", "β", "Coverage%", "Acc*(classif)", "Acc†(all)", "IDK_n"))
println("  " * "─"^56)

sweep_results = []
for beta in BETA_VALUES
    print(@sprintf("  β=%-5.2f  solving…", beta))
    r = build_and_solve(beta)
    if r === nothing
        println("  NO SOLUTION")
        continue
    end
    println(@sprintf("  %6.1f%%     %7.2f%%       %7.2f%%    %4d  (%.1fs)",
        r.coverage, r.acc_decided, r.acc_overall, r.n_idk, r.solve_time))
    push!(sweep_results, Dict(
        "beta"        => beta,
        "coverage"    => r.coverage,
        "acc_decided" => r.acc_decided,
        "acc_overall" => r.acc_overall,
        "n_idk"       => r.n_idk,
        "leaves"      => Dict(t => Dict("pred"=>res.leaf_info[t].pred,
                                        "idk" =>res.leaf_info[t].idk,
                                        "count"=>res.leaf_info[t].count)
                              for t in T_L),
    ))
end

# ── Find optimal β ───────────────────────────────────────────
println("\n" * "─"^62)
println("  OPTIMAL β SELECTION")
println("─"^62)

# Prefer the highest β that still produces meaningful IDK abstention (n_idk > 0).
# This avoids picking solver-artifact winners among the β≥0.3 group where
# all results are essentially the same model (0 IDK leaves) and small accuracy
# differences are just MIP gap noise.
idk_results = filter(d -> d["n_idk"] > 0, sweep_results)
if !isempty(idk_results)
    # Among betas with abstention, pick highest β (least conservative elbow)
    opt_idx  = argmax([d["beta"] for d in idk_results])
    opt_beta = idk_results[opt_idx]["beta"]
    println(@sprintf("  ★ Best β with IDK active → β=%.2f  acc_all=%.2f%%  coverage=%.1f%%  IDK=%d",
        opt_beta,
        idk_results[opt_idx]["acc_overall"],
        idk_results[opt_idx]["coverage"],
        idk_results[opt_idx]["n_idk"]))
    println("    (higher β values all give 0 IDK — no meaningful abstention)")
else
    opt_beta = sweep_results[argmax([d["acc_overall"] for d in sweep_results])]["beta"]
    println("  No β produced IDK leaves — using highest acc_overall")
end
println(@sprintf("\n  → Recommended β = %.2f  (highest β with meaningful abstention)", opt_beta))
println("─"^62)

# Save results JSON for Python visualiser
out = Dict(
    "demo_beta"    => BETA_DEMO,
    "opt_beta"     => opt_beta,
    "n"            => n,
    "D"            => D,
    "p"            => P_FEAT,
    "sweep"        => sweep_results,
    "demo_splits"  => res !== nothing ? Dict(string(t) => Dict("feat"=>s.feat,"thr"=>s.thr)
                                             for (t,s) in res.splits) : Dict(),
    "demo_leaves"  => res !== nothing ? Dict(string(t) => Dict("pred"=>res.leaf_info[t].pred,
                                                                "idk"=>res.leaf_info[t].idk,
                                                                "count"=>res.leaf_info[t].count,
                                                                "active"=>res.leaf_info[t].active)
                                             for t in T_L) : Dict(),
    "demo_metrics" => res !== nothing ? Dict(
        "coverage"    => res.coverage,
        "acc_decided" => res.acc_decided,
        "acc_overall" => res.acc_overall,
        "n_idk"       => res.n_idk,
        "TP"          => res.TP, "TN"=>res.TN,
        "FP"          => res.FP, "FN"=>res.FN,
    ) : Dict(),
)
open(joinpath(@__DIR__, "oct_idk_results.json"), "w") do f
    JSON.print(f, out, 2)
end
println("\n  Results saved → oct_idk_results.json")
println("  Run:  python3 visualize_oct_idk.py")
println("="^62)
