# =============================================================================
# Statistical tests on HV and IGD metrics
# For each simulator x program x metric:
#   - If at least one approach is non-normal → Kruskal-Wallis + Dunn (Bonferroni) + A12
#   - If all approaches are normal           → ANOVA + Tukey HSD (Bonferroni) + Cohen's d
#
# Add. Greedy has 1 real run → replicated 10 times (all equal values).
# =============================================================================

library(jsonlite)
library(FSA)        # dunnTest

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
json_path <- "../multi_obj_frontiers_eval.json"

programs <- c("flex", "grep", "gzip", "sed")

simulators <- list(
  list(key = "statevector_sim",     label = "Statevector (Ideal)"),
  list(key = "aer_sim",             label = "Aer Sim (Sampling Noise)"),
  list(key = "fake_brisbane",       label = "Fake Brisbane + Sampling Noise"),
  list(key = "depolarizing_sim/01", label = "Depolarizing 1% + Sampling Noise"),
  list(key = "depolarizing_sim/02", label = "Depolarizing 2% + Sampling Noise"),
  list(key = "depolarizing_sim/05", label = "Depolarizing 5% + Sampling Noise")
)

metrics <- c("hv", "igd")

stochastic_approaches <- list(
  list(key = "selectqa", label = "SelectQA"),
  list(key = "divga",    label = "DIV-GA"),
  list(key = "qaoa",     label = "SelectQAOA")
)

# -----------------------------------------------------------------------------
# EFFECT SIZE HELPERS
# -----------------------------------------------------------------------------

# Vargha-Delaney A12: P(x > y)
a12 <- function(x, y) {
  nx <- length(x)
  ny <- length(y)
  sum(outer(x, y, FUN = function(xi, yj) {
    ifelse(xi > yj, 1, ifelse(xi == yj, 0.5, 0))
  })) / (nx * ny)
}

# Pooled Cohen's d
cohen_d <- function(x, y) {
  s_pooled <- sqrt((var(x) + var(y)) / 2)
  if (s_pooled == 0) return(NA)
  (mean(x) - mean(y)) / s_pooled
}

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
data <- fromJSON(json_path)

# Extract 10 values for a given approach+metric.
# add_greedy stores a scalar → replicate 10 times.
get_values <- function(data, program, sim_key, approach_key, metric) {
  key   <- paste0(approach_key, "_", metric)
  entry <- data[[program]][[sim_key]][[key]]
  if (is.null(entry)) return(NULL)
  
  vals <- entry[[1]]   # first element of the [values, mean, std] triplet
  
  if (length(vals) == 1) {
    vals <- rep(as.numeric(vals), 10)   # degenerate single-run → replicate
  } else {
    vals <- as.numeric(vals)
  }
  return(vals)
}

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
for (sim in simulators) {
  sim_key   <- sim$key
  sim_label <- sim$label
  
  cat("\n")
  cat(strrep("=", 72), "\n")
  cat(sprintf("SIMULATOR: %s\n", sim_label))
  cat(strrep("=", 72), "\n")
  
  for (prog in programs) {
    cat(sprintf("\n  --- %s ---\n", toupper(prog)))
    
    for (metric in metrics) {
      cat(sprintf("\n    Metric: %s\n", toupper(metric)))
      
      # ---- Collect all 4 groups (3 stochastic + greedy) --------------------
      groups <- list()
      
      for (app in stochastic_approaches) {
        vals <- get_values(data, prog, sim_key, app$key, metric)
        if (!is.null(vals)) groups[[app$label]] <- vals
      }
      
      greedy_vals <- get_values(data, prog, sim_key, "add_greedy", metric)
      if (!is.null(greedy_vals)) groups[["Add. Greedy"]] <- greedy_vals
      
      if (length(groups) < 2) {
        cat("      Not enough groups – skipping.\n")
        next
      }
      
      # ---- Normality check (Shapiro-Wilk, alpha = 0.05) -------------------
      # Constant distributions (Add. Greedy, SelectQA) have var = 0.
      # shapiro.test would error or return p=1. We treat them as "normal"
      # since the test choice is driven by the stochastic approaches.
      normality_results <- sapply(names(groups), function(nm) {
        v <- groups[[nm]]
        if (var(v) == 0) return(TRUE)
        sw <- shapiro.test(v)
        return(sw$p.value > 0.05)
      })
      
      all_normal <- all(normality_results)
      
      if (all_normal) {
        # ==================================================================
        # PARAMETRIC: one-way ANOVA → Tukey HSD + Bonferroni + Cohen's d
        # ==================================================================
        cat("      All normal → ANOVA + Tukey HSD (Bonferroni) + Cohen's d\n\n")
        
        df_stat <- data.frame(
          value = unlist(groups),
          group = factor(rep(names(groups), times = sapply(groups, length)))
        )
        
        aov_model <- aov(value ~ group, data = df_stat)
        aov_p <- summary(aov_model)[[1]][["Pr(>F)"]][1]
        
        cat(sprintf("      ANOVA p-value: %.5f", aov_p))
        
        if (aov_p >= 0.05) {
          cat("  → not significant, no post-hoc needed.\n")
        } else {
          cat("  → significant → Tukey HSD with Bonferroni correction.\n\n")
          
          tukey    <- TukeyHSD(aov_model, conf.level = 0.95)
          tuk_df   <- as.data.frame(tukey$group)
          tuk_df$p.bonf <- p.adjust(tuk_df[["p adj"]], method = "bonferroni")
          
          cat(sprintf("      %-36s  %10s  %10s  %10s\n",
                      "Comparison", "p-value", "adj p (Bonf)", "Cohen's d"))
          cat(sprintf("      %s\n", strrep("-", 70)))
          
          group_names <- names(groups)
          
          for (comp in rownames(tuk_df)) {
            # TukeyHSD rownames: "B-A". Names may contain spaces/dots.
            # Strategy: try all pairs to find which one matches the rowname.
            g1_lab <- NA; g2_lab <- NA
            for (n1 in group_names) {
              for (n2 in group_names) {
                if (n1 == n2) next
                if (comp == paste0(n1, "-", n2)) { g1_lab <- n1; g2_lab <- n2; break }
              }
              if (!is.na(g1_lab)) break
            }
            
            p_raw  <- tuk_df[comp, "p adj"]
            p_bonf <- tuk_df[comp, "p.bonf"]
            d_val  <- if (!is.na(g1_lab)) {
              round(cohen_d(groups[[g1_lab]], groups[[g2_lab]]), 4)
            } else NA
            
            cat(sprintf("      %-36s  %10.5f  %10.5f  %10s\n",
                        comp, p_raw, p_bonf,
                        ifelse(is.na(d_val), "NA", sprintf("%.4f", d_val))))
          }
        }
        
      } else {
        # ==================================================================
        # NON-PARAMETRIC: Kruskal-Wallis → Dunn (Bonferroni) + A12
        # ==================================================================
        non_normal <- names(normality_results)[!normality_results]
        cat(sprintf("      Non-normal: %s\n", paste(non_normal, collapse = ", ")))
        cat("      → Kruskal-Wallis + Dunn (Bonferroni) + A12\n\n")
        
        flat_values  <- unlist(groups)
        group_labels <- factor(rep(names(groups), times = sapply(groups, length)))
        
        kw <- kruskal.test(flat_values ~ group_labels)
        cat(sprintf("      Kruskal-Wallis p-value: %.5f", kw$p.value))
        
        if (kw$p.value >= 0.05) {
          cat("  → not significant, no post-hoc needed.\n")
        } else {
          cat("  → significant → Dunn test with Bonferroni correction.\n\n")
          
          dunn <- dunnTest(flat_values ~ group_labels, method = "bonferroni")$res
          
          cat(sprintf("      %-36s  %10s  %10s  %10s\n",
                      "Comparison", "p-value", "adj p (Bonf)", "A12"))
          cat(sprintf("      %s\n", strrep("-", 70)))
          
          for (i in seq_len(nrow(dunn))) {
            row   <- dunn[i, ]
            comp  <- as.character(row$Comparison)  # format: "G1 - G2"
            parts <- strsplit(comp, " - ")[[1]]
            g1_lab <- trimws(parts[1])
            g2_lab <- trimws(parts[2])
            
            p_raw <- row$P.unadj
            p_adj <- row$P.adj
            
            a12_val <- if (g1_lab %in% names(groups) && g2_lab %in% names(groups)) {
              round(a12(groups[[g1_lab]], groups[[g2_lab]]), 4)
            } else NA
            
            cat(sprintf("      %-36s  %10.5f  %10.5f  %10s\n",
                        comp, p_raw, p_adj,
                        ifelse(is.na(a12_val), "NA", sprintf("%.4f", a12_val))))
          }
        }
      }
    }
  }
}

cat("\n")
cat(strrep("=", 72), "\n")
cat("Done.\n")
cat(strrep("=", 72), "\n")