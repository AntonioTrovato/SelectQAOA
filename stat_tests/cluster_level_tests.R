# ---------------------------------------------------------------------------
# Statistical comparison of the 6 simulators (cluster-level win counts)
#
# Per program:
#   1) Shapiro-Wilk normality test on each simulator's 10 win_count scores
#   2) If at least one simulator is NOT normal:
#         Kruskal-Wallis test on the 6 distributions
#         if significant (p < 0.05): Dunn's post-hoc test (Bonferroni) +
#            Vargha-Delaney A12 effect size for pairs with adj.p < 0.05
#         else: report "no statistical difference"
#      If ALL simulators are normal:
#         ANOVA on the 6 distributions
#         if significant (p < 0.05): Tukey HSD post-hoc test +
#            Cohen's d effect size for pairs with adj.p < 0.05
#         else: report "no statistical difference"
#
# Expects to be run from ./stat_tests, reading:
#   ../results/selectqaoa/cluster_level_comparison/wins_per_experiment.csv
# (produced by compare_simulators.py: columns program, experiment,
#  simulator, win_count)
# ---------------------------------------------------------------------------

library(data.table)
library(FSA)         # dunnTest
library(this.path)

setwd(this.dir())

INPUT_CSV <- file.path("..", "results", "selectqaoa", "cluster_level_comparison", "wins_per_experiment.csv")
OUTPUT_DIR <- file.path("..", "results", "selectqaoa", "cluster_level_comparison")
dir.create(OUTPUT_DIR, showWarnings = FALSE)
ALPHA <- 0.05

# --------------------------- effect size helpers ---------------------------

# Vargha-Delaney A12
a12 <- function(x, y) {
  nx <- length(x); ny <- length(y)
  sum(outer(x, y, FUN = function(xi, yj) ifelse(xi > yj, 1, ifelse(xi == yj, 0.5, 0)))) / (nx * ny)
}

# Cohen's d (pooled SD, unbiased)
cohen_d <- function(x, y) {
  nx <- length(x); ny <- length(y)
  s_pooled <- sqrt(((nx - 1) * var(x) + (ny - 1) * var(y)) / (nx + ny - 2))
  (mean(x) - mean(y)) / s_pooled
}

# --------------------------- data loading ---------------------------------

data <- as.data.frame(fread(INPUT_CSV))  # plain data.frame: avoids data.table's
# NSE column-shadowing on `[` subsetting
programs <- sort(unique(data$program))

# --------------------------- main loop -------------------------------------

for (prog in programs) {
  
  cat("\n==============================================\n")
  cat(sprintf("Program: %s\n", prog))
  cat("==============================================\n")
  
  sub <- data[data$program == prog, ]
  simulators <- sort(unique(sub$simulator))
  
  groups <- list()
  for (sim in simulators) {
    groups[[sim]] <- sub[sub$simulator == sim, ]$win_count
  }
  
  # ---------------- 1) Shapiro-Wilk normality test ----------------
  
  cat("\n--- Shapiro-Wilk normality test ---\n")
  normality <- data.frame(Simulator = character(), P_Value = numeric(),
                          Normal = logical(), stringsAsFactors = FALSE)
  
  for (sim in names(groups)) {
    vals <- groups[[sim]]
    sw_p <- NA_real_
    is_normal <- NA
    if (length(vals) >= 3 && length(unique(vals)) >= 2) {
      sw <- tryCatch(shapiro.test(vals), error = function(e) NULL)
      if (!is.null(sw)) {
        sw_p <- sw$p.value
        is_normal <- sw_p > ALPHA
      }
    }
    if (is.na(is_normal)) {
      cat(sprintf("%-20s : could not be tested (insufficient variance/data)\n", sim))
    } else {
      cat(sprintf("%-20s : W p-value = %.4f -> %s\n", sim, sw_p,
                  ifelse(is_normal, "NORMAL", "NOT NORMAL")))
    }
    normality <- rbind(normality, data.frame(Simulator = sim, P_Value = sw_p, Normal = is_normal))
  }
  write.csv(normality, file.path(OUTPUT_DIR, sprintf("%s_normality.csv", prog)), row.names = FALSE)
  
  all_normal <- all(!is.na(normality$Normal)) && all(normality$Normal)
  if (any(is.na(normality$Normal))) {
    cat("\n[NOTE] Normality could not be assessed for at least one simulator; defaulting to the non-parametric branch (Kruskal-Wallis).\n")
  }
  
  flat_values <- unlist(groups)
  group_labels <- rep(names(groups), times = sapply(groups, length))
  df_long <- data.frame(Value = flat_values, Group = as.factor(group_labels))
  
  posthoc_table <- NULL
  
  # ---------------- 2) significance test + post-hoc ----------------
  
  if (!all_normal) {
    
    # ---- Kruskal-Wallis ----
    cat("\n--- Kruskal-Wallis test (at least one simulator not normal) ---\n")
    kw <- kruskal.test(Value ~ Group, data = df_long)
    print(kw)
    
    if (!is.na(kw$p.value) && kw$p.value < ALPHA) {
      cat(sprintf("\nStatistically significant difference found (p = %.4g < 0.05). Running Dunn's post-hoc test (Bonferroni)...\n", kw$p.value))
      
      dunn <- dunnTest(Value ~ Group, data = df_long, method = "bonferroni")$res
      
      posthoc_table <- data.frame(Comparison = character(), P_unadj = numeric(),
                                  P_adj = numeric(), A12 = numeric(), stringsAsFactors = FALSE)
      for (i in seq_len(nrow(dunn))) {
        row <- dunn[i, ]
        comps <- trimws(unlist(strsplit(as.character(row$Comparison), "-")))
        g1 <- comps[1]; g2 <- comps[2]
        eff <- NA_real_
        if (!is.na(row$P.adj) && row$P.adj < ALPHA) {
          eff <- round(a12(groups[[g1]], groups[[g2]]), 3)
        }
        posthoc_table <- rbind(posthoc_table, data.frame(
          Comparison = row$Comparison,
          P_unadj = round(row$P.unadj, 4),
          P_adj = round(row$P.adj, 4),
          A12 = eff
        ))
      }
      cat("\n--- Dunn's post-hoc test (Bonferroni-adjusted), A12 shown only for adj.p < 0.05 ---\n")
      print(posthoc_table, row.names = FALSE)
      
    } else {
      cat(sprintf("\nNo statistically significant difference among simulators (Kruskal-Wallis p = %.4g >= 0.05).\n", kw$p.value))
    }
    
  } else {
    
    # ---- ANOVA ----
    cat("\n--- ANOVA (all simulators normal) ---\n")
    aov_model <- aov(Value ~ Group, data = df_long)
    print(summary(aov_model))
    p_anova <- summary(aov_model)[[1]][["Pr(>F)"]][1]
    
    if (!is.na(p_anova) && p_anova < ALPHA) {
      cat(sprintf("\nStatistically significant difference found (p = %.4g < 0.05). Running Tukey HSD post-hoc test...\n", p_anova))
      
      tukey <- TukeyHSD(aov_model)
      tukey_df <- as.data.frame(tukey$Group)
      comparisons <- rownames(tukey_df)
      
      # unadjusted pairwise p-values (no correction), for the "P_unadj" column
      pw <- pairwise.t.test(df_long$Value, df_long$Group, p.adjust.method = "none")$p.value
      
      posthoc_table <- data.frame(Comparison = character(), P_unadj = numeric(),
                                  P_adj = numeric(), Cohen_d = numeric(), stringsAsFactors = FALSE)
      for (comp in comparisons) {
        comps <- unlist(strsplit(comp, "-"))
        g1 <- comps[1]; g2 <- comps[2]
        p_adj <- tukey_df[comp, "p adj"]
        
        p_unadj <- NA_real_
        if (g1 %in% rownames(pw) && g2 %in% colnames(pw)) p_unadj <- pw[g1, g2]
        if (is.na(p_unadj) && g2 %in% rownames(pw) && g1 %in% colnames(pw)) p_unadj <- pw[g2, g1]
        
        eff <- NA_real_
        if (!is.na(p_adj) && p_adj < ALPHA) {
          eff <- round(cohen_d(groups[[g1]], groups[[g2]]), 3)
        }
        posthoc_table <- rbind(posthoc_table, data.frame(
          Comparison = comp,
          P_unadj = round(p_unadj, 4),
          P_adj = round(p_adj, 4),
          Cohen_d = eff
        ))
      }
      cat("\n--- Tukey HSD post-hoc test, Cohen's d shown only for adj.p < 0.05 ---\n")
      print(posthoc_table, row.names = FALSE)
      
    } else {
      cat(sprintf("\nNo statistically significant difference among simulators (ANOVA p = %.4g >= 0.05).\n", p_anova))
    }
  }
  
  if (!is.null(posthoc_table)) {
    write.csv(posthoc_table, file.path(OUTPUT_DIR, sprintf("%s_posthoc.csv", prog)), row.names = FALSE)
  }
}