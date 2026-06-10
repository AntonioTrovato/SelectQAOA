library(jsonlite)
library(coin)
library(FSA)
library(rstudioapi)
library(this.path)

setwd(this.dir())

# ── Effect size ──────────────────────────────────────────────────────────────

a12 <- function(x, y) {
  nx <- length(x)
  ny <- length(y)
  sum(outer(x, y, FUN = function(xi, yj) ifelse(xi > yj, 1, ifelse(xi == yj, 0.5, 0)))) / (nx * ny)
}

# ── Config ───────────────────────────────────────────────────────────────────

datasets <- c("elevator2", "elevator", "iofrol", "gsdtsr", "paintcontrol")

clusters_per_dataset <- c(
  gsdtsr      = 79,
  paintcontrol = 24,
  iofrol      = 443,
  elevator    = 881,
  elevator2   = 872
)

selectqaoa_configs <- c(
  "statevector_sim", "aer_sim", "fake_brisbane",   # was fake_vigo
  "depolarizing_sim/01", "depolarizing_sim/02", "depolarizing_sim/05"
)

igdec_qaoa_elevator2_configs <- c("ideal/qaoa_1/elevator_three", "noise/qaoa_1/elevator_one")
igdec_qaoa_configs            <- c("ideal/qaoa_1/", "noise/qaoa_1/")

# ── Main loop ────────────────────────────────────────────────────────────────

for (dataset in datasets) {
  cat(paste0("\n=========================\n", toupper(dataset), "\n=========================\n\n"))
  
  all_lists   <- list()
  names_lists <- c()
  n_clusters  <- clusters_per_dataset[dataset]
  
  # SelectQAOA
  for (config in selectqaoa_configs) {
    file_path <- paste0("../results/selectqaoa/", config, "/", dataset, ".csv")
    if (!file.exists(file_path)) next
    
    data       <- read.csv(file_path)
    qaoa_times <- fromJSON(data[["qpu_run_times.ms."]][1])
    
    expected <- n_clusters * 10
    if (length(qaoa_times) != expected) {
      warning(paste("Unexpected length for", dataset, config, ":", length(qaoa_times), "expected", expected))
      next
    }
    
    qaoa_means <- sapply(0:9, function(i) {
      start_idx <- i * n_clusters + 1
      end_idx   <- (i + 1) * n_clusters
      mean(qaoa_times[start_idx:end_idx])
    })
    
    key <- paste0("SelectQAOA_", config)
    all_lists[[key]] <- qaoa_means
    names_lists      <- c(names_lists, key)
  }
  
  # IgDec_QAOA
  if (dataset == "elevator2") {
    for (config in igdec_qaoa_elevator2_configs) {
      file_path <- paste0("../results/igdec_qaoa/", config, "/size_7/10/solution.csv")
      if (!file.exists(file_path)) next
      exec_times <- fromJSON(read.csv(file_path)[["execution_times"]][1])
      key <- paste0("IgDecQAOA_", config)
      all_lists[[key]] <- exec_times
      names_lists      <- c(names_lists, key)
    }
  } else {
    for (config in igdec_qaoa_configs) {
      if (dataset == "elevator") {
        file_path <- paste0("../results/igdec_qaoa/", config, "elevator_two/size_7/10/solution.csv")
      } else {
        file_path <- paste0("../results/igdec_qaoa/", config, dataset, "/size_7/10/solution.csv")
      }
      if (!file.exists(file_path)) next
      exec_times <- fromJSON(read.csv(file_path)[["execution_times"]][1])
      key <- paste0("IgDecQAOA_", gsub("/", "_", config), dataset)
      all_lists[[key]] <- exec_times
      names_lists      <- c(names_lists, key)
    }
  }
  
  # SelectQA + BootQA (gsdtsr and paintcontrol only)
  if (dataset %in% c("gsdtsr", "paintcontrol")) {
    file_selectqa <- paste0("../results/selectqa/", dataset, ".csv")
    if (file.exists(file_selectqa)) {
      val <- as.numeric(read.csv(file_selectqa)[["average_qpu_access_time.ms."]][1])
      all_lists[["SelectQA"]] <- rep(val, 10)
      names_lists <- c(names_lists, "SelectQA")
    }
    
    file_bootqa <- paste0("../results/bootqa/", dataset, ".csv")
    if (file.exists(file_bootqa)) {
      val <- fromJSON(read.csv(file_bootqa)[["exectution_times.ms."]][1])
      all_lists[["BootQA"]] <- val
      names_lists <- c(names_lists, "BootQA")
    }
  }
  
  # ── Summary statistics ─────────────────────────────────────────────────────
  
  cat("Summary statistics (mean ± sd):\n")
  for (key in names_lists) {
    vals <- as.numeric(all_lists[[key]]) / 1000
    cat(sprintf("  %s: %.2f ± %.2f s\n", key, mean(vals), sd(vals)))
  }
  cat("\n")
  
  # ── Statistical tests ──────────────────────────────────────────────────────
  
  if (dataset %in% c("gsdtsr", "paintcontrol")) {
    cat(">>> KRUSKAL-WALLIS + DUNN + A12 <<<\n\n")
    
    full_data <- data.frame(value = numeric(), group = character())
    for (i in seq_along(all_lists)) {
      full_data <- rbind(full_data, data.frame(
        value = all_lists[[i]],
        group = rep(names_lists[i], length(all_lists[[i]]))
      ))
    }
    
    kruskal <- kruskal.test(value ~ group, data = full_data)
    cat("Kruskal-Wallis: p =", format.pval(kruskal$p.value),
        ", H =", round(kruskal$statistic, 3), "\n\n")
    
    dunn        <- dunnTest(value ~ group, data = full_data, method = "bonferroni")
    dunn_result <- dunn$res
    
    for (row in 1:nrow(dunn_result)) {
      groups <- unlist(strsplit(dunn_result$Comparison[row], " - "))
      g1     <- all_lists[[groups[1]]]
      g2     <- all_lists[[groups[2]]]
      a12_val <- a12(g1, g2)
      sig    <- ifelse(dunn_result$P.adj[row] < 0.05, "*", "")
      cat(sprintf("[%s] vs [%s]: p = %s, adj p = %s, A12 = %.3f %s\n",
                  groups[1], groups[2],
                  format.pval(dunn_result$P.unadj[row], digits = 3),
                  format.pval(dunn_result$P.adj[row],   digits = 3),
                  a12_val, sig))
    }
    
  } else {
    # elevator, elevator2, iofrol — only 2 QAOA algorithms
    cat(">>> MANN-WHITNEY + A12 <<<\n\n")
    
    for (i in 1:(length(all_lists) - 1)) {
      for (j in (i + 1):length(all_lists)) {
        x  <- all_lists[[i]]
        y  <- all_lists[[j]]
        df <- data.frame(
          value = c(x, y),
          group = factor(c(rep("x", length(x)), rep("y", length(y))))
        )
        mw_test <- wilcox_test(value ~ group, data = df, distribution = "exact")
        a12_val <- a12(x, y)
        sig     <- ifelse(pvalue(mw_test) < 0.05, "*", "")
        cat(sprintf("[%s] vs [%s]: p = %s, A12 = %.3f %s\n",
                    names_lists[i], names_lists[j],
                    format.pval(pvalue(mw_test), digits = 4),
                    a12_val, sig))
      }
    }
  }
}